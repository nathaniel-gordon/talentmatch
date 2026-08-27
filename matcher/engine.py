"""Bidirectional matching engine: shortlists, matrix, calibration and tiers.

The engine holds two parsed views of every candidate - the raw resume and the
blind-redacted one - so any query can be answered either way without
re-parsing. Scores are computed once into a dense candidates x jobs matrix,
which makes both directions (best candidates for a job, best jobs for a
candidate) a single argsort.

Raw composite scores are not probabilities. A one-feature logistic model fitted
on a held-out split of pairs maps them onto P(qualified), which is what the
strong/possible/weak tiers threshold. Calibration quality (AUC, average
precision, Brier) is reported on the split the model never saw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .assignment import Assignment, greedy_assignment, optimal_assignment
from .bias import blind_redact
from .datagen import SCHOOL_PRESTIGE, SyntheticCandidate, SyntheticJob, generate, ground_truth_qualified
from .gap import GapReport, analyze_gap
from .parsing import DEFAULT_AS_OF, CandidateProfile, JobRequirements, parse_job, parse_resume
from .scoring import MatchResult, ScoringWeights, score_pair
from .taxonomy import SkillTaxonomy

TIER_THRESHOLDS: tuple[float, float] = (0.55, 0.20)  # strong >= .55, possible >= .20


@dataclass
class CalibrationReport:
    """Quality of the raw-score -> P(qualified) mapping, measured out of sample."""

    n_calibration: int
    n_holdout: int
    positive_rate: float
    auc: float
    average_precision: float
    brier: float
    brier_uncalibrated: float
    tier_counts: dict[str, int] = field(default_factory=dict)
    tier_precision: dict[str, float] = field(default_factory=dict)


@dataclass
class PlantedEval:
    """How well the engine recovers the planted ideal candidate per job."""

    n_jobs: int
    top1_hits: int
    mean_reciprocal_rank: float
    ideal_mean_score: float
    pool_mean_score: float

    @property
    def top1_accuracy(self) -> float:
        return round(self.top1_hits / self.n_jobs, 4)


def _tier(prob: float, thresholds: tuple[float, float] = TIER_THRESHOLDS) -> str:
    strong, possible = thresholds
    return "strong" if prob >= strong else "possible" if prob >= possible else "weak"


class MatchEngine:
    """Scores every candidate against every job, both plainly and blind."""

    def __init__(
        self,
        tax: SkillTaxonomy,
        jobs: list[JobRequirements],
        candidates: list[CandidateProfile],
        blind_candidates: list[CandidateProfile] | None = None,
        weights: ScoringWeights = ScoringWeights(),
        school_prestige: dict[str, float] | None = None,
        as_of: date = DEFAULT_AS_OF,
    ) -> None:
        if blind_candidates is not None and len(blind_candidates) != len(candidates):
            raise ValueError("blind_candidates must align 1:1 with candidates")
        self.tax = tax
        self.jobs = jobs
        self.candidates = candidates
        self.blind_candidates = blind_candidates or candidates
        self.weights = weights
        self.school_prestige = school_prestige or {}
        self.as_of = as_of
        self.job_index = {j.job_id: i for i, j in enumerate(jobs)}
        self.cand_index = {c.candidate_id: i for i, c in enumerate(candidates)}
        self._cache: dict[bool, list[list[MatchResult]]] = {}
        self._model: LogisticRegression | None = None

    # ------------------------------------------------------------------ core
    def _results(self, blind: bool) -> list[list[MatchResult]]:
        if blind not in self._cache:
            pool = self.blind_candidates if blind else self.candidates
            self._cache[blind] = [
                [score_pair(c, j, self.tax, self.weights, self.school_prestige,
                            self.as_of, blind=blind) for j in self.jobs]
                for c in pool
            ]
            self._apply_tiers(blind)
        return self._cache[blind]

    def result(self, candidate_id: str, job_id: str, blind: bool = False) -> MatchResult:
        return self._results(blind)[self.cand_index[candidate_id]][self.job_index[job_id]]

    def score_matrix(self, blind: bool = False, qualification_only: bool = False) -> np.ndarray:
        """Dense candidates x jobs score matrix."""
        rows = self._results(blind)
        attr = "qual_score" if qualification_only else "score"
        return np.array([[getattr(r, attr) for r in row] for row in rows], dtype=float)

    def rank_candidates_for_job(self, job_id: str, top: int = 10,
                                blind: bool = False) -> list[MatchResult]:
        """Shortlist for a posting (talent search direction)."""
        j = self.job_index[job_id]
        rows = self._results(blind)
        ranked = sorted((row[j] for row in rows), key=lambda r: (-r.score, r.candidate_id))
        return ranked[:top]

    def rank_jobs_for_candidate(self, candidate_id: str, top: int = 10,
                                blind: bool = False) -> list[MatchResult]:
        """Shortlist for a person (job search direction)."""
        row = self._results(blind)[self.cand_index[candidate_id]]
        return sorted(row, key=lambda r: (-r.score, r.job_id))[:top]

    def assign(self, blind: bool = False, qualification_only: bool = False,
               greedy: bool = False) -> Assignment:
        """Exclusive candidate-to-job matching (Hungarian, or greedy baseline).

        Independent shortlists can recommend the same person for every posting.
        This solves the round as a bipartite matching: each candidate and each
        job appears in at most one pair, maximising the sum of scores.
        """
        matrix = self.score_matrix(blind=blind, qualification_only=qualification_only)
        cand_ids = [c.candidate_id for c in self.candidates]
        job_ids = [j.job_id for j in self.jobs]
        solver = greedy_assignment if greedy else optimal_assignment
        assignment = solver(matrix, cand_ids, job_ids)
        rows = self._results(blind)
        for pair in assignment.pairs:
            ci = self.cand_index[pair.candidate_id]
            ji = self.job_index[pair.job_id]
            pair.result = rows[ci][ji]
        return assignment

    def gap(self, candidate_id: str, job_id: str, blind: bool = False) -> GapReport:
        pool = self.blind_candidates if blind else self.candidates
        cand = pool[self.cand_index[candidate_id]]
        job = self.jobs[self.job_index[job_id]]
        return analyze_gap(cand, job, self.tax, self.weights, self.school_prestige,
                           self.as_of, base=self.result(candidate_id, job_id, blind))

    # ----------------------------------------------------------- calibration
    def calibrate(self, labels: np.ndarray, calib_fraction: float = 0.4,
                  seed: int = 42) -> CalibrationReport:
        """Fit score -> P(qualified) on a candidate-disjoint split and score it.

        The split is by *candidate*, not by pair: splitting pairs would leak a
        candidate's other rows into training and inflate the held-out numbers.
        """
        scores = self.score_matrix()
        if labels.shape != scores.shape:
            raise ValueError(f"labels shape {labels.shape} != scores shape {scores.shape}")
        rng = np.random.default_rng(seed)
        n_cand = scores.shape[0]
        order = rng.permutation(n_cand)
        n_calib = max(2, int(round(calib_fraction * n_cand)))
        calib_rows, hold_rows = order[:n_calib], order[n_calib:]

        x_tr = scores[calib_rows].reshape(-1, 1)
        y_tr = labels[calib_rows].ravel()
        x_te = scores[hold_rows].reshape(-1, 1)
        y_te = labels[hold_rows].ravel()
        if len(np.unique(y_tr)) < 2:
            raise ValueError("calibration split contains a single class; change the seed")

        self._model = LogisticRegression(C=10.0, max_iter=1000).fit(x_tr, y_tr)
        p_te = self._model.predict_proba(x_te)[:, 1]
        for blind in list(self._cache):
            self._apply_tiers(blind)

        tiers = np.array([_tier(p) for p in p_te])
        counts = {t: int((tiers == t).sum()) for t in ("strong", "possible", "weak")}
        precision = {
            t: round(float(y_te[tiers == t].mean()), 4) if counts[t] else 0.0
            for t in counts
        }
        return CalibrationReport(
            n_calibration=int(x_tr.shape[0]),
            n_holdout=int(x_te.shape[0]),
            positive_rate=round(float(y_te.mean()), 4),
            auc=round(float(roc_auc_score(y_te, p_te)), 4),
            average_precision=round(float(average_precision_score(y_te, p_te)), 4),
            brier=round(float(brier_score_loss(y_te, p_te)), 4),
            brier_uncalibrated=round(float(brier_score_loss(y_te, x_te.ravel())), 4),
            tier_counts=counts,
            tier_precision=precision,
        )

    def _apply_tiers(self, blind: bool) -> None:
        """Attach probability and tier to cached results (no-op before calibration)."""
        rows = self._cache[blind]
        if self._model is None:
            for row in rows:
                for r in row:
                    r.probability, r.tier = None, "unrated"
            return
        flat = np.array([[r.score] for row in rows for r in row])
        probs = self._model.predict_proba(flat)[:, 1]
        i = 0
        for row in rows:
            for r in row:
                r.probability = round(float(probs[i]), 4)
                r.tier = _tier(r.probability)
                i += 1

    # ------------------------------------------------------------ evaluation
    def evaluate_planted(self, planted: dict[str, str]) -> PlantedEval:
        """Rank of each job's planted ideal candidate in its own shortlist."""
        hits, rr, ideal_scores = 0, [], []
        for job_id, cand_id in planted.items():
            ranked = self.rank_candidates_for_job(job_id, top=len(self.candidates))
            pos = next(i for i, r in enumerate(ranked) if r.candidate_id == cand_id)
            hits += int(pos == 0)
            rr.append(1.0 / (pos + 1))
            ideal_scores.append(ranked[pos].score)
        matrix = self.score_matrix()
        return PlantedEval(
            n_jobs=len(planted), top1_hits=hits,
            mean_reciprocal_rank=round(float(np.mean(rr)), 4),
            ideal_mean_score=round(float(np.mean(ideal_scores)), 4),
            pool_mean_score=round(float(matrix.mean()), 4),
        )


@dataclass
class Dataset:
    """Generated corpus plus everything parsed from it."""

    tax: SkillTaxonomy
    raw_jobs: list[SyntheticJob]
    raw_candidates: list[SyntheticCandidate]
    jobs: list[JobRequirements]
    candidates: list[CandidateProfile]
    blind_candidates: list[CandidateProfile]
    labels: np.ndarray
    planted_ideal: dict[str, str]
    planted_gap: dict[str, tuple[str, str]]
    proxy_by_candidate: dict[str, str]

    def engine(self, weights: ScoringWeights = ScoringWeights(),
               as_of: date = DEFAULT_AS_OF) -> MatchEngine:
        return MatchEngine(self.tax, self.jobs, self.candidates, self.blind_candidates,
                           weights, SCHOOL_PRESTIGE, as_of)


def build_dataset(seed: int = 42, n_jobs: int = 16, n_candidates: int = 120,
                  as_of: date = DEFAULT_AS_OF) -> Dataset:
    """Generate, parse and label the whole corpus for ``seed``."""
    tax = SkillTaxonomy()
    raw_jobs, raw_cands = generate(tax, seed=seed, n_jobs=n_jobs,
                                   n_random_candidates=n_candidates, as_of=as_of)
    jobs = [parse_job(j.job_id, j.text, tax, meta=j.latent) for j in raw_jobs]
    cands = [parse_resume(c.candidate_id, c.text, tax, as_of, meta=c.latent) for c in raw_cands]
    blind = [parse_resume(c.candidate_id, blind_redact(c.text), tax, as_of, meta=c.latent)
             for c in raw_cands]
    labels = np.array(
        [[ground_truth_qualified(tax, c.latent, j.latent) for j in raw_jobs] for c in raw_cands],
        dtype=int,
    )
    planted_ideal = {c.latent["planted_for"]: c.candidate_id
                     for c in raw_cands if c.latent["kind"] == "ideal"}
    planted_gap = {c.latent["planted_for"]: (c.candidate_id, c.latent["missing_skill"])
                   for c in raw_cands if c.latent["kind"] == "nearmiss"}
    proxy = {c.candidate_id: c.latent["group"] for c in raw_cands}
    return Dataset(tax, raw_jobs, raw_cands, jobs, cands, blind, labels,
                   planted_ideal, planted_gap, proxy)
