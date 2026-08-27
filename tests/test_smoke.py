"""Property and unit tests for TalentMatch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matcher.assignment import brute_force_max_sum, greedy_assignment, optimal_assignment
from matcher.datagen import ARCHETYPES, generate, ground_truth_qualified
from matcher.engine import MatchEngine, build_dataset
from matcher.parsing import parse_job, parse_resume
from matcher.scoring import score_pair
from matcher.taxonomy import SkillTaxonomy


def test_smoke():
    tax = SkillTaxonomy()
    jobs, candidates = generate(tax, seed=42, n_jobs=4, n_random_candidates=20)
    assert len(jobs) == 4
    assert len(candidates) > 0
    parsed_jobs = [parse_job(j.job_id, j.text, tax) for j in jobs]
    parsed_candidates = [parse_resume(c.candidate_id, c.text, tax) for c in candidates]
    engine = MatchEngine(tax, parsed_jobs, parsed_candidates)

    labels = np.array([
        [ground_truth_qualified(tax, c.latent, j.latent) for j in jobs]
        for c in candidates
    ])

    calib = engine.calibrate(labels)
    assert calib.auc >= 0.5
    print(f"talentmatch smoke test passed: AUC={calib.auc:.3f}")


def test_security_engineer_archetype_is_wired():
    assert "security-engineer" in ARCHETYPES
    tax = SkillTaxonomy()
    core = ARCHETYPES["security-engineer"]["core"]
    nice = ARCHETYPES["security-engineer"]["nice"]
    for sid in core + nice:
        assert sid in tax.skills
    jobs, _ = generate(tax, seed=7, n_jobs=9, n_random_candidates=5)
    assert any(j.latent["archetype"] == "security-engineer" for j in jobs)


def test_implication_credit_decays_with_hops():
    tax = SkillTaxonomy()
    assert tax.credit("pytorch", "pytorch") == 1.0
    one = tax.credit("pytorch", "deep-learning")
    two = tax.credit("pytorch", "machine-learning")
    assert 0.5 < one < 1.0
    assert 0.0 < two < one
    assert tax.credit("machine-learning", "pytorch") == 0.0


def test_score_is_monotonic_when_a_must_skill_is_added():
    tax = SkillTaxonomy()
    jobs, candidates = generate(tax, seed=3, n_jobs=2, n_random_candidates=8)
    job = parse_job(jobs[0].job_id, jobs[0].text, tax, meta=jobs[0].latent)
    miss = next(c for c in candidates if c.latent.get("kind") == "nearmiss"
                and c.latent.get("planted_for") == job.job_id)
    cand = parse_resume(miss.candidate_id, miss.text, tax, meta=miss.latent)
    base = score_pair(cand, job, tax)
    from matcher.gap import _with_skill
    from matcher.parsing import DEFAULT_AS_OF

    lifted = score_pair(_with_skill(cand, miss.latent["missing_skill"], DEFAULT_AS_OF), job, tax)
    assert lifted.score >= base.score


def test_hungarian_matches_brute_force_on_tiny_boards():
    rng = np.random.default_rng(0)
    for n_c, n_j in ((2, 2), (3, 3), (3, 2), (2, 4)):
        scores = rng.random((n_c, n_j))
        cand_ids = [f"C{i}" for i in range(n_c)]
        job_ids = [f"J{i}" for i in range(n_j)]
        assigned = optimal_assignment(scores, cand_ids, job_ids)
        expected = brute_force_max_sum(scores)
        assert assigned.n_filled == min(n_c, n_j)
        assert assigned.total_score == pytest.approx(expected, abs=1e-3)


def test_assignment_is_a_matching():
    scores = np.array([
        [0.9, 0.8, 0.1],
        [0.85, 0.2, 0.3],
        [0.1, 0.7, 0.6],
    ])
    assigned = optimal_assignment(scores, ["A", "B", "C"], ["X", "Y", "Z"])
    cands = [p.candidate_id for p in assigned.pairs]
    jobs = [p.job_id for p in assigned.pairs]
    assert len(cands) == len(set(cands)) == 3
    assert len(jobs) == len(set(jobs)) == 3
    greedy = greedy_assignment(scores, ["A", "B", "C"], ["X", "Y", "Z"])
    assert assigned.total_score >= greedy.total_score - 1e-9


def test_engine_assign_attaches_match_results():
    ds = build_dataset(seed=11, n_jobs=3, n_candidates=12)
    engine = ds.engine()
    matching = engine.assign()
    greedy = engine.assign(greedy=True)
    assert matching.n_filled == 3
    assert len(matching.unmatched_jobs) == 0
    assert matching.total_score >= greedy.total_score - 1e-9
    seen_c, seen_j = set(), set()
    for pair in matching.pairs:
        assert pair.result is not None
        assert pair.result.candidate_id == pair.candidate_id
        assert pair.result.job_id == pair.job_id
        assert pair.candidate_id not in seen_c
        assert pair.job_id not in seen_j
        seen_c.add(pair.candidate_id)
        seen_j.add(pair.job_id)


def test_gap_names_the_planted_missing_skill():
    ds = build_dataset(seed=5, n_jobs=2, n_candidates=8)
    engine = ds.engine()
    job_id, (cand_id, missing) = next(iter(ds.planted_gap.items()))
    report = engine.gap(cand_id, job_id)
    assert missing in report.missing_must


if __name__ == "__main__":
    test_smoke()
    test_security_engineer_archetype_is_wired()
    test_implication_credit_decays_with_hops()
    test_score_is_monotonic_when_a_must_skill_is_added()
    test_hungarian_matches_brute_force_on_tiny_boards()
    test_assignment_is_a_matching()
    test_engine_assign_attaches_match_results()
    test_gap_names_the_planted_missing_skill()
    print("all tests passed")
