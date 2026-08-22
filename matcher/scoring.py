"""Composite, fully decomposable match score.

The score is a weighted sum of six independent components, each in [0, 1]:

===================  ==========================================================
must_coverage        weighted share of MUST-have skills backed by evidence
nice_bonus           same, over NICE-to-have skills
seniority_fit        distance between posted level and candidate level
experience_fit       tenure against the stated minimum
freshness            recency of the must-have skills the candidate does have
education_fit        degree level against the stated floor
===================  ==========================================================

A seventh weight, ``prestige``, is **zero by default**. It exists so the bias
audit can demonstrate what happens when a school-ranking file is wired into a
hiring score, and so blind mode can be shown to neutralise it. Turning it on
rescales the six qualification weights, keeping the total at 1.0.

Every component carries the arithmetic that produced it, so a recruiter can be
shown "you lost 0.11 on seniority fit, here is why" rather than a bare number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .parsing import (
    DEFAULT_AS_OF,
    CandidateProfile,
    JobRequirements,
    SENIORITY_ORDER,
    SkillEvidence,
)
from .taxonomy import SkillTaxonomy

NEUTRAL_PRESTIGE = 0.5


@dataclass(frozen=True)
class ScoringWeights:
    """Component weights. The six qualification weights must sum to 1.0."""

    must_coverage: float = 0.42
    nice_bonus: float = 0.14
    seniority_fit: float = 0.14
    experience_fit: float = 0.14
    freshness: float = 0.09
    education_fit: float = 0.07
    prestige: float = 0.0

    QUALIFICATION_FIELDS = ("must_coverage", "nice_bonus", "seniority_fit",
                            "experience_fit", "freshness", "education_fit")

    def __post_init__(self) -> None:
        total = sum(getattr(self, f) for f in self.QUALIFICATION_FIELDS)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"qualification weights must sum to 1.0, got {total:.6f}")
        if not 0.0 <= self.prestige < 1.0:
            raise ValueError("prestige weight must be in [0, 1)")


@dataclass
class SkillMatch:
    """How one requirement was (or was not) satisfied."""

    skill_id: str
    kind: str
    credit: float
    effective: float
    via: str | None
    years: float
    freshness: float
    evidence: str

    @property
    def satisfied(self) -> bool:
        """0.5 is the boundary between 'partial evidence' and 'a real gap'."""
        return self.effective >= 0.5


@dataclass
class Component:
    """One interpretable term of the total score."""

    name: str
    raw: float
    weight: float
    detail: str

    @property
    def contribution(self) -> float:
        return round(self.raw * self.weight, 4)

    @property
    def lost(self) -> float:
        """Points this component gave up against a perfect candidate."""
        return round((1.0 - self.raw) * self.weight, 4)


@dataclass
class MatchResult:
    """A scored candidate/job pair with its full derivation."""

    candidate_id: str
    job_id: str
    score: float
    qual_score: float
    components: list[Component] = field(default_factory=list)
    skill_matches: list[SkillMatch] = field(default_factory=list)
    blind: bool = False
    probability: float | None = None
    tier: str = "unrated"

    def component(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def missing_must(self) -> list[str]:
        return [m.skill_id for m in self.skill_matches if m.kind == "must" and not m.satisfied]


def proficiency(ev: SkillEvidence | None) -> float:
    """Map dated experience to a [0.70, 1.0] confidence multiplier.

    A skill listed but never tied to a dated role keeps the 0.70 floor: it is
    a claim, not evidence. Four years is treated as full command; beyond that
    extra years say more about tenure (already its own component) than depth.
    """
    if ev is None:
        return 0.0
    return 0.70 + 0.30 * min(ev.years, 4.0) / 4.0


def _best_evidence(
    cand: CandidateProfile, want: str, tax: SkillTaxonomy
) -> tuple[float, float, str | None, SkillEvidence | None]:
    """Strongest (effective, raw_credit, source_skill, evidence) for one requirement."""
    best = (0.0, 0.0, None, None)
    for sid, ev in cand.skills.items():
        credit = tax.credit(sid, want)
        if credit <= 0.0:
            continue
        eff = credit * proficiency(ev)
        if eff > best[0]:
            best = (eff, credit, sid, ev)
    return best  # type: ignore[return-value]


def _match_skills(
    cand: CandidateProfile, wants: list[str], kind: str, tax: SkillTaxonomy, as_of: date
) -> list[SkillMatch]:
    out: list[SkillMatch] = []
    for want in wants:
        eff, credit, via, ev = _best_evidence(cand, want, tax)
        if ev is None:
            out.append(SkillMatch(want, kind, 0.0, 0.0, None, 0.0, 0.0, "no evidence found"))
            continue
        snippet = ev.mentions[0].snippet if ev.mentions else ""
        note = "direct evidence" if via == want else f"implied by {tax.display(via or '')}"
        if ev.declared_only:
            note += " (listed only, no dated role)"
        out.append(
            SkillMatch(want, kind, round(credit, 4), round(eff, 4), via, ev.years,
                       round(ev.freshness(as_of), 4), f"{note}: ...{snippet}...")
        )
    return out


def seniority_fit(cand_level: int, job_level: int) -> tuple[float, str]:
    """Under-levelling is punished hard, over-levelling only mildly.

    One rung above the posting is normal (people apply up); two or more starts
    to signal a compensation and retention mismatch, which is a real cost but
    a much smaller one than being unable to do the job.
    """
    gap = cand_level - job_level
    if gap >= 0:
        raw = max(0.0, 1.0 - 0.12 * max(0, gap - 1))
        word = "level match" if gap <= 1 else f"{gap} levels above posting"
    else:
        raw = max(0.0, 1.0 + 0.35 * gap)
        word = f"{-gap} level(s) below posting"
    c = SENIORITY_ORDER[max(0, min(len(SENIORITY_ORDER) - 1, cand_level))]
    j = SENIORITY_ORDER[max(0, min(len(SENIORITY_ORDER) - 1, job_level))]
    return raw, f"candidate {c} vs posting {j}: {word}"


def experience_fit(cand_years: float, min_years: float) -> tuple[float, str]:
    """Full marks at or above the bar; linear decay to zero at ~29% of the bar."""
    req = max(min_years, 0.5)
    ratio = cand_years / req
    raw = 1.0 if ratio >= 1.0 else max(0.0, 1.0 - 1.4 * (1.0 - ratio))
    return raw, f"{cand_years:.1f} yrs vs {min_years:.0f} required ({ratio * 100:.0f}% of bar)"


def education_fit(cand_level: int, job_level: int) -> tuple[float, str]:
    """Below the stated floor costs 0.30 per missing degree level."""
    from .parsing import EDUCATION_ORDER

    if job_level <= 0:
        return 1.0, "no degree requirement stated"
    raw = 1.0 if cand_level >= job_level else max(0.0, 1.0 - 0.30 * (job_level - cand_level))
    return raw, (f"{EDUCATION_ORDER[cand_level]} vs required "
                 f"{EDUCATION_ORDER[min(job_level, len(EDUCATION_ORDER) - 1)]}")


def score_pair(
    cand: CandidateProfile,
    job: JobRequirements,
    tax: SkillTaxonomy,
    weights: ScoringWeights = ScoringWeights(),
    school_prestige: dict[str, float] | None = None,
    as_of: date = DEFAULT_AS_OF,
    blind: bool = False,
) -> MatchResult:
    """Score one candidate against one job and return the full derivation."""
    must_matches = _match_skills(cand, job.must, "must", tax, as_of)
    nice_matches = _match_skills(cand, job.nice, "nice", tax, as_of)

    must_raw = (sum(m.effective for m in must_matches) / len(must_matches)) if must_matches else 1.0
    covered = [m for m in must_matches if m.effective > 0]
    must_detail = (f"{sum(1 for m in must_matches if m.satisfied)}/{len(must_matches)} "
                   f"must-haves satisfied") if must_matches else "no must-haves listed"

    if nice_matches:
        nice_raw = sum(m.effective for m in nice_matches) / len(nice_matches)
        nice_detail = f"{sum(1 for m in nice_matches if m.satisfied)}/{len(nice_matches)} nice-to-haves"
    else:
        nice_raw, nice_detail = 1.0, "no nice-to-haves listed (neutral)"

    if covered:
        fresh_raw = sum(m.freshness * m.credit for m in covered) / sum(m.credit for m in covered)
        stale = [m.skill_id for m in covered if m.freshness < 0.6]
        fresh_detail = ("all covered must-haves used recently" if not stale
                        else "stale: " + ", ".join(tax.display(s) for s in stale[:4]))
    else:
        fresh_raw, fresh_detail = 0.0, "no must-have evidence to date"

    sen_raw, sen_detail = seniority_fit(cand.seniority, job.seniority)
    exp_raw, exp_detail = experience_fit(cand.total_years, job.min_years)
    edu_raw, edu_detail = education_fit(cand.education_level, job.education_level)

    scale = 1.0 - weights.prestige
    components = [
        Component("must_coverage", round(must_raw, 4), weights.must_coverage * scale, must_detail),
        Component("nice_bonus", round(nice_raw, 4), weights.nice_bonus * scale, nice_detail),
        Component("seniority_fit", round(sen_raw, 4), weights.seniority_fit * scale, sen_detail),
        Component("experience_fit", round(exp_raw, 4), weights.experience_fit * scale, exp_detail),
        Component("freshness", round(fresh_raw, 4), weights.freshness * scale, fresh_detail),
        Component("education_fit", round(edu_raw, 4), weights.education_fit * scale, edu_detail),
    ]
    qual = sum(c.raw * getattr(weights, c.name) for c in components)

    if weights.prestige > 0.0:
        table = school_prestige or {}
        prestige_raw = table.get(cand.school, NEUTRAL_PRESTIGE)
        label = cand.school or "unknown"
        components.append(Component(
            "prestige", round(prestige_raw, 4), weights.prestige,
            f"school ranking for {label}"
            + ("" if cand.school in table else " (not in ranking file -> neutral)"),
        ))

    return MatchResult(
        candidate_id=cand.candidate_id,
        job_id=job.job_id,
        score=round(sum(c.contribution for c in components), 4),
        qual_score=round(qual, 4),
        components=components,
        skill_matches=must_matches + nice_matches,
        blind=blind,
    )
