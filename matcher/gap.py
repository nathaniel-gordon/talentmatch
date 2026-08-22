"""Gap analysis: what exactly is missing, and the cheapest way to fix it.

For every unsatisfied requirement we answer three questions a candidate
actually cares about:

1. *What is missing?*  - the requirement whose best evidence falls below the
   0.5 partial-evidence bar.
2. *How do I get it?*  - the shortest prerequisite-respecting route through the
   implication graph from something they already have (Dijkstra over
   time-to-competence), not a bare "learn Kubernetes".
3. *Is it worth it?*   - the score the skill would actually buy, measured by
   re-scoring the pair with the skill injected at one year of use.

Ranking the plan by score-lift per week turns the report into a study order
rather than a list of complaints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .parsing import DEFAULT_AS_OF, CandidateProfile, JobRequirements, SkillEvidence
from .scoring import MatchResult, ScoringWeights, score_pair
from .taxonomy import SkillTaxonomy, UpskillPath

# A hypothetical acquisition is credited with one year of use, which is enough
# to clear the "declared only" discount without pretending to deep expertise.
HYPOTHETICAL_MONTHS = 12


@dataclass
class GapItem:
    """One missing requirement plus its costed remedy."""

    skill_id: str
    display: str
    kind: str
    current_credit: float
    path: UpskillPath
    weeks: float
    score_lift: float

    @property
    def roi(self) -> float:
        """Score points bought per week of study."""
        return round(self.score_lift / self.weeks, 5) if self.weeks > 0 else 0.0

    @property
    def route(self) -> str:
        if not self.path.steps:
            return "already held"
        start = self.path.from_skill
        chain = " -> ".join(s.skill_id for s in self.path.steps)
        return f"{start} -> {chain}" if start else f"(from scratch) {chain}"


@dataclass
class GapReport:
    """Everything blocking one candidate from one job."""

    candidate_id: str
    job_id: str
    base_score: float
    projected_score: float
    items: list[GapItem] = field(default_factory=list)
    satisfied_must: list[str] = field(default_factory=list)

    @property
    def missing_must(self) -> list[str]:
        return [i.skill_id for i in self.items if i.kind == "must"]

    @property
    def missing_nice(self) -> list[str]:
        return [i.skill_id for i in self.items if i.kind == "nice"]

    @property
    def total_weeks(self) -> float:
        """Calendar weeks to close every MUST gap, sharing common prerequisites."""
        shared: dict[str, float] = {}
        for item in self.items:
            if item.kind != "must":
                continue
            for step in item.path.steps:
                shared[step.skill_id] = step.weeks
        return round(sum(shared.values()), 1)


def _with_skill(cand: CandidateProfile, skill_id: str, as_of: date) -> CandidateProfile:
    """Shallow clone of a profile with one skill added at one year of recent use."""
    clone = CandidateProfile(
        candidate_id=cand.candidate_id, name=cand.name, raw_text=cand.raw_text,
        roles=cand.roles, skills=dict(cand.skills), total_years=cand.total_years,
        seniority=cand.seniority, education_level=cand.education_level,
        school=cand.school, meta=cand.meta,
    )
    clone.skills[skill_id] = SkillEvidence(
        skill_id=skill_id, months=HYPOTHETICAL_MONTHS,
        last_used_ord=as_of.year * 12 + as_of.month,
    )
    return clone


def analyze_gap(
    cand: CandidateProfile,
    job: JobRequirements,
    tax: SkillTaxonomy,
    weights: ScoringWeights = ScoringWeights(),
    school_prestige: dict[str, float] | None = None,
    as_of: date = DEFAULT_AS_OF,
    include_nice: bool = True,
    base: MatchResult | None = None,
) -> GapReport:
    """Build the costed upskilling plan for one candidate/job pair."""
    base = base or score_pair(cand, job, tax, weights, school_prestige, as_of)
    held = cand.skill_ids

    items: list[GapItem] = []
    for match in base.skill_matches:
        if match.satisfied or (match.kind == "nice" and not include_nice):
            continue
        path = tax.upskill_path(held, match.skill_id)
        lifted = score_pair(_with_skill(cand, match.skill_id, as_of), job, tax,
                            weights, school_prestige, as_of)
        items.append(GapItem(
            skill_id=match.skill_id, display=tax.display(match.skill_id), kind=match.kind,
            current_credit=match.effective, path=path, weeks=max(path.weeks, 1.0),
            score_lift=round(lifted.score - base.score, 4),
        ))

    # Best return on study time first; must-haves always outrank nice-to-haves.
    items.sort(key=lambda i: (i.kind != "must", -i.roi, i.skill_id))

    projected = cand
    for item in items:
        if item.kind == "must":
            projected = _with_skill(projected, item.skill_id, as_of)
    projected_score = score_pair(projected, job, tax, weights, school_prestige, as_of).score

    return GapReport(
        candidate_id=cand.candidate_id, job_id=job.job_id, base_score=base.score,
        projected_score=projected_score, items=items,
        satisfied_must=[m.skill_id for m in base.skill_matches
                        if m.kind == "must" and m.satisfied],
    )
