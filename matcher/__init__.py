"""TalentMatch: ontology-grounded resume-to-job matching."""

from .assignment import Assignment, AssignedPair, greedy_assignment, optimal_assignment
from .engine import CalibrationReport, Dataset, MatchEngine, PlantedEval, build_dataset
from .scoring import MatchResult, ScoringWeights, score_pair
from .taxonomy import SkillTaxonomy

__all__ = [
    "Assignment",
    "AssignedPair",
    "CalibrationReport",
    "Dataset",
    "MatchEngine",
    "MatchResult",
    "PlantedEval",
    "ScoringWeights",
    "SkillTaxonomy",
    "build_dataset",
    "greedy_assignment",
    "optimal_assignment",
    "score_pair",
]
