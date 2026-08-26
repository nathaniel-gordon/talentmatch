"""Smoke test for talentmatch."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matcher.datagen import generate, ground_truth_qualified
from matcher.engine import MatchEngine
from matcher.parsing import parse_job, parse_resume
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

if __name__ == "__main__":
    test_smoke()
