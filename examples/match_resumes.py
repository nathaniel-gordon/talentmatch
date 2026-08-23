"""Example: Ontology Graph & Bipartite Matching Engine for Resume-Job Alignment.

Run directly:
    python examples/match_resumes.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from matcher.datagen import generate, ground_truth_qualified
from matcher.engine import MatchEngine
from matcher.parsing import parse_job, parse_resume
from matcher.taxonomy import SkillTaxonomy

OUT = Path(__file__).parent.parent / "output"
OUT.mkdir(exist_ok=True)

# ── 1. Initialize Taxonomy & Ingest Synthetic Pool ────────────────────────────
print("=================================================================")
print("  TALENTMATCH — ONTOLOGY GRAPH & BIPARTITE MATCHING              ")
print("=================================================================")
print("[1/4] Loading skill taxonomy graph and generating talent pool ...")
tax = SkillTaxonomy()
print(f"      Taxonomy loaded: {len(tax.skills)} unique skill nodes, {len(tax.parents)} hierarchical relations")

jobs, candidates = generate(tax, n_jobs=8, n_random_candidates=40, seed=42)
print(f"      Generated {len(candidates)} synthetic candidate resumes and {len(jobs)} job descriptions")

# ── 2. Build Bipartite Matching Engine ────────────────────────────────────────
print("\n[2/4] Constructing matching matrix and fitting calibration model ...")
parsed_jobs = [parse_job(j.job_id, j.text, tax) for j in jobs]
parsed_candidates = [parse_resume(c.candidate_id, c.text, tax) for c in candidates]
engine = MatchEngine(tax, parsed_jobs, parsed_candidates)

labels = np.array([
    [ground_truth_qualified(tax, c.latent, j.latent) for j in jobs]
    for c in candidates
])

calib = engine.calibrate(labels)
print(f"      Out-of-sample Calibration: AUC={calib.auc:.3f} | Brier={calib.brier:.3f}")

# ── 3. Query Bipartite Alignments ─────────────────────────────────────────────
print("\n[3/4] Evaluating Top-3 Match Recommendations for Job 0:")
sample_job = parsed_jobs[0]
print(f"      Target Role: {sample_job.title}")

top_matches = engine.rank_candidates_for_job(sample_job.job_id, top=3)
for rank, res in enumerate(top_matches, 1):
    gap = engine.gap(res.candidate_id, sample_job.job_id)
    print(f"\n  [{rank}] Candidate {res.candidate_id} (Tier: {res.tier} | Score: {res.score:.2f})")
    print(f"      Skill Score: {res.skill_score:.2f} | Exp Score: {res.exp_score:.2f}")
    if gap.missing_must:
        print(f"      Missing Mandatory: {', '.join(gap.missing_must)}")
    if gap.matched_must:
        print(f"      Matched Skills: {', '.join(gap.matched_must)}")

print("\n[4/4] talentmatch example completed successfully.")
