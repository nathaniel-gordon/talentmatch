# TalentMatch — Skill Taxonomy Ontology Graph & Bipartite Candidate Matching

TalentMatch is an automated recruiting intelligence engine that matches candidates to job requisitions using a **Hierarchical Skill Taxonomy Ontology Graph** and **Bipartite Optimal Assignment**. It eliminates keyword mismatching by understanding semantic skill hierarchies (e.g. `PyTorch` is a child of `Deep Learning` which is a child of `Machine Learning`) and calibrates candidate fit into probabilistic $P(	ext{qualified})$ scores.

## Key Features

- **Skill Ontology Graph**: Maps synonyms, parent-child relationships, and complementary tech stacks.
- **Bipartite Optimal Matching**: Solves global candidate-to-requisition assignments under capacity constraints.
- **Demographic Blind Redaction**: Automatically redacts demographic, collegiate, and geographic markers prior to scoring to enforce fairness.

## Usage

```bash
# Match candidate resumes against open job requisitions
python -m sbm --resumes output/demo_resumes/ --jobs output/demo_jobs/
```

## Tests

```bash
pytest tests/ -v
```
