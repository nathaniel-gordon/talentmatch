# TalentMatch — Resume-to-Job Skill Matching Engine

> Match candidates to roles with precision, not keywords. TalentMatch uses an ontology skill graph, bidirectional bipartite matching, P(qualified) logistic calibration, and demographic blind redaction to rank candidates by true qualification fit — removing bias while surfacing the best matches.

## What TalentMatch Does

- **Ontology skill graph** — skill synonyms, hierarchies, and domain clusters; "Python" == "py"
- **Bipartite matching** — optimal assignment of candidates to roles under constraint
- **P(qualified) calibration** — logistic model converts raw scores to interpretable probabilities
- **Demographic blind redaction** — strips name, location, graduation year before scoring
- **Explanation output** — per-match skill gap analysis and matched/missing skill breakdown

## Architecture

```
Resumes + Job Descriptions
    └─> SkillExtractor      (ontology-grounded NER)
    └─> OntologyGraph       (skill synonyms, hierarchies)
    └─> BipartiteMatcher    (optimal assignment)
    └─> QualificationModel  (P(qualified) logistic calibration)
    └─> BlindRedactor       (demographic PII removal)
    └─> MatchReport         (ranked candidates + skill gaps)
```

## Quickstart

```bash
python examples/match_resumes.py    # match synthetic resume pool to job descriptions
```

## Test

```bash
python tests/test_smoke.py
```

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
