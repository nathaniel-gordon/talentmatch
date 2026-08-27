"""Seeded synthetic talent pool and job board with planted ground truth.

Design intent
-------------
Everything the matcher sees is *text*. Everything the evaluation uses is the
*latent* profile that produced the text. Because the two are separated, the
reported AUC/top-1 numbers measure a real recovery problem — the parser has to
reconstruct skills (often written through aliases), per-skill years (from role
dates) and seniority before the scorer can rank anything.

Three planted structures make the smoke test sharp:

* ``ideal``    - one candidate per job that satisfies every requirement.
* ``nearmiss`` - the same candidate minus exactly one must-have skill *and*
  everything that would imply it, so gap analysis has a single correct answer.
* a demographic proxy column whose only causal link to the score is school
  prestige, so the disparate-impact audit measures a bias we injected on
  purpose and blind mode can be shown to remove it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

from .parsing import DEFAULT_AS_OF, EDUCATION_LEVEL
from .taxonomy import SkillTaxonomy

# Neutral job titles: the resume renderer must not smuggle a skill name into a
# role title, or the "candidate is missing exactly one skill" plant leaks.
ARCHETYPES: dict[str, dict] = {
    "ml-engineer": {
        "title": "Applied Scientist",
        "domain": "customer intelligence",
        "core": ["python", "pytorch", "machine-learning", "mlops", "docker",
                 "kubernetes", "aws", "sql", "feature-engineering", "model-deployment"],
        "nice": ["kafka", "spark", "tensorflow", "airflow", "observability"],
    },
    "data-engineer": {
        "title": "Platform Engineer",
        "domain": "petabyte-scale batch and streaming",
        "core": ["python", "spark", "airflow", "kafka", "sql", "snowflake",
                 "dbt", "aws", "etl", "data-modeling"],
        "nice": ["scala", "kubernetes", "terraform", "postgresql", "observability"],
    },
    "backend-engineer": {
        "title": "Backend Engineer",
        "domain": "high-throughput transactional",
        "core": ["java", "spring", "microservices", "rest-api", "postgresql",
                 "docker", "kubernetes", "system-design", "testing"],
        "nice": ["golang", "kafka", "redis", "grpc", "aws"],
    },
    "frontend-engineer": {
        "title": "Frontend Engineer",
        "domain": "customer-facing web",
        "core": ["javascript", "typescript", "react", "html-css", "testing",
                 "rest-api", "graphql", "nodejs"],
        "nice": ["vue", "angular", "docker", "ci-cd", "observability"],
    },
    "data-scientist": {
        "title": "Decision Scientist",
        "domain": "growth measurement",
        "core": ["python", "pandas", "numpy", "scikit-learn", "statistics",
                 "sql", "machine-learning", "time-series", "experimentation"],
        "nice": ["pytorch", "spark", "nlp", "aws", "airflow"],
    },
    "nlp-engineer": {
        "title": "Research Engineer",
        "domain": "language understanding",
        "core": ["python", "pytorch", "transformers", "nlp", "llm", "rag",
                 "deep-learning", "docker", "information-retrieval"],
        "nice": ["kubernetes", "aws", "mlops", "elasticsearch", "computer-vision"],
    },
    "devops-engineer": {
        "title": "Reliability Engineer",
        "domain": "multi-region platform",
        "core": ["terraform", "kubernetes", "docker", "ci-cd", "aws", "linux",
                 "observability", "bash", "security"],
        "nice": ["golang", "python", "azure", "gcp", "system-design"],
    },
    "fullstack-engineer": {
        "title": "Product Engineer",
        "domain": "end-to-end product",
        "core": ["typescript", "react", "nodejs", "postgresql", "docker",
                 "rest-api", "testing", "system-design"],
        "nice": ["graphql", "aws", "redis", "python", "ci-cd"],
    },
    "security-engineer": {
        "title": "Trust Engineer",
        "domain": "application and platform hardening",
        "core": ["security", "python", "linux", "docker", "kubernetes",
                 "aws", "observability", "ci-cd", "terraform", "bash"],
        "nice": ["golang", "kafka", "grpc", "system-design", "testing"],
    },
}

_SENIORITY_WORD = {0: "Intern", 1: "Junior", 2: "", 3: "Senior", 4: "Staff", 5: "Principal"}

_COMPANIES = ("Northwind Analytics", "Vertex Labs", "Bluepine Systems", "Corvid Tech",
              "Meridian Data", "Halcyon Cloud", "Ironwood Digital", "Solstice AI",
              "Quartz Retail", "Lumen Health", "Kestrel Logistics", "Aurora Fintech")
_LOCATIONS = ("Remote (EU)", "Berlin", "Austin, TX", "Dublin", "Toronto", "Remote (US)")

# Fictional institutions so the prestige signal is entirely synthetic.
_SCHOOLS_TOP = ("Northgate Institute of Technology", "Ashford University",
                "Calder Institute of Technology", "Westbrook University")
_SCHOOLS_MID = ("Lakeshore University", "Rivermont University", "Granite Bay University",
                "Eastfield University")
_SCHOOLS_LOW = ("Fairview State College", "Brookline Community College",
                "Pinehurst State College", "Millbrook Polytechnic")

# A vendor-supplied "school ranking" file of exactly the kind that leaks class
# and geography into hiring scores. The engine reads it only when
# ScoringWeights.prestige > 0, and blind mode removes the school from the text
# so the lookup misses and every candidate falls back to the neutral value.
SCHOOL_PRESTIGE: dict[str, float] = {
    **{s: 0.90 for s in _SCHOOLS_TOP},
    **{s: 0.55 for s in _SCHOOLS_MID},
    **{s: 0.20 for s in _SCHOOLS_LOW},
}

_FIRST_F = ("Amara", "Priya", "Sofia", "Chen", "Fatima", "Hannah", "Ines", "Mei",
            "Nadia", "Olivia", "Rosa", "Yuki")
_FIRST_M = ("Andre", "Bilal", "Diego", "Ethan", "Hiroshi", "Ivan", "Jamal", "Kenji",
            "Lars", "Marco", "Omar", "Tomas")
_LAST = ("Adeyemi", "Bauer", "Costa", "Dubois", "Eriksson", "Fernandez", "Gupta",
         "Haddad", "Ibrahim", "Jensen", "Kowalski", "Lindqvist", "Moreau", "Nakamura",
         "Okafor", "Petrov", "Quintero", "Rossi", "Santos", "Tanaka")

_DEGREE_TEXT = {
    1: "Certificate, Software Engineering Bootcamp",
    2: "A.A.S. in Information Technology",
    3: "B.S. in Computer Science",
    4: "M.S. in Computer Science",
    5: "Ph.D. in Computer Science",
}

_BULLETS = (
    "Owned the {d} stack end to end, shipping {a} and {b} into production.",
    "Built {a} pipelines backed by {b}, cutting p95 latency by {n}%.",
    "Led the migration to {a}; paired it with {b} to halve incident volume.",
    "Designed and shipped {a} services, instrumented with {b}, serving {n}M requests/day.",
    "Scaled {a} workloads and hardened {b}, reducing infrastructure spend by {n}%.",
    "Partnered with product on {a} rollouts; standardised {b} across four teams.",
)
_SOLO_BULLETS = (
    "Ran {a} for the {d} group and mentored two engineers.",
    "Reworked the {a} layer, unblocking a quarter of roadmap items.",
    "Introduced {a} reviews that cut regressions by {n}%.",
)
_FILLER_BULLETS = (
    "Reduced on-call pages by a third through better runbooks and ownership boundaries.",
    "Drove a quarterly planning process across three squads.",
    "Wrote the design docs that unblocked two dependent teams.",
)

_JOB_MISSION = (
    "We are rebuilding the {d} layer that every product team depends on.",
    "Our {d} systems doubled in volume this year and need an owner.",
    "You will lead the next generation of our {d} platform.",
)


@dataclass
class SyntheticJob:
    """A rendered posting plus the latent requirement it was rendered from."""

    job_id: str
    text: str
    latent: dict = field(default_factory=dict)


@dataclass
class SyntheticCandidate:
    """A rendered resume plus the latent profile it was rendered from."""

    candidate_id: str
    text: str
    latent: dict = field(default_factory=dict)


def _surface(tax: SkillTaxonomy, sid: str, rng: random.Random) -> str:
    """Pick a way to *write* a skill, sometimes an alias, to exercise normalisation."""
    if rng.random() < 0.55:
        return tax.display(sid)
    forms = [f for f in tax.skills[sid].surface_forms if tax.resolve(f) == sid and len(f) >= 2]
    form = rng.choice(forms) if forms else tax.display(sid)
    return form.upper() if len(form) <= 3 else form


def _ladder(years: float) -> int:
    return 1 if years < 2 else 2 if years < 5 else 3 if years < 9 else 4


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _shift(anchor: date, months: int) -> date:
    total = anchor.year * 12 + (anchor.month - 1) - months
    return date(total // 12, total % 12 + 1, 1)


def _render_job(tax: SkillTaxonomy, jid: str, arch_name: str, arch: dict,
                must: list[str], nice: list[str], years: int, seniority: int,
                edu_level: int, rng: random.Random) -> SyntheticJob:
    word = _SENIORITY_WORD[seniority]
    title = f"{word} {arch['title']}".strip()
    s = [_surface(tax, m, rng) for m in must]
    n = [_surface(tax, x, rng) for x in nice]
    edu_word = "Bachelor's" if edu_level <= 3 else "Master's"
    lines = [
        title,
        f"{rng.choice(_COMPANIES)} | {rng.choice(_LOCATIONS)} | Full-time",
        "",
        "About the role",
        rng.choice(_JOB_MISSION).format(d=arch["domain"]),
        "",
        "Requirements:",
        f"- {years}+ years of professional experience shipping production systems",
        f"- Must have strong {s[0]} and {s[1]}",
        f"- Required: hands-on {s[2]} in a production setting",
        f"- Solid {s[3]}; proven track record with {s[4]}",
        f"- {edu_word} degree in a quantitative field or equivalent practical experience",
        "",
        "Nice to have:",
        f"- Familiarity with {n[0]} or {n[1]}",
        f"- Exposure to {n[2]} is a plus",
    ]
    return SyntheticJob(
        job_id=jid,
        text="\n".join(lines),
        latent={
            "archetype": arch_name,
            "must": sorted(must),
            "nice": sorted(nice),
            "min_years": float(years),
            "seniority": seniority,
            "education_level": edu_level,
            "title": title,
        },
    )


def _render_resume(tax: SkillTaxonomy, cid: str, latent: dict, rng: random.Random,
                   as_of: date) -> SyntheticCandidate:
    """Lay skills out across dated roles so per-skill years are recoverable."""
    arch = ARCHETYPES[latent["archetype"]]
    years = latent["years"]
    n_roles = 2 if years < 3.5 else 3 if years < 8 else 4
    total_months = max(12, int(round(years * 12)))
    weights = [rng.uniform(0.8, 1.6) for _ in range(n_roles)]
    scale = total_months / sum(weights)
    durations = [max(9, int(round(w * scale))) for w in weights]

    ends: list[date] = []
    starts: list[date] = []
    cursor = as_of
    for dur in durations:  # index 0 == most recent
        ends.append(cursor)
        starts.append(_shift(cursor, dur))
        cursor = starts[-1]

    role_skills: list[list[str]] = [[] for _ in range(n_roles)]
    for sid in latent["role_skills"]:
        if latent.get("all_roles"):
            span = range(n_roles)
        elif rng.random() < latent.get("stale_p", 0.22) and n_roles > 1:
            oldest = rng.randrange(1, n_roles)
            span = range(rng.randint(1, oldest), oldest + 1)
        else:
            span = range(0, rng.randrange(n_roles) + 1)
        for i in span:
            role_skills[i].append(sid)

    word = _SENIORITY_WORD[latent["seniority"]]
    lines: list[str] = [
        latent["name"],
        f"{latent['email']} | {rng.choice(_LOCATIONS)} | +1 555 {rng.randint(1000, 9999)}",
    ]
    if latent.get("age"):
        lines.append(f"Age: {latent['age']}")
    lines += [
        "",
        "SUMMARY",
        f"{word or 'Mid-level'} {arch['title']} with {years:.0f} years of experience "
        f"delivering {arch['domain']} systems. "
        f"{latent['pronoun_subject'].capitalize()} is known for calm, reliable delivery.",
        "",
        "EXPERIENCE",
    ]
    for i in range(n_roles):
        # Titles descend in seniority going back in time; they carry no skill words.
        role_level = max(0, min(5, latent["seniority"] - i))
        role_word = _SENIORITY_WORD[role_level]
        role_title = f"{role_word} {arch['title']}".strip()
        end_txt = "Present" if i == 0 else f"{ends[i].year:04d}-{ends[i].month:02d}"
        lines.append(
            f"{role_title} | {rng.choice(_COMPANIES)} | "
            f"{starts[i].year:04d}-{starts[i].month:02d} to {end_txt}"
        )
        pool = list(role_skills[i])
        rng.shuffle(pool)
        bullets: list[str] = []
        # The pool is drained completely: a skill the renderer silently dropped
        # would break the latent-vs-parsed contract the evaluation relies on.
        while len(pool) >= 2:
            a, b = pool.pop(), pool.pop()
            bullets.append(rng.choice(_BULLETS).format(
                a=_surface(tax, a, rng), b=_surface(tax, b, rng),
                d=arch["domain"], n=rng.randint(12, 68)))
        if pool:
            bullets.append(rng.choice(_SOLO_BULLETS).format(
                a=_surface(tax, pool.pop(), rng), d=arch["domain"], n=rng.randint(12, 68)))
        if not bullets:
            bullets.append(rng.choice(_FILLER_BULLETS))
        lines.extend(f"- {b}" for b in bullets)
        lines.append("")

    lines += [
        "EDUCATION",
        f"{_DEGREE_TEXT[latent['education_level']]}, {latent['school']}, {latent['grad_year']}",
        "",
    ]
    if latent["declared_skills"]:
        lines += ["SKILLS",
                  ", ".join(_surface(tax, s, rng) for s in latent["declared_skills"])]
    return SyntheticCandidate(candidate_id=cid, text="\n".join(lines), latent=latent)


def _make_identity(rng: random.Random, group: str, prestige: float,
                   years: float, as_of: date, edu_level: int) -> dict:
    female = rng.random() < 0.5
    first = rng.choice(_FIRST_F if female else _FIRST_M)
    last = rng.choice(_LAST)
    pool = _SCHOOLS_TOP if prestige >= 0.62 else _SCHOOLS_MID if prestige >= 0.33 else _SCHOOLS_LOW
    grad_year = as_of.year - int(years) - rng.choice([0, 1, 1, 2])
    return {
        "name": f"{first} {last}",
        "gender": "F" if female else "M",
        "pronoun_subject": "she" if female else "he",
        "email": f"{first.lower()}.{last.lower()}@mail.example",
        "group": group,
        "prestige": round(prestige, 3),
        "school": rng.choice(pool),
        "grad_year": grad_year,
        "age": (as_of.year - grad_year + 22 + rng.choice([0, 1])) if rng.random() < 0.3 else None,
        "education_level": edu_level,
    }


def _draw_prestige(rng: random.Random, group: str) -> float:
    """Prestige is the ONLY channel through which the proxy group touches score."""
    if group == "A":
        return rng.uniform(0.62, 1.0)
    if group == "B":
        return rng.uniform(0.02, 0.38)
    return rng.uniform(0.30, 0.78)


def _descendants(tax: SkillTaxonomy, target: str) -> set[str]:
    """Every skill that would imply ``target`` (must be removed for a true gap)."""
    return {sid for sid in tax.skills if target in tax.ancestors(sid)}


def generate(
    tax: SkillTaxonomy,
    seed: int = 42,
    n_jobs: int = 16,
    n_random_candidates: int = 120,
    as_of: date = DEFAULT_AS_OF,
) -> tuple[list[SyntheticJob], list[SyntheticCandidate]]:
    """Build the job board and talent pool for ``seed`` (fully deterministic)."""
    rng = random.Random(seed)
    arch_names = list(ARCHETYPES)

    jobs: list[SyntheticJob] = []
    for j in range(n_jobs):
        name = arch_names[j % len(arch_names)]
        arch = ARCHETYPES[name]
        must = rng.sample(arch["core"], 5)
        rest = [s for s in arch["core"] if s not in must]
        nice = rng.sample(arch["nice"] + rest, 3)
        req_years = rng.choice([2, 3, 4, 5, 6, 7])
        seniority = _ladder(req_years)
        edu_level = rng.choice([3, 3, 3, 4])
        jobs.append(_render_job(tax, f"J{j + 1:02d}", name, arch, must, nice,
                                req_years, seniority, edu_level, rng))

    candidates: list[SyntheticCandidate] = []

    # --- background pool -------------------------------------------------
    for i in range(n_random_candidates):
        name = rng.choice(arch_names)
        arch = ARCHETYPES[name]
        core = arch["core"]
        k = max(2, int(round(rng.uniform(0.5, 1.0) * len(core))))
        held = set(rng.sample(core, k))
        held.update(rng.sample(arch["nice"], rng.randint(0, 3)))
        # Some candidates state only a specialisation where the job states the
        # generic skill; the implication graph is what rescues them.
        for generic in sorted(held):
            if rng.random() < 0.3:
                kids = sorted(c for c in tax.children[generic] if c not in held)
                if kids:
                    held.discard(generic)
                    held.add(rng.choice(kids))
        other = ARCHETYPES[rng.choice([a for a in arch_names if a != name])]
        declared = set(rng.sample(other["core"], rng.randint(0, 3))) - held

        years = round(rng.uniform(1.5, 15.0), 1)
        seniority = max(0, min(5, _ladder(years) + rng.choice([-1, 0, 0, 0, 1])))
        edu_level = rng.choices([1, 2, 3, 4, 5], weights=[5, 8, 45, 32, 10])[0]
        group = rng.choices(["A", "B", "C"], weights=[0.38, 0.34, 0.28])[0]
        latent = _make_identity(rng, group, _draw_prestige(rng, group), years, as_of, edu_level)
        latent.update({
            "archetype": name, "years": years, "seniority": seniority,
            "role_skills": sorted(held), "declared_skills": sorted(declared),
            "skills": sorted(held | declared), "kind": "pool",
            "planted_for": None, "missing_skill": None, "stale_p": 0.22,
        })
        candidates.append(_render_resume(tax, f"C{i + 1:03d}", latent, rng, as_of))

    # --- planted ideal and near-miss candidates --------------------------
    for j, job in enumerate(jobs):
        lat = job.latent
        arch = ARCHETYPES[lat["archetype"]]
        base = set(lat["must"]) | set(lat["nice"])
        base.update(rng.sample(arch["core"], 2))
        years = max(lat["min_years"] + 2.0, 5.0)
        group = "ABC"[j % 3]

        ideal_id = _make_identity(rng, group, 0.5, years, as_of,
                                  max(lat["education_level"], EDUCATION_LEVEL["bachelor"]))
        ideal = dict(ideal_id)
        ideal.update({
            "archetype": lat["archetype"], "years": years, "seniority": lat["seniority"],
            "role_skills": sorted(base), "declared_skills": [], "skills": sorted(base),
            "kind": "ideal", "planted_for": job.job_id, "missing_skill": None,
            "all_roles": True, "stale_p": 0.0,
        })
        candidates.append(_render_resume(tax, f"IDEAL-{job.job_id}", ideal, rng, as_of))

        # Drop one must-have plus everything that would imply it.
        target = max(lat["must"], key=lambda s: (tax.skills[s].difficulty, s))
        stripped = (base - {target}) - _descendants(tax, target)
        miss_id = _make_identity(rng, "ABC"[(j + 1) % 3], 0.5, years, as_of,
                                 max(lat["education_level"], EDUCATION_LEVEL["bachelor"]))
        miss = dict(miss_id)
        miss.update({
            "archetype": lat["archetype"], "years": years, "seniority": lat["seniority"],
            "role_skills": sorted(stripped), "declared_skills": [], "skills": sorted(stripped),
            "kind": "nearmiss", "planted_for": job.job_id, "missing_skill": target,
            "all_roles": True, "stale_p": 0.0,
        })
        cand = _render_resume(tax, f"GAP-{job.job_id}", miss, rng, as_of)
        # The plant is only valid if the rendered text truly lacks the skill.
        found = {m.skill_id for m in tax.extract(cand.text)}
        if target in found or found & _descendants(tax, target):
            raise AssertionError(f"near-miss plant leaked {target} into {cand.candidate_id}")
        candidates.append(cand)

    return jobs, candidates


def ground_truth_qualified(tax: SkillTaxonomy, cand_latent: dict, job_latent: dict) -> int:
    """Label a pair from LATENT facts only - never from parsed text or scores.

    A candidate is qualified when every must-have is held directly or implied
    within two hops, tenure is within half a year of the requirement, and they
    are no more than one level below the posted seniority.
    """
    expanded = tax.expand(set(cand_latent["skills"]))
    covered = all(expanded.get(m, 0.0) >= 0.5 for m in job_latent["must"])
    return int(
        covered
        and cand_latent["years"] >= job_latent["min_years"] - 0.5
        and cand_latent["seniority"] >= job_latent["seniority"] - 1
    )
