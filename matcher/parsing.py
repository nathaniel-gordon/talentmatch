"""Parsers that turn free text into structured, auditable profiles.

Two directions, one vocabulary:

* :func:`parse_resume` recovers work history, per-skill years (from the dates of
  the roles that actually mention the skill, not from a self-declared list),
  recency, seniority and education.
* :func:`parse_job` splits a posting into MUST-have and NICE-to-have skills,
  minimum years, seniority and education floor.

Every extracted skill keeps its character span so a human can check the claim.
Years are derived from *merged* role intervals: a person who used Kafka in two
overlapping contracts has not doubled their Kafka experience.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .taxonomy import SkillMention, SkillTaxonomy

DEFAULT_AS_OF = date(2026, 1, 1)

SENIORITY_ORDER: tuple[str, ...] = (
    "intern", "junior", "mid", "senior", "staff", "principal", "executive",
)
SENIORITY_LEVEL: dict[str, int] = {name: i for i, name in enumerate(SENIORITY_ORDER)}

# Longest markers first so "mid-level" is not shadowed by "mid".
_SENIORITY_MARKERS: tuple[tuple[str, int], ...] = (
    (r"vice president|vp of|head of|director", 6),
    (r"principal|distinguished|fellow", 5),
    (r"staff|tech lead|technical lead|team lead|\blead\b", 4),
    (r"\bsenior\b|\bsr\.?\b", 3),
    (r"mid-?level|\bmid\b", 2),
    (r"\bjunior\b|\bjr\.?\b|associate|entry-?level", 1),
    (r"intern|trainee|apprentice", 0),
)

EDUCATION_ORDER: tuple[str, ...] = ("none", "bootcamp", "associate", "bachelor", "master", "phd")
EDUCATION_LEVEL: dict[str, int] = {name: i for i, name in enumerate(EDUCATION_ORDER)}

# Abbreviated degrees end in '.', where \b is useless ("B.S." followed by a
# space is not a word boundary), so each pattern guards with explicit
# character-class lookarounds instead.
_EDUCATION_MARKERS: tuple[tuple[str, int], ...] = (
    (r"ph\.?\s?d|doctorate|doctoral", 5),
    (r"master'?s?|(?<![a-z0-9])m\.?\s?s\.?c?(?![a-z0-9])|(?<![a-z0-9])m\.?eng(?![a-z0-9])|(?<![a-z0-9])mba(?![a-z0-9])", 4),
    (r"bachelor'?s?|(?<![a-z0-9])b\.?\s?s\.?c?(?![a-z0-9])|(?<![a-z0-9])b\.?\s?a\.?(?![a-z0-9])"
     r"|(?<![a-z0-9])b\.?eng(?![a-z0-9])|undergraduate degree", 3),
    (r"associate'?s? degree|(?<![a-z0-9])a\.?\s?a\.?\s?s?\.?(?![a-z0-9])", 2),
    (r"bootcamp|certificate program|certificate,", 1),
)

_ROLE_RE = re.compile(
    r"^(?P<title>[^|\n]+?)\s*\|\s*(?P<company>[^|\n]+?)\s*\|\s*"
    r"(?P<start>\d{4}-\d{2})\s*(?:to|-|–)\s*(?P<end>\d{4}-\d{2}|present)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_YEARS_RE = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*\+?\s*year", re.IGNORECASE)
SECTION_RE = re.compile(r"^(SUMMARY|EXPERIENCE|EDUCATION|SKILLS|PROJECTS)\s*:?\s*$", re.MULTILINE)

_NICE_HEADINGS = ("nice to have", "nice-to-have", "preferred", "bonus", "good to have",
                  "desirable", "pluses", "would be a plus")
_MUST_HEADINGS = ("requirement", "must have", "must-have", "required", "qualification",
                  "what you need", "you will need", "minimum")
_NICE_INLINE = ("preferred", "a plus", "bonus", "nice to have", "familiarity", "exposure to",
                "would be great", "ideally")
_MUST_INLINE = ("must ", "required", "strong ", "solid ", "expert", "deep experience", "proven")


def _month_ord(d: date) -> int:
    return d.year * 12 + d.month


def _parse_ym(token: str, as_of: date) -> date:
    if token.lower() == "present":
        return as_of
    year, month = token.split("-")
    return date(int(year), int(month), 1)


def _merge_months(intervals: list[tuple[int, int]]) -> int:
    """Total months covered by a union of [start, end) month ranges."""
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total, cur_lo, cur_hi = 0, *ordered[0]
    for lo, hi in ordered[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    return total + cur_hi - cur_lo


def detect_seniority(text: str) -> int | None:
    """Highest seniority marker present in ``text`` (None if the text is silent)."""
    for pattern, level in _SENIORITY_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return None


def detect_education(text: str) -> int:
    """Highest degree level mentioned; 0 when nothing is stated."""
    for pattern, level in _EDUCATION_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return 0


@dataclass
class WorkRole:
    """One position, with the block of prose that described it."""

    title: str
    company: str
    start: date
    end: date
    body: str
    seniority: int | None = None

    @property
    def months(self) -> int:
        return max(0, _month_ord(self.end) - _month_ord(self.start))


@dataclass
class SkillEvidence:
    """Everything the parser knows about one skill on one resume."""

    skill_id: str
    months: int = 0
    last_used_ord: int = 0
    roles: list[str] = field(default_factory=list)
    mentions: list[SkillMention] = field(default_factory=list)

    @property
    def years(self) -> float:
        return round(self.months / 12.0, 2)

    @property
    def declared_only(self) -> bool:
        """True when the skill is only listed, never tied to dated work."""
        return self.months == 0

    def freshness(self, as_of: date, half_life_months: float = 36.0) -> float:
        """Exponential recency decay; a skill unused for 3 years is worth half."""
        if self.last_used_ord == 0:
            return 0.35
        gap = max(0, _month_ord(as_of) - self.last_used_ord)
        return float(0.5 ** (gap / half_life_months))


@dataclass
class CandidateProfile:
    """Structured resume."""

    candidate_id: str
    name: str
    raw_text: str
    roles: list[WorkRole] = field(default_factory=list)
    skills: dict[str, SkillEvidence] = field(default_factory=dict)
    total_years: float = 0.0
    seniority: int = 2
    education_level: int = 0
    school: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def skill_ids(self) -> set[str]:
        return set(self.skills)


@dataclass
class JobRequirements:
    """Structured job posting."""

    job_id: str
    title: str
    raw_text: str
    must: list[str] = field(default_factory=list)
    nice: list[str] = field(default_factory=list)
    min_years: float = 0.0
    seniority: int = 2
    education_level: int = 0
    evidence: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _split_sections(text: str) -> dict[str, str]:
    """Map upper-case section headings to their bodies (``_head`` = preamble)."""
    parts: dict[str, str] = {}
    marks = list(SECTION_RE.finditer(text))
    parts["_head"] = text[: marks[0].start()] if marks else text
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        parts[m.group(1).lower()] = text[m.end(): end]
    return parts


def parse_resume(
    candidate_id: str,
    text: str,
    tax: SkillTaxonomy,
    as_of: date = DEFAULT_AS_OF,
    meta: dict | None = None,
) -> CandidateProfile:
    """Parse a resume into a :class:`CandidateProfile`.

    Per-skill years come from the dated roles whose prose mentions the skill.
    A skill that appears only under SKILLS gets zero years and is flagged
    ``declared_only`` so scoring can discount an unbacked claim.
    """
    sections = _split_sections(text)
    head = sections.get("_head", "")
    name = next((ln.strip() for ln in head.splitlines() if ln.strip()), candidate_id)

    experience = sections.get("experience", "")
    roles: list[WorkRole] = []
    matches = list(_ROLE_RE.finditer(experience))
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(experience)
        roles.append(
            WorkRole(
                title=m.group("title").strip(),
                company=m.group("company").strip(),
                start=_parse_ym(m.group("start"), as_of),
                end=_parse_ym(m.group("end"), as_of),
                body=experience[m.end(): body_end],
                seniority=detect_seniority(m.group("title")),
            )
        )

    skills: dict[str, SkillEvidence] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}

    def _record(mention: SkillMention, role: WorkRole | None) -> None:
        ev = skills.setdefault(mention.skill_id, SkillEvidence(mention.skill_id))
        ev.mentions.append(mention)
        if role is None:
            return
        intervals.setdefault(mention.skill_id, []).append(
            (_month_ord(role.start), _month_ord(role.end))
        )
        ev.last_used_ord = max(ev.last_used_ord, _month_ord(role.end))
        label = f"{role.title} @ {role.company}"
        if label not in ev.roles:
            ev.roles.append(label)

    for role in roles:
        for mention in tax.extract(role.title + "\n" + role.body):
            _record(mention, role)
    for key in ("_head", "summary", "skills", "projects", "education"):
        for mention in tax.extract(sections.get(key, "")):
            _record(mention, None)

    for sid, spans in intervals.items():
        skills[sid].months = _merge_months(spans)

    total_months = _merge_months([(_month_ord(r.start), _month_ord(r.end)) for r in roles])
    total_years = round(total_months / 12.0, 2)

    latest = max(roles, key=lambda r: _month_ord(r.end), default=None)
    seniority = latest.seniority if latest and latest.seniority is not None else None
    if seniority is None:
        seniority = detect_seniority(head) if detect_seniority(head) is not None else None
    if seniority is None:
        # No title marker anywhere: fall back to a tenure ladder.
        seniority = 1 if total_years < 2 else 2 if total_years < 5 else 3 if total_years < 9 else 4

    edu_text = sections.get("education", "")
    school_match = re.search(
        r"([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)*\s+"
        r"(?:University|Institute|College|Polytechnic|School))|(University\s+of\s+[A-Z][\w'-]+)",
        edu_text,
    )
    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        raw_text=text,
        roles=roles,
        skills=skills,
        total_years=total_years,
        seniority=int(seniority),
        education_level=detect_education(edu_text),
        school=(school_match.group(0).strip() if school_match else ""),
        meta=dict(meta or {}),
    )


def _heading_bucket(line: str) -> str | None:
    """Return the bucket a line switches to, or None when it is body text.

    Headings are recognised by cue phrase *and* shape (short, ends like a
    heading) so a sentence such as "you must have shipped..." is not mistaken
    for a section break.
    """
    low = line.lower().strip(" -*\t")
    if len(low) >= 60:
        return None
    if any(h in low for h in _NICE_HEADINGS) and low.endswith((":", "have", "preferred", "plus", "bonus")):
        return "nice"
    if any(h in low for h in _MUST_HEADINGS) and low.endswith((":", "have", "required", "qualifications", "need")):
        return "must"
    return None


def parse_job(
    job_id: str,
    text: str,
    tax: SkillTaxonomy,
    meta: dict | None = None,
) -> JobRequirements:
    """Parse a posting into must/nice skills plus years, seniority and education.

    Section headings set the default bucket; inline cues ("a plus",
    "familiarity with") override it for a single bullet, because real postings
    mix a "preferred" line into a "requirements" block all the time.
    """
    lines = text.splitlines()
    title = next((ln.strip() for ln in lines if ln.strip()), job_id)
    bucket = "must"
    must: dict[str, str] = {}
    nice: dict[str, str] = {}
    must_year_hits: list[int] = []

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        heading = _heading_bucket(line)
        if heading is not None:
            bucket = heading
            continue
        low = line.lower()
        target = bucket
        if any(cue in low for cue in _NICE_INLINE):
            target = "nice"
        elif any(cue in low for cue in _MUST_INLINE):
            target = "must"
        sink = must if target == "must" else nice
        for mention in tax.extract(line):
            sink.setdefault(mention.skill_id, line)
        if target == "must":
            must_year_hits.extend(int(m.group(1)) for m in _YEARS_RE.finditer(line))

    for sid in must:  # an explicit requirement always outranks a preference
        nice.pop(sid, None)

    if not must_year_hits:
        must_year_hits = [int(m.group(1)) for m in _YEARS_RE.finditer(text)]
    min_years = float(max(must_year_hits)) if must_year_hits else 0.0

    seniority = detect_seniority(title)
    if seniority is None:
        seniority = 1 if min_years < 2 else 2 if min_years < 5 else 3 if min_years < 8 else 4

    return JobRequirements(
        job_id=job_id,
        title=title,
        raw_text=text,
        must=sorted(must),
        nice=sorted(nice),
        min_years=min_years,
        seniority=int(seniority),
        education_level=detect_education(text),
        evidence={**must, **nice},
        meta=dict(meta or {}),
    )
