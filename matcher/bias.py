"""Bias mitigation: blind-mode redaction and a disparate-impact audit.

Two distinct jobs live here.

**Blind mode** rewrites the resume *text* before parsing, removing name, contact
details, gendered pronouns, stated age, graduation year and institution. It is
deliberately surgical: everything that evidences a skill must survive intact,
which is what the smoke test verifies by requiring identical qualification
rankings before and after redaction.

**The audit** takes a supplied demographic-proxy column and reports score
distribution parity plus the EEOC "four-fifths rule": if the least-selected
group's selection rate is below 80% of the most-selected group's, the selection
procedure shows adverse impact and needs justification.

Neither function infers protected attributes. The proxy column is supplied by
the caller, exactly as a compliance team would supply it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import mannwhitneyu

from .parsing import SECTION_RE  # section map is shared with the resume parser

REDACTED_NAME = "[REDACTED_NAME]"
REDACTED_SCHOOL = "[REDACTED_SCHOOL]"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_AGE_LINE_RE = re.compile(r"^\s*(?:age|date of birth|dob)\s*[:\-].*$", re.IGNORECASE | re.MULTILINE)
_AGE_PHRASE_RE = re.compile(r"\b\d{2}\s+years?\s+old\b", re.IGNORECASE)
_SCHOOL_RE = re.compile(
    r"\b(?:[A-Z][\w.&'-]*\s+)*"
    r"(?:University|Institute of Technology|Institute|College|Polytechnic|Academy)"
    r"(?:\s+of\s+[A-Z][\w'-]+)?"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Subject pronouns become a noun phrase so the sentence stays grammatical;
# possessives collapse to "their".
_PRONOUN_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:he|she)\b", re.IGNORECASE), "the candidate"),
    (re.compile(r"\b(?:his|her)\b", re.IGNORECASE), "their"),
    (re.compile(r"\b(?:him|hers)\b", re.IGNORECASE), "them"),
    (re.compile(r"\b(?:mr|mrs|ms|miss)\.?\b", re.IGNORECASE), "the candidate"),
)


def blind_redact(text: str) -> str:
    """Strip identity, age and institution signals while preserving skill evidence.

    Graduation years are removed only inside the EDUCATION block: the dates in
    EXPERIENCE are load-bearing (they are how per-skill years are computed) and
    stripping them would destroy the very evidence blind screening is meant to
    protect.
    """
    marks = list(SECTION_RE.finditer(text))
    head_end = marks[0].start() if marks else len(text)
    head, rest = text[:head_end], text[head_end:]

    head_lines = head.splitlines()
    out_lines: list[str] = []
    seen_name = False
    for line in head_lines:
        if not line.strip():
            out_lines.append(line)
            continue
        if not seen_name:
            out_lines.append(REDACTED_NAME)
            seen_name = True
            continue
        out_lines.append(_EMAIL_RE.sub("[REDACTED_CONTACT]", line))
    head = "\n".join(out_lines)
    head = _AGE_LINE_RE.sub("[REDACTED_AGE]", head)

    # Rebuild the tail section by section so year stripping is scoped.
    tail_parts: list[str] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        name, body = m.group(1).upper(), text[m.end(): end]
        if name == "EDUCATION":
            body = _SCHOOL_RE.sub(REDACTED_SCHOOL, body)
            body = _YEAR_RE.sub("[REDACTED_YEAR]", body)
        tail_parts.append(text[m.start(): m.end()] + body)
    rest = "".join(tail_parts) if marks else rest

    combined = head + rest
    combined = _AGE_PHRASE_RE.sub("[REDACTED_AGE]", combined)
    combined = _EMAIL_RE.sub("[REDACTED_CONTACT]", combined)
    for pattern, repl in _PRONOUN_SUBS:
        combined = pattern.sub(repl, combined)
    return combined


@dataclass
class GroupStat:
    """Selection and score statistics for one proxy-column value."""

    group: str
    n_candidates: int
    n_decisions: int
    n_selected: int
    selection_rate: float
    mean_score: float
    median_score: float
    std_score: float
    p_value_vs_reference: float | None = None


@dataclass
class AuditReport:
    """Result of a four-fifths / score-parity audit."""

    top_fraction: float
    groups: list[GroupStat] = field(default_factory=list)
    reference_group: str = ""
    impact_ratio: float = 1.0
    passes_four_fifths: bool = True
    most_selected: str = ""
    least_selected: str = ""
    n_decisions: int = 0

    @property
    def verdict(self) -> str:
        if self.passes_four_fifths:
            return f"PASS - impact ratio {self.impact_ratio:.3f} >= 0.80"
        return (f"ADVERSE IMPACT - impact ratio {self.impact_ratio:.3f} < 0.80 "
                f"({self.least_selected} vs {self.most_selected})")


def disparate_impact_audit(
    scores_by_job: dict[str, list[tuple[str, float]]],
    proxy_by_candidate: dict[str, str],
    top_fraction: float = 0.20,
) -> AuditReport:
    """Four-fifths rule over a top-``top_fraction`` shortlist per job.

    ``scores_by_job`` maps job id to (candidate_id, score) for every candidate
    considered. Selection rate is pooled across jobs: one candidate shortlisted
    for three of sixteen jobs contributes three positive decisions out of
    sixteen, which is how a real screening funnel behaves.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    selected: dict[str, int] = {}
    decisions: dict[str, int] = {}
    all_scores: dict[str, list[float]] = {}
    total_decisions = 0

    for _job_id, rows in scores_by_job.items():
        ranked = sorted(rows, key=lambda r: (-r[1], r[0]))
        k = max(1, math.ceil(top_fraction * len(ranked)))
        chosen = {cid for cid, _ in ranked[:k]}
        for cid, score in ranked:
            group = proxy_by_candidate.get(cid)
            if group is None:
                continue
            decisions[group] = decisions.get(group, 0) + 1
            total_decisions += 1
            all_scores.setdefault(group, []).append(score)
            if cid in chosen:
                selected[group] = selected.get(group, 0) + 1

    group_names = sorted(decisions)
    if not group_names:
        raise ValueError("no candidate in scores_by_job carried a proxy value")

    n_cands: dict[str, int] = {}
    for cid, g in proxy_by_candidate.items():
        n_cands[g] = n_cands.get(g, 0) + 1

    reference = max(group_names, key=lambda g: selected.get(g, 0) / max(1, decisions[g]))
    ref_scores = np.asarray(all_scores[reference], dtype=float)

    stats: list[GroupStat] = []
    for g in group_names:
        arr = np.asarray(all_scores[g], dtype=float)
        rate = selected.get(g, 0) / decisions[g]
        p_val: float | None = None
        if g != reference and arr.size > 1 and ref_scores.size > 1:
            p_val = float(mannwhitneyu(arr, ref_scores, alternative="two-sided").pvalue)
        stats.append(GroupStat(
            group=g, n_candidates=n_cands.get(g, 0), n_decisions=decisions[g],
            n_selected=selected.get(g, 0), selection_rate=round(rate, 4),
            mean_score=round(float(arr.mean()), 4), median_score=round(float(np.median(arr)), 4),
            std_score=round(float(arr.std(ddof=0)), 4), p_value_vs_reference=p_val,
        ))

    rates = {s.group: s.selection_rate for s in stats}
    top_group = max(rates, key=lambda g: rates[g])
    low_group = min(rates, key=lambda g: rates[g])
    ratio = (rates[low_group] / rates[top_group]) if rates[top_group] > 0 else 1.0
    return AuditReport(
        top_fraction=top_fraction, groups=stats, reference_group=reference,
        impact_ratio=round(ratio, 4), passes_four_fifths=ratio >= 0.80,
        most_selected=top_group, least_selected=low_group, n_decisions=total_decisions,
    )
