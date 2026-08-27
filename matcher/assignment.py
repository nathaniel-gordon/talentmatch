"""Optimal one-to-one assignment of candidates to jobs.

Independent ranking (best N for a posting) can give the same person every
role. A hiring round that must fill several seats at once needs a *matching*:
each candidate and each job appears in at most one pair, and the chosen pairs
maximise total fit.

This module solves that with the Hungarian algorithm on the dense score
matrix already produced by :class:`~matcher.engine.MatchEngine`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .scoring import MatchResult


@dataclass
class AssignedPair:
    """One exclusive candidate/job seat in a matching."""

    candidate_id: str
    job_id: str
    score: float
    result: MatchResult | None = None


@dataclass
class Assignment:
    """A globally consistent matching plus leftover people and seats."""

    pairs: list[AssignedPair] = field(default_factory=list)
    total_score: float = 0.0
    unmatched_candidates: list[str] = field(default_factory=list)
    unmatched_jobs: list[str] = field(default_factory=list)

    @property
    def n_filled(self) -> int:
        return len(self.pairs)


def greedy_assignment(scores: np.ndarray, candidate_ids: list[str],
                      job_ids: list[str]) -> Assignment:
    """Pick highest remaining cell until the board is exhausted.

    Fast, but not guaranteed optimal; used as a baseline for tests and as a
    fallback explanation ("the Hungarian matching recovered X more score").
    """
    matrix = np.asarray(scores, dtype=float).copy()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: list[AssignedPair] = []
    n_rows, n_cols = matrix.shape
    while len(used_rows) < n_rows and len(used_cols) < n_cols:
        masked = matrix.copy()
        for r in used_rows:
            masked[r, :] = -np.inf
        for c in used_cols:
            masked[:, c] = -np.inf
        idx = int(np.argmax(masked))
        r, c = divmod(idx, n_cols)
        if not np.isfinite(masked[r, c]):
            break
        used_rows.add(r)
        used_cols.add(c)
        pairs.append(AssignedPair(candidate_ids[r], job_ids[c], float(matrix[r, c])))
    pairs.sort(key=lambda p: (-p.score, p.candidate_id, p.job_id))
    return _finish(pairs, candidate_ids, job_ids, used_rows, used_cols)


def optimal_assignment(scores: np.ndarray, candidate_ids: list[str],
                       job_ids: list[str]) -> Assignment:
    """Maximise sum of scores subject to at most one pair per row and column."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D candidates x jobs matrix")
    if matrix.shape != (len(candidate_ids), len(job_ids)):
        raise ValueError("score matrix shape must match candidate and job id lists")
    if matrix.size == 0:
        return Assignment(unmatched_candidates=list(candidate_ids),
                          unmatched_jobs=list(job_ids))

    # linear_sum_assignment minimises; invert scores. A constant shift keeps
    # every entry non-negative so empty / all-zero boards stay well-defined.
    cost = matrix.max() - matrix
    rows, cols = linear_sum_assignment(cost)
    used_rows = set(int(r) for r in rows)
    used_cols = set(int(c) for c in cols)
    pairs = [
        AssignedPair(candidate_ids[int(r)], job_ids[int(c)], float(matrix[int(r), int(c)]))
        for r, c in zip(rows, cols)
    ]
    pairs.sort(key=lambda p: (-p.score, p.candidate_id, p.job_id))
    return _finish(pairs, candidate_ids, job_ids, used_rows, used_cols)


def brute_force_max_sum(scores: np.ndarray) -> float:
    """Exact maximum matching sum for tiny boards (test oracle only)."""
    matrix = np.asarray(scores, dtype=float)
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return 0.0
    if n_rows <= n_cols:
        best = 0.0
        for cols in permutations(range(n_cols), n_rows):
            best = max(best, float(sum(matrix[r, c] for r, c in enumerate(cols))))
        return best
    best = 0.0
    for rows in permutations(range(n_rows), n_cols):
        best = max(best, float(sum(matrix[r, c] for c, r in enumerate(rows))))
    return best


def _finish(pairs: list[AssignedPair], candidate_ids: list[str], job_ids: list[str],
            used_rows: set[int], used_cols: set[int]) -> Assignment:
    unmatched_c = [candidate_ids[i] for i in range(len(candidate_ids)) if i not in used_rows]
    unmatched_j = [job_ids[i] for i in range(len(job_ids)) if i not in used_cols]
    total = round(sum(p.score for p in pairs), 4)
    return Assignment(
        pairs=pairs, total_score=total,
        unmatched_candidates=unmatched_c, unmatched_jobs=unmatched_j,
    )
