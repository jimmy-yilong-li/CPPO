"""Evaluation metrics for code generation."""

from __future__ import annotations

import math

# Canonical attempt-budget ladder reported for pass@K sweeps. A single pool of
# n completions supports every k <= n, so one n=32 pool yields the whole ladder
# without re-sampling. K=4 is the budget used by the main results; the larger
# rungs show whether a gain persists as the budget grows.
DEFAULT_K_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Standard unbiased pass@k estimator.

    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: k in pass@k.

    Returns:
        Estimated probability that at least one of k samples is correct.

    Uses the combinatorial formula:
        pass@k = 1 - C(n-c, k) / C(n, k)
    where C(n-c, k) = 0 when n-c < k (meaning pass@k = 1.0).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if c < 0 or c > n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n")

    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # Use log-space to avoid overflow with large combinatorials
    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    log_prod = 0.0
    for i in range(k):
        log_prod += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_prod)


def pass_at_k_sweep(n: int, c: int, ks=DEFAULT_K_LADDER) -> dict[int, float]:
    """Compute pass@k for multiple k from a single (n, c) sample pool.

    All k share the same n and c, so this is just ``pass_at_k`` invoked per k.
    The helper makes the "single sampling pool, multiple eval-k" protocol
    explicit: an n=32 pool reads off pass@1 through pass@32 without re-sampling
    for each k, which would be both wasteful and unfair.

    Args:
        n: Total samples generated per problem.
        c: Number of correct samples in those n.
        ks: Iterable of k values, defaulting to ``DEFAULT_K_LADDER``.

    Returns:
        Dict mapping each k to the unbiased pass@k estimate, ascending by k.
        Values of k greater than n are **omitted**: pass@k is not estimable
        from fewer than k samples, so reporting pass@32 requires n >= 32.
    """
    return {k: pass_at_k(n, c, k) for k in sorted(set(ks)) if k <= n}


def mean_pass_at_k_sweep(per_problem: list[tuple[int, int]], ks=DEFAULT_K_LADDER) -> dict[int, float]:
    """Average ``pass_at_k_sweep`` over problems, one (n, c) pair per problem.

    A rung is reported only when *every* problem has n >= k, so the mean at
    each k is taken over the same problem set. A pool that is short on even one
    problem drops that rung rather than averaging over a shifting denominator.

    Args:
        per_problem: One ``(n, c)`` pair per problem; pairs with n <= 0 are skipped.
        ks: Iterable of k values, defaulting to ``DEFAULT_K_LADDER``.

    Returns:
        Dict mapping k to the mean pass@k across problems, ascending by k.
        Empty when no problem produced samples.
    """
    pools = [(n, c) for n, c in per_problem if n > 0]
    if not pools:
        return {}
    min_n = min(n for n, _ in pools)
    usable = [k for k in sorted(set(ks)) if k <= min_n]
    return {
        k: sum(pass_at_k(n, c, k) for n, c in pools) / len(pools)
        for k in usable
    }
