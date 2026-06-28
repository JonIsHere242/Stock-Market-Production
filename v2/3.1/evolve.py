"""Evolutionary search over a parametrized feature-family's discrete parameter grid.

A feature family is a builder plus a grid of parameters (window / stat / decay /
neutralization). Exhaustively scoring every grid combination is wasteful once the grid
is large, so this module EVOLVES good combinations with a genetic algorithm guided by a
``score_fn`` callback (e.g. the framework's cross-sectional tail-lift screen). It is fully
decoupled from the engine: the only contract is ``score_fn(params: dict) -> float`` with
HIGHER == better. Scores are cached per unique individual, so the same params are never
re-scored, and all randomness flows through one seeded ``numpy.random.default_rng``.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def grid_size(param_space: dict[str, list]) -> int:
    """Return the exhaustive count of the grid (product of per-param value counts)."""
    total = 1
    for values in param_space.values():
        total *= len(values)
    return total


def _key(params: dict) -> str:
    """Canonical cache key for an individual: ``str`` of its sorted (name, value) tuple."""
    return str(tuple(sorted(params.items())))


def _random_individual(param_space: dict[str, list], rng: np.random.Generator) -> dict:
    """Sample one individual by drawing an independent random value per parameter."""
    return {name: values[int(rng.integers(len(values)))] for name, values in param_space.items()}


def _crossover(parent_a: dict, parent_b: dict, rng: np.random.Generator) -> dict:
    """Uniform crossover: each gene is taken independently from parent A or B."""
    child = {}
    for name in parent_a:
        child[name] = parent_a[name] if rng.random() < 0.5 else parent_b[name]
    return child


def _mutate(
    individual: dict,
    param_space: dict[str, list],
    rng: np.random.Generator,
    mutation_rate: float,
) -> dict:
    """With probability ``mutation_rate`` resample ONE random gene from its grid."""
    if rng.random() >= mutation_rate:
        return dict(individual)
    mutant = dict(individual)
    name = list(param_space.keys())[int(rng.integers(len(param_space)))]
    values = param_space[name]
    mutant[name] = values[int(rng.integers(len(values)))]
    return mutant


def _tournament(
    population: list[dict],
    scores: list[float],
    rng: np.random.Generator,
    tournament_k: int,
) -> dict:
    """Pick ``tournament_k`` distinct-index contestants at random; return the best one."""
    n = len(population)
    k = min(tournament_k, n)
    idx = rng.choice(n, size=k, replace=False)
    best = max(idx, key=lambda i: scores[i])
    return population[int(best)]


# --------------------------------------------------------------------------- #
# main optimizer
# --------------------------------------------------------------------------- #
def evolve_params(
    param_space: dict[str, list],
    score_fn: Callable[[dict], float],
    *,
    pop_size: int = 24,
    n_generations: int = 12,
    seed: int = 0,
    elitism: int = 2,
    mutation_rate: float = 0.3,
    tournament_k: int = 3,
    max_evals: Optional[int] = None,
) -> dict:
    """Evolve good parameter combinations over a discrete grid via a genetic algorithm.

    Initializes ``pop_size`` distinct random individuals, then each generation keeps the
    top ``elitism`` and breeds the rest by tournament selection -> uniform crossover ->
    mutation. ``score_fn`` results are cached per unique individual (so re-scoring never
    happens), and ``n_evals`` counts only UNIQUE calls. Stops after ``n_generations`` or
    once ``n_evals >= max_evals`` (if given). Returns best params/score, a per-generation
    history (best + mean), the unique-eval count, and the full cache of evaluated params.
    """
    rng = np.random.default_rng(seed)
    cache: dict[str, float] = {}

    def budget_left() -> bool:
        return max_evals is None or len(cache) < max_evals

    def evaluate(individual: dict) -> Optional[float]:
        """Cached score; returns None if this is a NEW eval and the budget is spent."""
        key = _key(individual)
        if key in cache:
            return cache[key]
        if not budget_left():
            return None
        cache[key] = float(score_fn(individual))
        return cache[key]

    # ---- initialize a population of distinct individuals -------------------- #
    grid_total = grid_size(param_space)
    target_pop = min(pop_size, grid_total)
    population: list[dict] = []
    seen_keys: set[str] = set()
    attempts = 0
    attempt_cap = max(target_pop * 50, 1000)
    while len(population) < target_pop and attempts < attempt_cap:
        candidate = _random_individual(param_space, rng)
        key = _key(candidate)
        if key not in seen_keys:
            seen_keys.add(key)
            population.append(candidate)
        attempts += 1
    # if the grid is tiny and sampling stalled, enumerate the remainder.
    if len(population) < target_pop:
        for combo in itertools.product(*param_space.values()):
            candidate = dict(zip(param_space.keys(), combo))
            key = _key(candidate)
            if key not in seen_keys:
                seen_keys.add(key)
                population.append(candidate)
            if len(population) >= target_pop:
                break

    history: list[dict] = []
    best_params: dict = dict(population[0])
    best_score = -math.inf

    for gen in range(n_generations):
        # score in order; a None means the eval budget is spent on a new individual.
        scored: list[tuple[dict, float]] = []
        for ind in population:
            value = evaluate(ind)
            if value is not None:
                scored.append((ind, value))
        if not scored:
            break

        scored.sort(key=lambda pair: pair[1], reverse=True)
        if scored[0][1] > best_score:
            best_score = scored[0][1]
            best_params = dict(scored[0][0])

        # history['best'] is the running max -> guaranteed monotone non-decreasing.
        history.append(
            {
                "gen": gen,
                "best": best_score,
                "mean": float(np.mean([s for _, s in scored])),
            }
        )

        if not budget_left():
            break

        # 'scored' is already sorted best-first, so the elites are the leading entries.
        population = [ind for ind, _ in scored]
        scores = [s for _, s in scored]

        # ---- breed the next generation ------------------------------------ #
        n_elite = min(elitism, len(population))
        next_pop: list[dict] = [dict(population[i]) for i in range(n_elite)]
        while len(next_pop) < len(population):
            parent_a = _tournament(population, scores, rng, tournament_k)
            parent_b = _tournament(population, scores, rng, tournament_k)
            child = _crossover(parent_a, parent_b, rng)
            child = _mutate(child, param_space, rng, mutation_rate)
            next_pop.append(child)
        population = next_pop

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
        "n_evals": len(cache),
        "evaluated": dict(cache),
    }


# --------------------------------------------------------------------------- #
# optional multi-objective nicety
# --------------------------------------------------------------------------- #
def pareto_front(
    evaluated: dict,
    score_fn_secondary: Optional[Callable[[dict], float]] = None,
) -> list:
    """Return the non-dominated (high primary score, low secondary cost) parameter set.

    ``evaluated`` maps ``_key(params)`` -> primary score. If ``score_fn_secondary`` is
    None this just returns ``[(key, score)]`` for the single best primary score. Otherwise
    a point dominates another when its primary score is >= and its secondary cost is <=
    with at least one strict; the returned list holds the non-dominated points as
    ``(key, score, cost)`` tuples, sorted by ascending cost. Kept deliberately simple.
    """
    if not evaluated:
        return []

    if score_fn_secondary is None:
        best_key = max(evaluated, key=lambda k: evaluated[k])
        return [(best_key, evaluated[best_key])]

    # reconstruct params from the canonical key to evaluate the secondary objective.
    points = []
    for key, score in evaluated.items():
        params = dict(_eval_key(key))
        cost = float(score_fn_secondary(params))
        points.append((key, float(score), cost))

    front = []
    for key_i, score_i, cost_i in points:
        dominated = False
        for key_j, score_j, cost_j in points:
            if key_j == key_i:
                continue
            if score_j >= score_i and cost_j <= cost_i and (score_j > score_i or cost_j < cost_i):
                dominated = True
                break
        if not dominated:
            front.append((key_i, score_i, cost_i))
    front.sort(key=lambda p: p[2])
    return front


def _eval_key(key: str):
    """Inverse of ``_key``: safely parse the canonical sorted-tuple string back to a tuple."""
    import ast

    return ast.literal_eval(key)


# --------------------------------------------------------------------------- #
# self-test: a discrete problem with a KNOWN optimum
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    space = {
        "window": [5, 10, 20, 50, 100, 200],
        "stat": ["mean", "std", "zscore", "skew"],
        "decay": [0.9, 0.95, 0.99, 0.999],
    }
    # True optimum: window=50, stat="zscore", decay=0.99 -> score == 0.0 (the max).
    true_best = {"window": 50, "stat": "zscore", "decay": 0.99}

    def score(params: dict) -> float:
        window = params["window"]
        stat = params["stat"]
        decay = params["decay"]
        s = -((window - 50) ** 2) / 1000.0
        s -= 0.5 if stat != "zscore" else 0.0
        s -= ((decay - 0.99) ** 2) * 100.0
        return s

    true_opt = score(true_best)
    full_grid = grid_size(space)

    result = evolve_params(
        space,
        score,
        pop_size=16,
        n_generations=10,
        seed=8,
        elitism=2,
        mutation_rate=0.3,
        tournament_k=3,
    )

    print("--- evolve_params self-test ---")
    print("grid_size (exhaustive)   :", full_grid)
    print("n_evals (unique)         :", result["n_evals"])
    print("best_params              :", result["best_params"])
    print("best_score               :", round(result["best_score"], 6))
    print("true optimum params      :", true_best)
    print("true optimum score       :", round(true_opt, 6))

    bests = [h["best"] for h in result["history"]]
    monotone = all(bests[i] <= bests[i + 1] for i in range(len(bests) - 1))
    print("history best (per gen)   :", [round(b, 4) for b in bests])
    print("history best monotone    :", monotone)

    # secondary-objective demo: prefer smaller windows as a 'cost'.
    front = pareto_front(result["evaluated"], score_fn_secondary=lambda p: float(p["window"]))
    print("pareto front size        :", len(front))

    tol = 1e-9
    assert result["best_score"] >= true_opt - 1e-3, "best score below true optimum"
    assert abs(result["best_score"] - true_opt) <= 1e-3, "did not reach (near-)optimal score"
    assert result["n_evals"] < full_grid, "did not save evaluations vs exhaustive grid"
    assert monotone, "history best is not monotone non-decreasing"
    print("PASS: reached near-optimal score (within tol)")
    print("PASS: n_evals (%d) < grid_size (%d)" % (result["n_evals"], full_grid))
    print("PASS: history best is monotone non-decreasing")
