from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence, Tuple

# Repo root on sys.path so `python self_play_eval_weights.py` finds `submission`.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pypokerengine.api.game import setup_config, start_poker

from call_player import CallPlayer
from equity_player import EquityPlayer
from fold_pressure_player import FoldPressurePlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from submission.custom_player import CustomPlayer, EvaluationWeights
from value_player import ValuePlayer

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


WEIGHT_ATTRS: Tuple[str, ...] = tuple(field.name for field in fields(EvaluationWeights))
WEIGHT_COORD_LO = 0.02
WEIGHT_COORD_HI = 8.0
WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "behavior": (0.10, 4.0),
    "cost": (0.03, 1.5),
    "opponent": (0.10, 8.0),
    "opponent_equity": (0.0, 4.0),
    "risk": (0.05, 2.0),
}
BASELINE_OPPONENTS = {
    "call": CallPlayer,
    "equity": EquityPlayer,
    "fold_pressure": FoldPressurePlayer,
    "raise": RaisedPlayer,
    "random": RandomPlayer,
    "value": ValuePlayer,
}


def _fmt_weights(weights: EvaluationWeights) -> str:
    values = ", ".join(f"{attr}={getattr(weights, attr):.2f}" for attr in WEIGHT_ATTRS)
    return f"EvaluationWeights({values})"


def _weights_to_dict(weights: EvaluationWeights) -> Dict[str, float]:
    return {attr: float(getattr(weights, attr)) for attr in WEIGHT_ATTRS}


def _settings_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "iterations": args.iterations,
        "delta": args.delta,
        "step_scale": args.step_scale,
        "delta_decay_ratio": args.delta_decay_ratio,
        "history_len": args.history_len,
        "history_opponents": args.history_opponents,
        "initial_random_history": args.initial_random_history,
        "games_per_opp": args.games_per_opp,
        "rounds": args.rounds,
        "stack": args.stack,
        "sb": args.sb,
        "max_depth": args.max_depth,
        "chance_samples": args.chance_samples,
        "seed": args.seed,
        "opponents": args.opponents,
        "weight_bounds": {
            attr: list(_weight_bounds(attr))
            for attr in WEIGHT_ATTRS
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)


def _random_evaluation_weights(rng: random.Random) -> EvaluationWeights:
    return EvaluationWeights(**{
        attr: rng.uniform(*_weight_bounds(attr))
        for attr in WEIGHT_ATTRS
    })


def _weight_bounds(attr: str) -> Tuple[float, float]:
    return WEIGHT_BOUNDS.get(attr, (WEIGHT_COORD_LO, WEIGHT_COORD_HI))


def _scaled_step_cap(attr: str, delta_t: float) -> float:
    lo, hi = _weight_bounds(attr)
    full_range = WEIGHT_COORD_HI - WEIGHT_COORD_LO
    attr_range = hi - lo
    return delta_t * (attr_range / full_range)


def _player_with_weights(
    weights: EvaluationWeights,
    think_seed: int | None,
    *,
    max_depth: int,
    chance_samples: int,
) -> CustomPlayer:
    p = CustomPlayer(verbose_thinking=False)
    p.search_config.weights = replace(weights)
    p.search_config.random_seed = think_seed
    p.search_config.max_depth = max_depth
    p.search_config.chance_samples = chance_samples
    return p


def _weights_opponent(name: str, weights: EvaluationWeights) -> Dict[str, Any]:
    return {
        "type": "weights",
        "name": name,
        "weights": replace(weights),
    }


def _baseline_opponent(name: str) -> Dict[str, Any]:
    return {
        "type": "baseline",
        "name": name,
        "player_cls": BASELINE_OPPONENTS[name],
    }


def _opponent_to_log(opponent: Dict[str, Any]) -> Dict[str, Any]:
    if opponent["type"] == "weights":
        return {
            "type": "weights",
            "name": opponent["name"],
            "weights": _weights_to_dict(opponent["weights"]),
        }
    return {
        "type": "baseline",
        "name": opponent["name"],
    }


def _player_from_opponent(
    opponent: Dict[str, Any],
    think_seed: int | None,
    *,
    max_depth: int,
    chance_samples: int,
):
    if opponent["type"] == "weights":
        return _player_with_weights(
            opponent["weights"],
            think_seed,
            max_depth=max_depth,
            chance_samples=chance_samples,
        )
    return opponent["player_cls"]()


def _one_match_chip_ev_hero(
    hero_weights: EvaluationWeights,
    opponent: Dict[str, Any],
    *,
    hero_is_first_registered: bool,
    max_round: int,
    initial_stack: int,
    small_blind: int,
    hero_seed: int | None,
    opp_seed: int | None,
    max_depth: int,
    chance_samples: int,
) -> float:
    """Hero chip EV for one full game (one registration order)."""
    config = setup_config(
        max_round=max_round,
        initial_stack=initial_stack,
        small_blind_amount=small_blind,
    )
    hero = _player_with_weights(
        hero_weights, hero_seed, max_depth=max_depth, chance_samples=chance_samples
    )
    opp = _player_from_opponent(
        opponent, opp_seed, max_depth=max_depth, chance_samples=chance_samples
    )
    if hero_is_first_registered:
        config.register_player(name="Hero", algorithm=hero)
        config.register_player(name="Opp", algorithm=opp)
        hero_idx = 0
    else:
        config.register_player(name="Opp", algorithm=opp)
        config.register_player(name="Hero", algorithm=hero)
        hero_idx = 1

    result = start_poker(config, verbose=0)
    hero_stack = int(result["players"][hero_idx]["stack"])
    return float(hero_stack - initial_stack)


def balanced_hero_chip_ev(
    hero_weights: EvaluationWeights,
    opponent: Dict[str, Any],
    *,
    max_round: int,
    initial_stack: int,
    small_blind: int,
    base_seed: int,
    max_depth: int,
    chance_samples: int,
) -> float:
    """
    Average hero chip EV over both registration orders (same RNG-derived seeds
    for the two orders so variance is paired where possible).
    """
    ev0 = _one_match_chip_ev_hero(
        hero_weights,
        opponent,
        hero_is_first_registered=True,
        max_round=max_round,
        initial_stack=initial_stack,
        small_blind=small_blind,
        hero_seed=base_seed,
        opp_seed=base_seed + 1,
        max_depth=max_depth,
        chance_samples=chance_samples,
    )
    ev1 = _one_match_chip_ev_hero(
        hero_weights,
        opponent,
        hero_is_first_registered=False,
        max_round=max_round,
        initial_stack=initial_stack,
        small_blind=small_blind,
        hero_seed=base_seed,
        opp_seed=base_seed + 1,
        max_depth=max_depth,
        chance_samples=chance_samples,
    )
    return (ev0 + ev1) / 2.0


def mean_return_vs_opponents(
    hero_weights: EvaluationWeights,
    opponents: Sequence[Dict[str, Any]],
    *,
    games_per_opp: int,
    max_round: int,
    initial_stack: int,
    small_blind: int,
    rng: random.Random,
    max_depth: int,
    chance_samples: int,
    progress: bool = False,
    pbar_desc: str = "eval hero EV",
    pbar_leave: bool = False,
) -> float:
    """Mean balanced hero chip EV, averaged over opponents and repeated games."""
    if not opponents:
        return 0.0
    total = 0.0
    n = 0
    use_tqdm = bool(progress and tqdm is not None and len(opponents) > 0)
    opp_bar = (
        tqdm(
            total=len(opponents),
            desc=pbar_desc,
            leave=pbar_leave,
            unit="opp",
            dynamic_ncols=True,
        )
        if use_tqdm
        else None
    )
    try:
        for opp_idx, opponent in enumerate(opponents, start=1):
            if opp_bar is not None:
                opp_bar.set_postfix_str(f"opp={opponent['name']} ({opp_idx}/{len(opponents)})")
            for _ in range(games_per_opp):
                base_seed = rng.randint(0, 2**30 - 1)
                total += balanced_hero_chip_ev(
                    hero_weights,
                    opponent,
                    max_round=max_round,
                    initial_stack=initial_stack,
                    small_blind=small_blind,
                    base_seed=base_seed,
                    max_depth=max_depth,
                    chance_samples=chance_samples,
                )
                n += 1
            if opp_bar is not None:
                opp_bar.update(1)
    finally:
        if opp_bar is not None:
            opp_bar.close()
    return total / max(1, n)


def _perturb(
    w: EvaluationWeights, attr: str, delta: float
) -> Tuple[EvaluationWeights, EvaluationWeights, float]:
    """
    Returns (w_plus, w_minus, effective_span). `effective_span` is plus - minus
    on the perturbed coordinate (handles clamping at bounds).
    """
    base = getattr(w, attr)
    lo, hi = _weight_bounds(attr)
    plus_v = min(hi, base + delta)
    minus_v = max(lo, base - delta)
    span = plus_v - minus_v
    return replace(w, **{attr: plus_v}), replace(w, **{attr: minus_v}), span


def _opponent_pool(
    current: EvaluationWeights,
    history: Deque[EvaluationWeights],
    rng: random.Random,
    n_history_sample: int,
    opponent_names: Sequence[str],
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    if "self" in opponent_names:
        pool.append(_weights_opponent("self_current", current))
        hist_list = list(history)
        if hist_list and n_history_sample > 0:
            k = min(n_history_sample, len(hist_list))
            for history_index, weights in enumerate(rng.sample(hist_list, k)):
                pool.append(_weights_opponent(f"self_history_{history_index}", weights))

    for name in opponent_names:
        if name != "self":
            pool.append(_baseline_opponent(name))

    return pool


def train_self_play(
    *,
    iterations: int,
    delta: float,
    delta_decay_ratio: float,
    step_scale: float,
    history_maxlen: int,
    n_history_opponents: int,
    games_per_opp: int,
    max_round: int,
    initial_stack: int,
    small_blind: int,
    max_depth: int,
    chance_samples: int,
    seed: int,
    verbose: bool,
    initial_random_history: int,
    opponent_names: Sequence[str],
    progress: bool = True,
) -> Tuple[EvaluationWeights, Dict[str, Any]]:
    rng = random.Random(seed)
    w = EvaluationWeights()
    history: Deque[EvaluationWeights] = deque(maxlen=history_maxlen)
    for _ in range(max(0, initial_random_history)):
        history.append(_random_evaluation_weights(rng))

    trace: List[Dict[str, Any]] = []
    best_weights = replace(w)
    best_score = float("-inf")

    if verbose:
        if initial_random_history > 0:
            print(
                f"History queue seeded with {len(history)} random profile(s) "
                "(uniform within each coordinate's configured bounds):",
                flush=True,
            )
            for i, rw in enumerate(history):
                print(f"  seed[{i}]: {_fmt_weights(rw)}", flush=True)
        print(f"Starting weights: {_fmt_weights(w)}\n", flush=True)

    for it in range(iterations):
        delta_t = delta * (delta_decay_ratio**it)
        w_frozen = replace(w)
        opponents = _opponent_pool(w_frozen, history, rng, n_history_opponents, opponent_names)

        if verbose:
            print(
                f"\n[iter {it + 1}/{iterations}] current={_fmt_weights(w_frozen)} "
                f"opponents={len(opponents)} delta_t={delta_t:.4f} ",
                flush=True,
            )

        updates: List[Dict[str, Any]] = []
        for attr in WEIGHT_ATTRS:
            w_plus, w_minus, span = _perturb(w, attr, delta_t)

            mean_plus = mean_return_vs_opponents(
                w_plus,
                opponents,
                games_per_opp=games_per_opp,
                max_round=max_round,
                initial_stack=initial_stack,
                small_blind=small_blind,
                rng=rng,
                max_depth=max_depth,
                chance_samples=chance_samples,
                progress=progress,
                pbar_leave=False,
                pbar_desc=f"iter {it + 1}/{iterations} {attr} w+",
            )
            mean_minus = mean_return_vs_opponents(
                w_minus,
                opponents,
                games_per_opp=games_per_opp,
                max_round=max_round,
                initial_stack=initial_stack,
                small_blind=small_blind,
                rng=rng,
                max_depth=max_depth,
                chance_samples=chance_samples,
                progress=progress,
                pbar_leave=False,
                pbar_desc=f"iter {it + 1}/{iterations} {attr} w-",
            )

            skip_update = False
            diff = mean_plus - mean_minus
            base = getattr(w, attr)

            if skip_update:
                raw_step = 0.0
                step = 0.0
                grad_est = 0.0
                denom = span if span > 1e-9 else 2.0 * delta_t
                new_val = base
            else:
                denom = span if span > 1e-9 else 2.0 * delta_t
                grad_est = diff / denom
                raw_step = step_scale * grad_est
                step = raw_step
                cap = _scaled_step_cap(attr, delta_t)
                step = max(-cap, min(cap, step))
                new_val = base + step
                new_val = max(_weight_bounds(attr)[0], min(_weight_bounds(attr)[1], new_val))
                w = replace(w, **{attr: new_val})

            if verbose:
                status = "SKIP" if skip_update else "APPLY"
                print(
                    f"  - {attr:<8} {status} base={base:.2f} new={new_val:.2f} "
                    f"plus={mean_plus:+.2f} minus={mean_minus:+.2f} "
                    f"grad={grad_est:+.2f} step={step:+.2f}",
                    flush=True,
                )

            updates.append({
                "attribute": attr,
                "status": "skip" if skip_update else "apply",
                "base": base,
                "new": new_val,
                "plus_weights": _weights_to_dict(w_plus),
                "minus_weights": _weights_to_dict(w_minus),
                "mean_plus": mean_plus,
                "mean_minus": mean_minus,
                "gradient_estimate": grad_est,
                "raw_step": raw_step,
                "step_cap": _scaled_step_cap(attr, delta_t),
                "clipped_step": step,
                "span": span,
            })

        history.append(w_frozen)

        if verbose:
            print(f"  end={_fmt_weights(w)}", flush=True)

        iteration_score = mean_return_vs_opponents(
            w,
            opponents,
            games_per_opp=games_per_opp,
            max_round=max_round,
            initial_stack=initial_stack,
            small_blind=small_blind,
            rng=rng,
            max_depth=max_depth,
            chance_samples=chance_samples,
            progress=progress,
            pbar_leave=False,
            pbar_desc=f"iter {it + 1}/{iterations} final",
        )
        if iteration_score > best_score:
            best_score = iteration_score
            best_weights = replace(w)

        trace.append({
            "iteration": it + 1,
            "delta": delta_t,
            "start_weights": _weights_to_dict(w_frozen),
            "end_weights": _weights_to_dict(w),
            "mean_return": iteration_score,
            "best_mean_return_so_far": best_score,
            "opponents": [_opponent_to_log(opponent) for opponent in opponents],
            "updates": updates,
        })

        if verbose:
            print(f"  validation_mean_return={iteration_score:+.2f}", flush=True)

    return w, {
        "final_weights": _weights_to_dict(w),
        "best_weights": _weights_to_dict(best_weights),
        "best_mean_return": best_score,
        "history": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-play tuning of CustomPlayer linear bucket EvaluationWeights",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="outer loops; each loop updates every EvaluationWeights coordinate",
    )
    parser.add_argument("--delta", type=float, default=0.5, help="symmetric finite-difference step")
    parser.add_argument(
        "--step-scale",
        type=float,
        default=0.005,
        help="multiplier on finite-difference gradient estimate",
    )
    parser.add_argument(
        "--delta-decay-ratio",
        type=float,
        default=0.9,
        help="per-iteration multiplier for delta_t = delta0 * ratio^iter (0,1]",
    )
    parser.add_argument("--history-len", type=int, default=32)
    parser.add_argument(
        "--history-opponents",
        type=int,
        default=4,
        help="how many past weight vectors to sample as opponents per iter",
    )
    parser.add_argument(
        "--initial-random-history",
        type=int,
        default=6,
        help="number of random EvaluationWeights to enqueue before training (0 disables)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=sorted(["self"] + list(BASELINE_OPPONENTS.keys())),
        default=["self", "call", "value", "equity", "fold_pressure"],
        help="opponents used to score each candidate weight vector",
    )
    parser.add_argument("--games-per-opp", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=50, help="hands per match")
    parser.add_argument("--stack", type=int, default=4000)
    parser.add_argument("--sb", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--chance-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log.json"),
        help="JSON file where training settings, history, and weights are written",
    )
    args = parser.parse_args()
    if args.delta <= 0.0:
        raise ValueError("--delta must be > 0")
    if not (0.0 < args.delta_decay_ratio <= 1.0):
        raise ValueError("--delta-decay-ratio must be in (0, 1]")

    show_progress = (not args.quiet) and (not args.no_progress)
    start = time.time()

    final_w, results = train_self_play(
        iterations=args.iterations,
        delta=args.delta,
        delta_decay_ratio=args.delta_decay_ratio,
        step_scale=args.step_scale,
        history_maxlen=args.history_len,
        n_history_opponents=args.history_opponents,
        games_per_opp=args.games_per_opp,
        max_round=args.rounds,
        initial_stack=args.stack,
        small_blind=args.sb,
        max_depth=args.max_depth,
        chance_samples=args.chance_samples,
        seed=args.seed,
        verbose=not args.quiet,
        initial_random_history=args.initial_random_history,
        opponent_names=args.opponents,
        progress=show_progress,
    )
    results["elapsed_seconds"] = round(time.time() - start, 3)
    results["settings"] = _settings_to_dict(args)
    _write_json(args.output, results)

    if not args.quiet:
        print("Final weights:", _fmt_weights(final_w), flush=True)
        print("Best weights:", _fmt_weights(EvaluationWeights(**results["best_weights"])), flush=True)
        print("Wrote", args.output, flush=True)
    else:
        print(_fmt_weights(final_w))


if __name__ == "__main__":
    main()
