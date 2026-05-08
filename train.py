from pypokerengine.api.game import setup_config, start_poker
from call_player import CallPlayer
from equity_player import EquityPlayer
from fold_player import FoldPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from submission.custom_player import CustomPlayer, EvaluationWeights
from value_player import ValuePlayer
import json
import random
import time
from collections import deque
from dataclasses import fields, replace
from pathlib import Path


WEIGHT_ATTRS = tuple(field.name for field in fields(EvaluationWeights))
WEIGHT_BOUNDS = {"equity": (0.02, 8.0), "hand": (0.02, 8.0), "potential": (0.02, 8.0), "pot": (0.02, 8.0), "behavior": (0.10, 4.0), "cost": (0.03, 1.5), "opponent": (0.10, 8.0), "opponent_equity": (0.0, 4.0), "pressure": (0.02, 8.0), "risk": (0.05, 2.0)}
BASELINE_OPPONENTS = {"call": CallPlayer, "equity": EquityPlayer, "fold": FoldPlayer, "raise": RaisedPlayer, "random": RandomPlayer, "value": ValuePlayer}
MAX_INT = (2 ** 30) - 1


def format_weights(weights):
    values = ", ".join(f"{attr}={getattr(weights, attr):.2f}" for attr in WEIGHT_ATTRS)
    return f"EvaluationWeights({values})"


def weights_to_dict(weights):
    return {attr: float(getattr(weights, attr)) for attr in WEIGHT_ATTRS}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True)


def random_evaluation_weights(rng):
    return EvaluationWeights(**{attr: rng.uniform(*WEIGHT_BOUNDS[attr]) for attr in WEIGHT_ATTRS})


def scaled_step_cap(attr, delta_t):
    low, high = WEIGHT_BOUNDS[attr]
    full_range = max(upper - lower for lower, upper in WEIGHT_BOUNDS.values())
    attr_range = high - low
    return delta_t * (attr_range / full_range)


def player_with_weights(weights, seed, max_depth):
    player = CustomPlayer()
    player.search_config.weights = replace(weights)
    player.search_config.random_seed = seed
    player.search_config.max_depth = max_depth
    return player


def self_opponent(name, weights):
    return {"type": "weights", "name": name, "weights": replace(weights)}


def baseline_opponent(name):
    return {"type": "baseline", "name": name, "player": BASELINE_OPPONENTS[name]}


def log_opponent(opponent):
    if opponent["type"] == "weights":
        return {"type": "weights", "name": opponent["name"], "weights": weights_to_dict(opponent["weights"])}
    return {"type": "baseline", "name": opponent["name"]}


def player_to_opponent(opponent, seed, max_depth):
    if opponent["type"] == "weights":
        return player_with_weights(opponent["weights"], seed, max_depth=max_depth)
    return opponent["player"]()


def game_ev(player_weights, opponent, player_order, max_round, initial_stack, small_blind, player_seed, opponent_seed, max_depth):
    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind)
    player = player_with_weights(player_weights, player_seed, max_depth=max_depth)
    opponent = player_to_opponent(opponent, opponent_seed, max_depth=max_depth)

    if player_order == 0:
        config.register_player(name="Player", algorithm=player)
        config.register_player(name="Opponent", algorithm=opponent)
        player_idx = 0
    else:
        config.register_player(name="Opponent", algorithm=opponent)
        config.register_player(name="Player", algorithm=player)
        player_idx = 1

    result = start_poker(config, verbose=0)
    player_stack = int(result["players"][player_idx]["stack"])
    return float(player_stack - initial_stack)


def average_player_ev(player_weights, opponent, max_round, initial_stack, small_blind, seed, max_depth):
    ev0 = game_ev(player_weights, opponent, player_order=0, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, player_seed=seed, opponent_seed=seed + 1, max_depth=max_depth)
    ev1 = game_ev(player_weights, opponent, player_order=1, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, player_seed=seed, opponent_seed=seed + 1, max_depth=max_depth)
    return (ev0 + ev1) / 2


def average_reward(player_weights, opponents, games_per_opponent, max_round, initial_stack, small_blind, rng, max_depth):
    if not opponents:
        return 0.0

    total = 0.0
    n = 0

    for opponent in opponents:
        for i in range(games_per_opponent):
            seed = rng.randint(0, MAX_INT)
            total += average_player_ev(player_weights, opponent, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, seed=seed, max_depth=max_depth)
            n += 1

    return total / max(1, n)


def perturb(weights, attr, delta):
    base = getattr(weights, attr)
    low, high = WEIGHT_BOUNDS[attr]
    max_val = min(high, base + delta)
    min_val = max(low, base - delta)
    span = max_val - min_val
    return replace(weights, **{attr: max_val}), replace(weights, **{attr: min_val}), span


def get_opponents(current, history, rng, n_history_sample, opponent_names):
    opponents = []

    if "self" in opponent_names:
        opponents.append(self_opponent("self_current", current))
        history_lst = list(history)
        if history_lst and n_history_sample > 0:
            k = min(n_history_sample, len(history_lst))
            for idx, weights in enumerate(rng.sample(history_lst, k)):
                opponents.append(self_opponent(f"self_history_{idx}", weights))

    for name in opponent_names:
        if name != "self":
            opponents.append(baseline_opponent(name))

    return opponents


def train(iterations, delta, delta_decay_ratio, step_scale, history_maxlen, n_history_opponents, games_per_opp, max_round, initial_stack, small_blind, max_depth, seed, verbose, initial_random_history, opponent_names):
    rng = random.Random(seed)
    weights = EvaluationWeights()
    history = deque(maxlen=history_maxlen)
    for i in range(max(0, initial_random_history)):
        history.append(random_evaluation_weights(rng))

    trace = []
    best_weights = replace(weights)
    best_score = float("-inf")

    if verbose:
        if initial_random_history > 0:
            print(f"History queue seeded with {len(history)} random profile(s) (uniform within each coordinate's configured bounds):")
            for i, random_weights in enumerate(history):
                print(f"  seed[{i}]: {format_weights(random_weights)}")
        print(f"Starting weights: {format_weights(weights)}\n")

    for iteration in range(iterations):
        delta_t = delta * (delta_decay_ratio**iteration)
        start_weights = replace(weights)
        opponents = get_opponents(start_weights, history, rng, n_history_opponents, opponent_names)

        if verbose:
            print(f"\n[iter {iteration + 1}/{iterations}] current={format_weights(start_weights)} opponents={len(opponents)} delta_t={delta_t:.4f} ")

        updates = []
        for attr in WEIGHT_ATTRS:
            weights_plus, weights_minus, span = perturb(weights, attr, delta_t)

            mean_plus = average_reward(weights_plus, opponents, games_per_opponent=games_per_opp, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, rng=rng, max_depth=max_depth)
            mean_minus = average_reward(weights_minus, opponents, games_per_opponent=games_per_opp, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, rng=rng, max_depth=max_depth)

            skip_update = False
            diff = mean_plus - mean_minus
            base = getattr(weights, attr)

            if skip_update:
                raw_step = 0.0
                step = 0.0
                grad_est = 0.0
                new_val = base
            else:
                denom = span if span > 1e-9 else 2.0 * delta_t
                grad_est = diff / denom
                raw_step = step_scale * grad_est
                cap = scaled_step_cap(attr, delta_t)
                step = max(-cap, min(cap, raw_step))
                new_val = base + step
                new_val = max(WEIGHT_BOUNDS[attr][0], min(WEIGHT_BOUNDS[attr][1], new_val))
                weights = replace(weights, **{attr: new_val})

            if verbose:
                status = "SKIP" if skip_update else "APPLY"
                print(f"  - {attr:<8} {status} base={base:.2f} new={new_val:.2f} plus={mean_plus:+.2f} minus={mean_minus:+.2f} grad={grad_est:+.2f} step={step:+.2f}")

            updates.append({"attribute": attr, "status": "skip" if skip_update else "apply", "base": base, "new": new_val, "plus_weights": weights_to_dict(weights_plus), "minus_weights": weights_to_dict(weights_minus), "mean_plus": mean_plus, "mean_minus": mean_minus, "gradient_estimate": grad_est, "raw_step": raw_step, "step_cap": scaled_step_cap(attr, delta_t), "clipped_step": step, "span": span,})

        history.append(start_weights)

        if verbose:
            print(f"  end={format_weights(weights)}")

        iteration_score = average_reward(weights, opponents, games_per_opp=games_per_opp, max_round=max_round, initial_stack=initial_stack, small_blind=small_blind, rng=rng, max_depth=max_depth)
        if iteration_score > best_score:
            best_score = iteration_score
            best_weights = replace(weights)

        trace.append({
            "iteration": iteration + 1,
            "delta": delta_t,
            "start_weights": weights_to_dict(start_weights),
            "end_weights": weights_to_dict(weights),
            "mean_return": iteration_score,
            "best_mean_return_so_far": best_score,
            "opponents": [log_opponent(opponent) for opponent in opponents],
            "updates": updates,
        })

        if verbose:
            print(f"  validation_mean_return={iteration_score:+.2f}")

    return weights, {"final_weights": weights_to_dict(weights), "best_weights": weights_to_dict(best_weights), "best_mean_return": best_score, "history": trace}


def main():
    iterations = 10
    delta = 0.5
    step_scale = 0.005
    delta_decay_ratio = 0.9
    history_len = 32
    history_opponents = 4
    initial_random_history = 6
    opponents = ["self", "call", "value", "equity", "fold"]
    games_per_opp = 3
    rounds = 50
    stack = 4000
    small_blind = 10
    max_depth = 2
    seed = 42
    quiet = False
    output = Path("log_experiment.json")

    start = time.time()
    final_weights, results = train(iterations=iterations, delta=delta, delta_decay_ratio=delta_decay_ratio, step_scale=step_scale, history_maxlen=history_len, n_history_opponents=history_opponents, games_per_opp=games_per_opp, max_round=rounds, initial_stack=stack, small_blind=small_blind, max_depth=max_depth, seed=seed, verbose=not quiet, initial_random_history=initial_random_history, opponent_names=opponents)
    results["elapsed_seconds"] = round(time.time() - start, 3)
    results["settings"] = {"iterations": iterations, "delta": delta, "step_scale": step_scale, "delta_decay_ratio": delta_decay_ratio, "history_len": history_len, "history_opponents": history_opponents, "initial_random_history": initial_random_history, "games_per_opp": games_per_opp, "rounds": rounds, "stack": stack, "small_blind": small_blind, "max_depth": max_depth, "seed": seed, "opponents": opponents}

    write_json(output, results)

    if not quiet:
        print("Final weights:", format_weights(final_weights))
        print("Best weights:", format_weights(EvaluationWeights(**results["best_weights"])))
        print("Wrote", output)
    else:
        print(format_weights(final_weights))


if __name__ == "__main__":
    main()
