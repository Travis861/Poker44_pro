from __future__ import annotations

from collections import Counter
from typing import Any

import math
import statistics


MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")
AGGRESSIVE_ACTIONS = ("bet", "raise")
PASSIVE_ACTIONS = ("call", "check")
POSITION_NAMES = ("button", "small_blind", "big_blind", "early", "middle", "late", "unknown")
POSTFLOP_STREETS = ("flop", "turn", "river")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(statistics.fmean(values)),
        f"{prefix}_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def _action_counts(actions: list[dict[str, Any]]) -> Counter[str]:
    return Counter((action.get("action_type") or "").lower() for action in actions)


def _street_count(streets: list[Any], actions: list[dict[str, Any]]) -> float:
    if streets:
        return float(len(streets))
    street_names = {
        (action.get("street") or "").lower()
        for action in actions
        if action.get("street")
    }
    street_names.discard("")
    return float(len(street_names))


def _street_depth_ratio(street_count: float) -> float:
    # Hold'em typically has preflop, flop, turn, river.
    return clamp01(street_count / 4.0)


def _position_name(seat: int | None, button_seat: int | None, n_players: int) -> str:
    if not seat or not button_seat or n_players < 2:
        return "unknown"
    distance = (seat - button_seat) % max(n_players, 1)
    if distance == 0:
        return "button"
    if distance == 1:
        return "small_blind"
    if distance == 2:
        return "big_blind"
    if n_players <= 3:
        return "late" if distance >= n_players - 1 else "middle"
    if distance <= 3:
        return "early"
    if distance >= n_players - 1:
        return "late"
    return "middle"


def _seat_action_profile(actions: list[dict[str, Any]], seat: int | None) -> dict[str, float]:
    if not seat:
        return {
            "hero_action_count": 0.0,
            "hero_vpip_proxy": 0.0,
            "hero_raise_freq": 0.0,
            "hero_fold_freq": 0.0,
            "hero_aggression_ratio": 0.0,
        }

    seat_actions = [action for action in actions if action.get("actor_seat") == seat]
    counts = _action_counts(seat_actions)
    meaningful = max(1, sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    aggressive = sum(counts.get(kind, 0) for kind in AGGRESSIVE_ACTIONS)
    passive = sum(counts.get(kind, 0) for kind in PASSIVE_ACTIONS)
    vpip = counts.get("call", 0) + counts.get("bet", 0) + counts.get("raise", 0)
    return {
        "hero_action_count": float(len(seat_actions)),
        "hero_vpip_proxy": safe_div(vpip, meaningful),
        "hero_raise_freq": safe_div(counts.get("raise", 0), meaningful),
        "hero_fold_freq": safe_div(counts.get("fold", 0), meaningful),
        "hero_aggression_ratio": safe_div(aggressive, aggressive + passive),
    }


def _normalized_entropy(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    probs = [value / total for value in positives]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return safe_div(entropy, math.log(len(probs)))


def _first_postblind_preflop_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for action in actions:
        action_type = (action.get("action_type") or "").lower()
        if (action.get("street") or "").lower() != "preflop":
            continue
        if action_type in {"small_blind", "big_blind", "ante", "straddle"}:
            continue
        return action
    return None


def _street_openers(actions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    openers: dict[str, dict[str, Any]] = {}
    for action in actions:
        street = (action.get("street") or "").lower()
        if street and street not in openers:
            openers[street] = action
    return openers


def _action_transition_entropy(action_types: list[str]) -> float:
    if len(action_types) <= 1:
        return 0.0
    transitions = Counter(zip(action_types[:-1], action_types[1:]))
    return _normalized_entropy(list(transitions.values()))


def hand_features(hand: dict[str, Any]) -> dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    metadata = hand.get("metadata") or {}

    counts = _action_counts(actions)
    action_types = [(action.get("action_type") or "").lower() for action in actions]
    meaningful = max(1, sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    aggressive = sum(counts.get(kind, 0) for kind in AGGRESSIVE_ACTIONS)
    passive = sum(counts.get(kind, 0) for kind in PASSIVE_ACTIONS)

    amounts_bb = [
        safe_float(action.get("normalized_amount_bb"))
        for action in actions
        if action.get("normalized_amount_bb") is not None
    ]
    bet_like_sizes = [
        safe_float(action.get("normalized_amount_bb"))
        for action in actions
        if (action.get("action_type") or "").lower() in AGGRESSIVE_ACTIONS
        and action.get("normalized_amount_bb") is not None
    ]
    pot_after_values = [
        safe_float(action.get("pot_after"))
        for action in actions
        if action.get("pot_after") is not None
    ]
    actor_seats = [
        int(action.get("actor_seat"))
        for action in actions
        if action.get("actor_seat") is not None
    ]
    player_stacks = [
        safe_float(player.get("starting_stack"))
        for player in players
        if player.get("starting_stack") is not None
    ]
    revealed = sum(1 for player in players if player.get("showed_hand"))

    street_count = _street_count(streets, actions)
    button_seat = metadata.get("button_seat")
    hero_seat = metadata.get("hero_seat")
    bb_size = safe_float(metadata.get("bb"), 1.0) or 1.0
    hero_position = _position_name(hero_seat, button_seat, len(players))
    hero_profile = _seat_action_profile(actions, hero_seat)

    final_pot = safe_float(outcome.get("total_pot")) or (max(pot_after_values) if pot_after_values else 0.0)
    final_pot_bb = safe_div(final_pot, bb_size)
    player_stacks_bb = [safe_div(stack, bb_size) for stack in player_stacks]
    street_action_counts = Counter(
        (action.get("street") or "").lower()
        for action in actions
        if action.get("street")
    )
    actor_counts = Counter(actor_seats)
    meaningful_action_count = float(sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    size_buckets = Counter(int(round(size * 2.0)) for size in bet_like_sizes if size > 0.0)
    preflop_opener = _first_postblind_preflop_action(actions)
    street_openers = _street_openers(actions)
    donk_bet_count = 0
    donk_bet_opportunities = 0
    previous_street_actor: int | None = None
    for street in POSTFLOP_STREETS:
        opener = street_openers.get(street)
        if opener is None:
            continue
        opener_type = (opener.get("action_type") or "").lower()
        opener_actor = opener.get("actor_seat")
        if opener_type in AGGRESSIVE_ACTIONS:
            donk_bet_opportunities += 1
            if previous_street_actor is not None and opener_actor != previous_street_actor:
                donk_bet_count += 1
        street_actions = [a for a in actions if (a.get("street") or "").lower() == street]
        aggressive_actors = [
            a.get("actor_seat")
            for a in street_actions
            if (a.get("action_type") or "").lower() in AGGRESSIVE_ACTIONS
        ]
        previous_street_actor = aggressive_actors[-1] if aggressive_actors else previous_street_actor
    limp_flag = 0.0
    if preflop_opener is not None and (preflop_opener.get("action_type") or "").lower() == "call":
        limp_flag = 1.0
    first_action_aggressive = 1.0 if action_types and action_types[0] in AGGRESSIVE_ACTIONS else 0.0

    feats = {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "n_streets": street_count,
        "street_depth_ratio": _street_depth_ratio(street_count),
        "showdown": 1.0 if outcome.get("showdown") else 0.0,
        "revealed_players_ratio": safe_div(revealed, max(len(players), 1)),
        "call_ratio": safe_div(counts.get("call", 0), meaningful),
        "check_ratio": safe_div(counts.get("check", 0), meaningful),
        "bet_ratio": safe_div(counts.get("bet", 0), meaningful),
        "raise_ratio": safe_div(counts.get("raise", 0), meaningful),
        "fold_ratio": safe_div(counts.get("fold", 0), meaningful),
        "passive_ratio": safe_div(passive, meaningful),
        "aggression_ratio": safe_div(aggressive, aggressive + passive),
        "vpip_proxy": safe_div(
            counts.get("call", 0) + counts.get("bet", 0) + counts.get("raise", 0),
            meaningful,
        ),
        "raise_frequency": safe_div(counts.get("raise", 0), meaningful),
        "fold_to_action_tendency": safe_div(counts.get("fold", 0), aggressive + counts.get("call", 0)),
        "bet_like_count": float(aggressive),
        "avg_action_size_bb": float(statistics.fmean(amounts_bb)) if amounts_bb else 0.0,
        "avg_bet_size_bb": float(statistics.fmean(bet_like_sizes)) if bet_like_sizes else 0.0,
        "bet_size_std_bb": float(statistics.pstdev(bet_like_sizes)) if len(bet_like_sizes) > 1 else 0.0,
        "max_bet_size_bb": float(max(bet_like_sizes)) if bet_like_sizes else 0.0,
        "action_size_std_bb": float(statistics.pstdev(amounts_bb)) if len(amounts_bb) > 1 else 0.0,
        "bet_size_bucket_entropy": _normalized_entropy(list(size_buckets.values())),
        "final_pot": final_pot,
        "final_pot_bb": final_pot_bb,
        "avg_starting_stack": float(statistics.fmean(player_stacks)) if player_stacks else 0.0,
        "avg_starting_stack_bb": float(statistics.fmean(player_stacks_bb)) if player_stacks_bb else 0.0,
        "short_stack_ratio": safe_div(sum(1 for stack in player_stacks_bb if stack <= 20.0), max(len(player_stacks_bb), 1)),
        "all_in_like_ratio": safe_div(sum(1 for size in bet_like_sizes if size >= 20.0), max(len(bet_like_sizes), 1)),
        "preflop_action_ratio": safe_div(street_action_counts.get("preflop", 0), max(len(actions), 1)),
        "flop_action_ratio": safe_div(street_action_counts.get("flop", 0), max(len(actions), 1)),
        "turn_action_ratio": safe_div(street_action_counts.get("turn", 0), max(len(actions), 1)),
        "river_action_ratio": safe_div(street_action_counts.get("river", 0), max(len(actions), 1)),
        "action_entropy": _normalized_entropy([counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS]),
        "street_entropy": _normalized_entropy(list(street_action_counts.values())),
        "action_transition_entropy": _action_transition_entropy(action_types),
        "actor_count_ratio": safe_div(len(actor_counts), max(len(players), 1)),
        "actor_concentration": safe_div(max(actor_counts.values()) if actor_counts else 0.0, max(len(actions), 1)),
        "heads_up_flag": 1.0 if len(players) == 2 else 0.0,
        "full_ring_flag": 1.0 if len(players) >= 6 else 0.0,
        "deep_stack_flag": 1.0 if (player_stacks_bb and statistics.fmean(player_stacks_bb) >= 100.0) else 0.0,
        "aggressive_to_pot_ratio": safe_div(sum(bet_like_sizes), final_pot_bb if final_pot_bb > 0 else 1.0),
        "meaningful_action_count": meaningful_action_count,
        "donk_bet_rate": safe_div(donk_bet_count, donk_bet_opportunities),
        "limp_flag": limp_flag,
        "first_action_aggressive": first_action_aggressive,
        "hero_on_button": 1.0 if hero_position == "button" else 0.0,
        "hero_in_blinds": 1.0 if hero_position in {"small_blind", "big_blind"} else 0.0,
        "hero_early_position": 1.0 if hero_position == "early" else 0.0,
        "hero_middle_position": 1.0 if hero_position == "middle" else 0.0,
        "hero_late_position": 1.0 if hero_position == "late" else 0.0,
        "hero_position_known": 1.0 if hero_position != "unknown" else 0.0,
    }
    feats.update(hero_profile)
    return feats


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"chunk_size": 0.0}

    per_hand = [hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    out = {"chunk_size": float(len(chunk))}
    for name in feature_names:
        values = [row[name] for row in per_hand]
        out.update(summarize(values, name))

    vpip_vals = [row["vpip_proxy"] for row in per_hand]
    aggr_vals = [row["aggression_ratio"] for row in per_hand]
    raise_vals = [row["raise_frequency"] for row in per_hand]
    fold_vals = [row["fold_to_action_tendency"] for row in per_hand]
    stack_vals = [row["avg_starting_stack_bb"] for row in per_hand]
    pot_vals = [row["final_pot_bb"] for row in per_hand]

    out["consistency_score"] = safe_div(
        1.0,
        1.0
        + (statistics.pstdev(vpip_vals) if len(vpip_vals) > 1 else 0.0)
        + (statistics.pstdev(aggr_vals) if len(aggr_vals) > 1 else 0.0)
        + (statistics.pstdev(raise_vals) if len(raise_vals) > 1 else 0.0),
    )
    out["vpip_aggr_gap_mean"] = float(statistics.fmean(v - a for v, a in zip(vpip_vals, aggr_vals)))
    out["fold_raise_gap_mean"] = float(statistics.fmean(f - r for f, r in zip(fold_vals, raise_vals)))
    out["showdown_rate"] = float(statistics.fmean(row["showdown"] for row in per_hand))
    out["deep_street_rate"] = float(statistics.fmean(1.0 if row["n_streets"] >= 3.0 else 0.0 for row in per_hand))
    out["avg_players"] = float(statistics.fmean(row["n_players"] for row in per_hand))
    out["avg_actions"] = float(statistics.fmean(row["n_actions"] for row in per_hand))
    out["avg_streets"] = float(statistics.fmean(row["n_streets"] for row in per_hand))
    out["avg_stack_to_pot_ratio"] = float(
        statistics.fmean(safe_div(stack, pot if pot > 0 else 1.0) for stack, pot in zip(stack_vals, pot_vals))
    )
    out["action_entropy_mean"] = float(statistics.fmean(row["action_entropy"] for row in per_hand))
    out["street_entropy_mean"] = float(statistics.fmean(row["street_entropy"] for row in per_hand))
    out["actor_concentration_mean"] = float(statistics.fmean(row["actor_concentration"] for row in per_hand))
    out["bet_size_cv_mean"] = float(
        statistics.fmean(
            safe_div(row["bet_size_std_bb"], row["avg_bet_size_bb"] if row["avg_bet_size_bb"] > 0 else 1.0)
            for row in per_hand
        )
    )
    out["donk_bet_rate_mean"] = float(statistics.fmean(row["donk_bet_rate"] for row in per_hand))
    out["limp_rate"] = float(statistics.fmean(row["limp_flag"] for row in per_hand))
    out["first_action_aggressive_rate"] = float(
        statistics.fmean(row["first_action_aggressive"] for row in per_hand)
    )
    out["bet_size_bucket_entropy_mean"] = float(
        statistics.fmean(row["bet_size_bucket_entropy"] for row in per_hand)
    )
    out["action_transition_entropy_mean"] = float(
        statistics.fmean(row["action_transition_entropy"] for row in per_hand)
    )
    out["log_chunk_size"] = math.log1p(len(chunk))
    return out
