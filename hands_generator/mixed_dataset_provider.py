from __future__ import annotations

import hashlib
import gzip
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import bittensor as bt

from hands_generator.bot_hands.generate_poker_data import BotProfile
from hands_generator.data_generator import _default_bot_profiles, generate_bot_chunk
from poker44.core.hand_json import from_standard_json
from poker44.core.models import LabeledHandBatch
from poker44.validator.sanitization import (
    sanitize_hand_for_miner,
    sanitized_chunk_signature,
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_JSON_PATH = REPO_ROOT / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "validator_mixed_chunks.json"
UTC = timezone.utc


@dataclass
class MixedDatasetConfig:
    human_json_path: Path = DEFAULT_HUMAN_JSON_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    chunk_count: int = 80
    min_hands_per_chunk: int = 60
    max_hands_per_chunk: int = 120
    human_ratio: float = 0.5
    refresh_seconds: int = 60 * 60
    seed: Optional[int] = None
    # Bot generation robustness knobs
    bot_candidate_attempts_per_chunk: int = 8
    max_bot_generation_rounds: int = 4
    max_shortcut_rule_accuracy: float = 0.70


def _current_window_id(refresh_seconds: int, now: Optional[float] = None) -> int:
    if refresh_seconds <= 0:
        raise ValueError("refresh_seconds must be > 0")
    ts = time.time() if now is None else now
    return int(ts // refresh_seconds)


def _effective_seed(base_seed: Optional[int], window_id: int) -> int:
    seed_material = f"{0 if base_seed is None else int(base_seed)}:{window_id}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_hand_fingerprint(hand: Dict[str, Any]) -> str:
    payload = json.dumps(hand, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _window_effective_seed(
    cfg: MixedDatasetConfig, window_id: int, *, window_start_iso: Optional[str] = None
) -> int:
    return _effective_seed(cfg.seed, window_id)


def _window_start_iso_for_id(cfg: MixedDatasetConfig, window_id: int) -> str:
    anchor_ts = (window_id * cfg.refresh_seconds) + 1
    return datetime.fromtimestamp(anchor_ts, tz=UTC).isoformat()


def _window_human_sizes(
    cfg: MixedDatasetConfig, window_id: int, *, window_start_iso: Optional[str] = None
) -> List[int]:
    effective_seed = _window_effective_seed(
        cfg, window_id, window_start_iso=window_start_iso
    )
    rng = random.Random(effective_seed)

    n_human = int(round(cfg.chunk_count * cfg.human_ratio))
    n_human = max(1, min(cfg.chunk_count - 1, n_human))
    return _split_chunk_sizes(
        rng, n_human, cfg.min_hands_per_chunk, cfg.max_hands_per_chunk
    )


def _iter_top_level_array_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Yield object JSON strings from a top-level JSON array without loading the whole file."""
    if path.suffix != ".gz":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected top-level JSON array in {path}")
        for item in payload:
            yield json.dumps(item, ensure_ascii=False)
        return

    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("rt", encoding="utf-8")
    with handle as f:
        in_string = False
        escape = False
        depth = 0
        collecting = False
        buf: List[str] = []
        seen_array_start = False

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for ch in chunk:
                if collecting:
                    buf.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue

                if ch == "[":
                    depth += 1
                    if depth == 1:
                        seen_array_start = True
                    continue

                if ch == "{":
                    if seen_array_start and depth == 1 and not collecting:
                        collecting = True
                        buf = ["{"]
                    depth += 1
                    continue

                if ch == "}":
                    depth -= 1
                    if collecting and depth == 1:
                        yield "".join(buf)
                        collecting = False
                        buf = []
                    continue

                if ch == "]":
                    depth -= 1


def _is_valid_human_hand(hand: Dict[str, Any]) -> bool:
    players = hand.get("players")
    actions = hand.get("actions")
    if not isinstance(players, list) or len(players) < 2:
        return False
    if not isinstance(actions, list) or len(actions) == 0:
        return False
    return True


def _reservoir_sample_humans(path: Path, sample_size: int, rng: random.Random) -> List[Dict[str, Any]]:
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    for raw in _iter_top_level_array_objects(path):
        try:
            hand = json.loads(raw)
        except Exception:
            continue

        if not _is_valid_human_hand(hand):
            continue

        # Force canonical label for the validator pipeline.
        hand["label"] = "human"

        if len(reservoir) < sample_size:
            reservoir.append(hand)
        else:
            j = rng.randint(0, seen)
            if j < sample_size:
                reservoir[j] = hand
        seen += 1

    return reservoir


def _deterministic_human_selection(
    path: Path,
    sample_size: int,
    cfg: MixedDatasetConfig,
    window_id: int,
) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in {path}")

    valid_hands: List[Dict[str, Any]] = []
    for raw_hand in payload:
        if not isinstance(raw_hand, dict) or not _is_valid_human_hand(raw_hand):
            continue
        hand = dict(raw_hand)
        hand["label"] = "human"
        valid_hands.append(hand)

    if not valid_hands:
        raise RuntimeError("Could not sample any valid human hands from source JSON")

    secret = str(cfg.seed or 0)
    ordered_hands = sorted(
        valid_hands,
        key=lambda hand: hashlib.sha256(
            f"{secret}:{_stable_hand_fingerprint(hand)}".encode("utf-8")
        ).hexdigest(),
    )

    # Advance by full sample windows so consecutive windows avoid overlap
    # whenever enough unique human hands exist.
    offset = (window_id * sample_size) % len(ordered_hands)
    selected: List[Dict[str, Any]] = []
    for index in range(sample_size):
        selected.append(ordered_hands[(offset + index) % len(ordered_hands)])
    return selected


def _split_chunk_sizes(rng: random.Random, n_chunks: int, min_hands: int, max_hands: int) -> List[int]:
    return [rng.randint(min_hands, max_hands) for _ in range(n_chunks)]


def _paired_chunk_sizes(
    rng: random.Random,
    n_human: int,
    n_bot: int,
    min_hands: int,
    max_hands: int,
) -> Tuple[List[int], List[int]]:
    paired = min(n_human, n_bot)
    base_sizes = _split_chunk_sizes(rng, paired, min_hands, max_hands)
    human_sizes = list(base_sizes)
    bot_sizes = list(base_sizes)
    if n_human > paired:
        human_sizes.extend(_split_chunk_sizes(rng, n_human - paired, min_hands, max_hands))
    if n_bot > paired:
        bot_sizes.extend(_split_chunk_sizes(rng, n_bot - paired, min_hands, max_hands))
    return human_sizes, bot_sizes


def _compute_dataset_hash(labeled_chunks: List[Dict[str, Any]]) -> str:
    payload = json.dumps(labeled_chunks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chunk_behavior_signature(
    hands: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Return coarse per-hand behavior averages for matching bot/human chunk shape."""
    if not hands:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_calls = 0
    total_checks = 0
    total_raises = 0
    total_folds = 0
    total_actions = 0
    total_streets = 0
    total_players = 0
    total_action_amount = 0.0
    total_action_pot_after = 0.0
    for hand in hands:
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        total_players += len(players)
        total_actions += len(actions)
        for action in actions:
            action_type = action.get("action_type")
            total_action_amount += float(action.get("amount", 0.0) or 0.0)
            total_action_pot_after += float(action.get("pot_after", 0.0) or 0.0)
            if action_type == "call":
                total_calls += 1
            elif action_type == "check":
                total_checks += 1
            elif action_type == "raise":
                total_raises += 1
            elif action_type == "fold":
                total_folds += 1
        total_streets += len(hand.get("streets") or [])

    n = float(len(hands))
    return (
        total_calls / n,
        total_checks / n,
        total_raises / n,
        total_folds / n,
        total_actions / n,
        total_streets / n,
        total_players / n,
        total_action_amount / n,
        total_action_pot_after / n,
    )


def _sanitized_chunk_behavior_signature(
    hands: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    return sanitized_chunk_signature(hands)


def _chunk_structure_signature(
    hands: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Miner-visible chunk structure: ending street mix and occupied seat mix."""
    if not hands:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sanitized_hands = [sanitize_hand_for_miner(hand) for hand in hands]
    preflop = flop = turn = river = 0
    p2 = p3 = p4 = p5 = p6p = 0

    for hand in sanitized_hands:
        streets = hand.get("streets") or []
        street_count = len(streets)
        if street_count <= 0:
            preflop += 1
        elif street_count == 1:
            flop += 1
        elif street_count == 2:
            turn += 1
        else:
            river += 1

        player_count = len(hand.get("players") or [])
        if player_count <= 2:
            p2 += 1
        elif player_count == 3:
            p3 += 1
        elif player_count == 4:
            p4 += 1
        elif player_count == 5:
            p5 += 1
        else:
            p6p += 1

    n = float(len(sanitized_hands))
    return (
        preflop / n,
        flop / n,
        turn / n,
        river / n,
        p2 / n,
        p3 / n,
        p4 / n,
        p5 / n,
        p6p / n,
    )


def _signature_distance(
    a: Tuple[float, float, float, float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float, float, float, float],
) -> float:
    """Weighted distance between chunk signatures for human/bot matching."""
    (
        calls_a,
        checks_a,
        raises_a,
        folds_a,
        actions_a,
        streets_a,
        players_a,
        action_amount_a,
        pot_after_a,
    ) = a
    (
        calls_b,
        checks_b,
        raises_b,
        folds_b,
        actions_b,
        streets_b,
        players_b,
        action_amount_b,
        pot_after_b,
    ) = b
    return (
        abs(calls_a - calls_b) * 1.5
        + abs(checks_a - checks_b) * 1.4
        + abs(raises_a - raises_b) * 1.8
        + abs(folds_a - folds_b) * 2.6
        + abs(actions_a - actions_b) * 0.8
        + abs(streets_a - streets_b) * 2.4
        + abs(players_a - players_b) * 1.5
        + abs(action_amount_a - action_amount_b) * 0.6
        + abs(pot_after_a - pot_after_b) * 0.25
    )


def _structure_distance(
    a: Tuple[float, float, float, float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float, float, float, float],
) -> float:
    (
        preflop_a,
        flop_a,
        turn_a,
        river_a,
        p2_a,
        p3_a,
        p4_a,
        p5_a,
        p6p_a,
    ) = a
    (
        preflop_b,
        flop_b,
        turn_b,
        river_b,
        p2_b,
        p3_b,
        p4_b,
        p5_b,
        p6p_b,
    ) = b
    return (
        abs(preflop_a - preflop_b) * 8.0
        + abs(flop_a - flop_b) * 5.5
        + abs(turn_a - turn_b) * 6.5
        + abs(river_a - river_b) * 7.5
        + abs(p2_a - p2_b) * 0.8
        + abs(p3_a - p3_b) * 0.8
        + abs(p4_a - p4_b) * 1.0
        + abs(p5_a - p5_b) * 1.0
        + abs(p6p_a - p6p_b) * 1.2
    )


def _clamp_profile_value(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _profiles_for_target_signature(
    bot_profiles: List[BotProfile],
    target_signature: Tuple[float, float, float, float, float, float, float, float, float],
) -> List[BotProfile]:
    (
        target_calls,
        target_checks,
        target_raises,
        target_folds,
        _target_actions,
        target_streets,
        _target_players,
        target_amount,
        _target_pot_after,
    ) = target_signature

    loosen_delta = (
        (target_streets - 0.9) * 0.16
        + (target_calls - 0.7) * 0.08
        - (target_folds - 4.5) * 0.03
    )
    aggression_delta = (
        (target_raises - 0.8) * 0.10
        + (target_amount - 0.35) * 0.10
        - (target_checks - 1.0) * 0.03
    )
    loosen_delta = _clamp_profile_value(loosen_delta, -0.12, 0.12)
    aggression_delta = _clamp_profile_value(aggression_delta, -0.10, 0.12)

    tuned_profiles: List[BotProfile] = []
    for profile in bot_profiles:
        street_early_bias = _clamp_profile_value((1.05 - target_streets) * 1.15, -0.30, 0.30)
        fold_bias = _clamp_profile_value((target_folds - 4.7) * 0.08, -0.24, 0.24)
        call_bias = _clamp_profile_value((target_calls - 0.72) * 0.14, -0.18, 0.18)
        # Earlier target endings should push bots to continue less postflop, not more.
        continue_bias = _clamp_profile_value((-1.0 * street_early_bias) + fold_bias - (0.40 * call_bias), -0.35, 0.28)
        defend_bias = _clamp_profile_value((-0.55 * street_early_bias) + (0.80 * call_bias) - (0.30 * fold_bias), -0.28, 0.28)
        trap_bias = _clamp_profile_value((target_streets - 1.12) * 0.55, -0.22, 0.16)
        tuned_profiles.append(
            BotProfile(
                name=f"{profile.name}_targeted",
                tightness=_clamp_profile_value(profile.tightness - loosen_delta, 0.30, 0.82),
                aggression=_clamp_profile_value(profile.aggression + aggression_delta, 0.25, 0.90),
                bluff_freq=_clamp_profile_value(
                    profile.bluff_freq + (0.20 * aggression_delta) + (0.10 * loosen_delta),
                    0.0,
                    0.18,
                ),
                max_risk_fraction_of_stack=_clamp_profile_value(
                    profile.max_risk_fraction_of_stack + (0.06 * loosen_delta) + (0.04 * aggression_delta),
                    0.10,
                    0.30,
                ),
                tilt_factor=profile.tilt_factor,
                bet_pot_fraction_small=_clamp_profile_value(
                    profile.bet_pot_fraction_small + (0.04 * aggression_delta),
                    0.18,
                    0.55,
                ),
                bet_pot_fraction_medium=_clamp_profile_value(
                    profile.bet_pot_fraction_medium + (0.06 * aggression_delta),
                    0.30,
                    0.85,
                ),
                bet_pot_fraction_large=_clamp_profile_value(
                    profile.bet_pot_fraction_large + (0.08 * aggression_delta),
                    0.50,
                    1.10,
                ),
                preflop_defend_bias=_clamp_profile_value(
                    profile.preflop_defend_bias + defend_bias,
                    -1.0,
                    1.0,
                ),
                postflop_continue_bias=_clamp_profile_value(
                    profile.postflop_continue_bias + continue_bias,
                    -1.0,
                    1.0,
                ),
                trap_frequency=_clamp_profile_value(
                    profile.trap_frequency + trap_bias,
                    -1.0,
                    1.0,
                ),
            )
        )
    return tuned_profiles


def _street_depth_gap(
    a: Tuple[float, float, float, float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float, float, float, float],
) -> float:
    _, _, _, _, _, streets_a, _, _, _ = a
    _, _, _, _, _, streets_b, _, _, _ = b
    return abs(streets_a - streets_b)


def _signature_axis_penalty(
    a: Tuple[float, float, float, float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float, float, float, float],
) -> float:
    calls_a, checks_a, raises_a, folds_a, actions_a, streets_a, players_a, amount_a, pot_a = a
    calls_b, checks_b, raises_b, folds_b, actions_b, streets_b, players_b, amount_b, pot_b = b
    return (
        abs(raises_a - raises_b) * 3.2
        + abs(folds_a - folds_b) * 2.4
        + abs(calls_a - calls_b) * 1.7
        + abs(checks_a - checks_b) * 1.5
        + abs(actions_a - actions_b) * 1.2
        + abs(players_a - players_b) * 1.0
        + abs(amount_a - amount_b) * 0.9
        + abs(pot_a - pot_b) * 0.35
        + abs(streets_a - streets_b) * 3.6
    )


def _chunk_features_for_shortcut_rule(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    if not hands:
        return {
            "chunk_size": 0.0,
            "avg_players": 0.0,
            "avg_actions": 0.0,
            "avg_streets": 0.0,
            "avg_call": 0.0,
            "avg_raise": 0.0,
            "avg_check": 0.0,
            "avg_fold": 0.0,
        }

    total_players = 0
    total_actions = 0
    total_streets = 0
    total_calls = 0
    total_raises = 0
    total_checks = 0
    total_folds = 0
    total_amount = 0.0
    total_pot_after = 0.0

    for hand in hands:
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        streets = hand.get("streets") or []
        total_players += len(players)
        total_actions += len(actions)
        total_streets += len(streets)
        for action in actions:
            action_type = action.get("action_type")
            total_amount += float(action.get("amount", 0.0) or 0.0)
            total_pot_after += float(action.get("pot_after", 0.0) or 0.0)
            if action_type == "call":
                total_calls += 1
            elif action_type == "raise":
                total_raises += 1
            elif action_type == "check":
                total_checks += 1
            elif action_type == "fold":
                total_folds += 1

    n = float(len(hands))
    return {
        "chunk_size": float(len(hands)),
        "avg_players": total_players / n,
        "avg_actions": total_actions / n,
        "avg_streets": total_streets / n,
        "avg_call": total_calls / n,
        "avg_raise": total_raises / n,
        "avg_check": total_checks / n,
        "avg_fold": total_folds / n,
        "avg_amount_sum": total_amount / n,
        "avg_pot_after_sum": total_pot_after / n,
    }


def _best_single_rule_accuracy(
    labeled_chunks: List[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """Estimate leakage via the best one-feature threshold rule at chunk level."""
    rows: List[Tuple[int, Dict[str, float]]] = []
    for chunk in labeled_chunks:
        y = 1 if bool(chunk.get("is_bot", False)) else 0
        rows.append((y, _chunk_features_for_shortcut_rule(chunk.get("hands", []))))

    if not rows:
        return 0.0, {"rule": None}

    feature_names = list(rows[0][1].keys())
    best_acc = 0.0
    best_rule: Dict[str, Any] = {"type": None}
    total = float(len(rows))

    for feature in feature_names:
        uniq = sorted({r[1][feature] for r in rows})
        if not uniq:
            continue
        if len(uniq) > 200:
            step = max(1, len(uniq) // 200)
            uniq = uniq[::step]

        for threshold in uniq:
            for pred_bot_if_gt in (0, 1):
                ok = 0
                for y, feats in rows:
                    pred = pred_bot_if_gt if feats[feature] > threshold else (1 - pred_bot_if_gt)
                    if pred == y:
                        ok += 1
                acc = ok / total
                if acc > best_acc:
                    best_acc = acc
                    best_rule = {
                        "type": "gt",
                        "feature": feature,
                        "threshold": threshold,
                        "pred_bot_if_gt": pred_bot_if_gt,
                    }

    return best_acc, best_rule


def _build_bot_chunks(
    *,
    bot_sizes: List[int],
    bot_profiles: List[BotProfile],
    human_pool: List[Dict[str, Any]],
    human_signatures: List[
        Tuple[float, float, float, float, float, float, float, float, float]
    ],
    human_structures: List[
        Tuple[float, float, float, float, float, float, float, float, float]
    ],
    rng: random.Random,
    candidate_attempts: int,
) -> List[Dict[str, Any]]:
    bot_chunks: List[Dict[str, Any]] = []
    base_candidate_attempts = max(1, int(candidate_attempts))
    for idx, size in enumerate(bot_sizes):
        target_signature = human_signatures[idx % len(human_signatures)]
        target_structure = human_structures[idx % len(human_structures)]
        target_profiles = _profiles_for_target_signature(bot_profiles, target_signature)
        per_chunk_candidates = base_candidate_attempts
        if target_signature[5] >= 0.9 or target_signature[3] <= 4.9:
            per_chunk_candidates += 4
        if target_structure[0] <= 0.58:
            per_chunk_candidates += 2
        best_hands: List[Dict[str, Any]] = []
        best_dist = float("inf")
        for _ in range(per_chunk_candidates):
            candidate_hands = generate_bot_chunk(
                size=size,
                profiles=target_profiles,
                reference_hands=human_pool,
                seed=rng.randint(0, 10**9),
            )
            candidate_sig = _sanitized_chunk_behavior_signature(candidate_hands)
            candidate_structure = _chunk_structure_signature(candidate_hands)
            street_gap = _street_depth_gap(candidate_sig, target_signature)
            dist = (
                _signature_distance(candidate_sig, target_signature)
                + _structure_distance(candidate_structure, target_structure)
                + _signature_axis_penalty(candidate_sig, target_signature)
                + (street_gap * 6.0)
            )
            if dist < best_dist:
                best_dist = dist
                best_hands = candidate_hands
            if street_gap <= 0.10 and dist <= 0.40:
                break

        for hand in best_hands:
            hand["label"] = "bot"
        bot_chunks.append({"hands": best_hands, "is_bot": True})
    return bot_chunks


def _compute_chunk_depth_summary(
    labeled_chunks: List[Dict[str, Any]]
) -> Dict[str, float]:
    human_depths: List[float] = []
    bot_depths: List[float] = []

    for chunk in labeled_chunks:
        signature = _sanitized_chunk_behavior_signature(chunk.get("hands", []))
        street_depth = float(signature[5])
        if chunk.get("is_bot", False):
            bot_depths.append(street_depth)
        else:
            human_depths.append(street_depth)

    if not human_depths or not bot_depths:
        return {
            "human_avg_streets_min": 0.0,
            "human_avg_streets_max": 0.0,
            "bot_avg_streets_min": 0.0,
            "bot_avg_streets_max": 0.0,
            "avg_streets_gap": 0.0,
        }

    return {
        "human_avg_streets_min": min(human_depths),
        "human_avg_streets_max": max(human_depths),
        "bot_avg_streets_min": min(bot_depths),
        "bot_avg_streets_max": max(bot_depths),
        "avg_streets_gap": abs(sum(bot_depths) / len(bot_depths) - (sum(human_depths) / len(human_depths))),
    }


def build_mixed_labeled_chunks(
    cfg: MixedDatasetConfig, *, window_id: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    resolved_window_id = (
        _current_window_id(cfg.refresh_seconds) if window_id is None else int(window_id)
    )
    window_start_iso = None
    window_end_iso = None
    effective_seed = _effective_seed(cfg.seed, resolved_window_id)
    rng = random.Random(effective_seed)

    if cfg.chunk_count <= 0:
        raise ValueError("chunk_count must be > 0")

    if cfg.min_hands_per_chunk <= 0 or cfg.max_hands_per_chunk < cfg.min_hands_per_chunk:
        raise ValueError("Invalid hands per chunk range")

    if not cfg.human_json_path.exists():
        raise FileNotFoundError(f"Missing human JSON source: {cfg.human_json_path}")

    n_human = int(round(cfg.chunk_count * cfg.human_ratio))
    n_human = max(1, min(cfg.chunk_count - 1, n_human))
    n_bot = cfg.chunk_count - n_human

    human_sizes, bot_sizes = _paired_chunk_sizes(
        rng,
        n_human,
        n_bot,
        cfg.min_hands_per_chunk,
        cfg.max_hands_per_chunk,
    )

    needed_human_hands = sum(human_sizes)
    if cfg.human_json_path.suffix != ".gz":
        human_pool = _deterministic_human_selection(
            cfg.human_json_path, needed_human_hands, cfg, resolved_window_id
        )
    else:
        human_pool = _reservoir_sample_humans(cfg.human_json_path, needed_human_hands, rng)
        if not human_pool:
            raise RuntimeError("Could not sample any valid human hands from source JSON")
        if len(human_pool) < needed_human_hands:
            human_pool.extend(rng.choices(human_pool, k=needed_human_hands - len(human_pool)))

    human_chunks: List[Dict[str, Any]] = []
    cursor = 0
    for size in human_sizes:
        human_chunks.append({"hands": human_pool[cursor : cursor + size], "is_bot": False})
        cursor += size
    human_signatures = [
        _sanitized_chunk_behavior_signature(chunk["hands"]) for chunk in human_chunks
    ]
    human_structures = [_chunk_structure_signature(chunk["hands"]) for chunk in human_chunks]

    bot_profiles: List[BotProfile] = _default_bot_profiles()
    rounds = max(1, int(cfg.max_bot_generation_rounds))
    best_labeled_chunks: List[Dict[str, Any]] = []
    best_shortcut_acc = math.inf
    best_shortcut_rule: Dict[str, Any] = {"type": None}
    selected_round = 1

    for round_idx in range(1, rounds + 1):
        bot_chunks = _build_bot_chunks(
            bot_sizes=bot_sizes,
            bot_profiles=bot_profiles,
            human_pool=human_pool,
            human_signatures=human_signatures,
            human_structures=human_structures,
            rng=rng,
            candidate_attempts=cfg.bot_candidate_attempts_per_chunk,
        )
        candidate_chunks = human_chunks + bot_chunks
        rng.shuffle(candidate_chunks)
        sanitized_candidate_chunks = [
            {
                "hands": [sanitize_hand_for_miner(hand) for hand in chunk.get("hands", [])],
                "is_bot": bool(chunk.get("is_bot", False)),
            }
            for chunk in candidate_chunks
        ]
        shortcut_acc, shortcut_rule = _best_single_rule_accuracy(sanitized_candidate_chunks)
        if shortcut_acc < best_shortcut_acc:
            best_shortcut_acc = shortcut_acc
            best_shortcut_rule = shortcut_rule
            best_labeled_chunks = candidate_chunks
            selected_round = round_idx
        if shortcut_acc <= cfg.max_shortcut_rule_accuracy:
            break

    labeled_chunks = best_labeled_chunks

    dataset_hash = _compute_dataset_hash(labeled_chunks)

    stats = {
        "chunk_count": len(labeled_chunks),
        "human_chunks": n_human,
        "bot_chunks": n_bot,
        "total_hands": sum(len(c["hands"]) for c in labeled_chunks),
        "human_hands": sum(len(c["hands"]) for c in labeled_chunks if not c["is_bot"]),
        "bot_hands": sum(len(c["hands"]) for c in labeled_chunks if c["is_bot"]),
        "dataset_hash": dataset_hash,
        "generated_at": int(time.time()),
        "window_id": resolved_window_id,
        "effective_seed": effective_seed,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
        "shortcut_rule_accuracy": best_shortcut_acc,
        "shortcut_rule": best_shortcut_rule,
        "bot_generation_rounds": rounds,
        "selected_bot_generation_round": selected_round,
    }
    stats.update(_compute_chunk_depth_summary(labeled_chunks))
    return labeled_chunks, dataset_hash, stats


def save_mixed_dataset(output_path: Path, labeled_chunks: List[Dict[str, Any]], dataset_hash: str, stats: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "stats": stats,
        "dataset_hash": dataset_hash,
        "labeled_chunks": labeled_chunks,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def load_mixed_dataset(path: Path) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        labeled_chunks = payload
        dataset_hash = _compute_dataset_hash(labeled_chunks)
        stats = {
            "chunk_count": len(labeled_chunks),
            "dataset_hash": dataset_hash,
            "generated_at": int(path.stat().st_mtime),
        }
        return labeled_chunks, dataset_hash, stats

    labeled_chunks = payload.get("labeled_chunks", [])
    dataset_hash = payload.get("dataset_hash") or _compute_dataset_hash(labeled_chunks)
    stats = payload.get("stats") or {
        "chunk_count": len(labeled_chunks),
        "dataset_hash": dataset_hash,
        "generated_at": int(path.stat().st_mtime),
    }
    return labeled_chunks, dataset_hash, stats


class TimedMixedDatasetProvider:
    """Serves a stable mixed human/bot dataset and refreshes it every N seconds."""

    def __init__(self, cfg: MixedDatasetConfig):
        self.cfg = cfg
        self._data: List[Dict[str, Any]] = []
        self._dataset_hash: str = ""
        self._stats: Dict[str, Any] = {}
        self._last_refresh_ts: float = 0.0
        self._window_id: Optional[int] = None

        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        current_window_id = _current_window_id(self.cfg.refresh_seconds)
        if self.cfg.output_path.exists():
            try:
                data, ds_hash, stats = load_mixed_dataset(self.cfg.output_path)
                stored_window_id = stats.get("window_id")
                if data and stored_window_id == current_window_id:
                    self._data = data
                    self._dataset_hash = ds_hash
                    self._stats = stats
                    self._window_id = int(stored_window_id)
                    self._last_refresh_ts = float(stats.get("generated_at", int(time.time())))
                    bt.logging.info(
                        f"Loaded mixed dataset from disk: {self.cfg.output_path} | chunks={len(self._data)} hash={self._dataset_hash[:12]}"
                    )
                    return
            except Exception as e:
                bt.logging.warning(f"Failed to load mixed dataset from disk, regenerating: {e}")

        self.force_refresh(window_id=current_window_id)

    def force_refresh(self, *, window_id: Optional[int] = None) -> None:
        resolved_window_id = (
            _current_window_id(self.cfg.refresh_seconds) if window_id is None else int(window_id)
        )
        labeled_chunks, ds_hash, stats = build_mixed_labeled_chunks(
            self.cfg, window_id=resolved_window_id
        )
        save_mixed_dataset(self.cfg.output_path, labeled_chunks, ds_hash, stats)
        self._data = labeled_chunks
        self._dataset_hash = ds_hash
        self._stats = stats
        self._window_id = int(stats["window_id"])
        self._last_refresh_ts = time.time()
        bt.logging.info(
            f"Generated mixed dataset | chunks={len(self._data)} hash={self._dataset_hash[:12]} saved={self.cfg.output_path}"
        )

    def refresh_if_due(self) -> None:
        current_window_id = _current_window_id(self.cfg.refresh_seconds)
        if self._window_id == current_window_id:
            return

        bt.logging.info("Mixed dataset refresh window reached. Regenerating candidate dataset...")
        labeled_chunks, new_hash, new_stats = build_mixed_labeled_chunks(
            self.cfg, window_id=current_window_id
        )

        if new_hash != self._dataset_hash:
            bt.logging.info(
                f"New mixed dataset differs from current one ({self._dataset_hash[:12]} -> {new_hash[:12]}). Replacing."
            )
            save_mixed_dataset(self.cfg.output_path, labeled_chunks, new_hash, new_stats)
            self._data = labeled_chunks
            self._dataset_hash = new_hash
            self._stats = new_stats
            self._window_id = int(new_stats["window_id"])
        else:
            bt.logging.info("Regenerated dataset is identical. Keeping current dataset.")

        self._window_id = current_window_id
        self._last_refresh_ts = time.time()

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def fetch_hand_batch(
        self,
        *,
        limit: int = 80,
        include_integrity: bool = True,
    ) -> List[LabeledHandBatch]:
        if not self._data:
            return []

        selected = self._data[: max(0, limit)]
        batches: List[LabeledHandBatch] = []
        for entry in selected:
            hands_raw = entry.get("hands", [])
            is_bot = bool(entry.get("is_bot", False))
            hands = [from_standard_json(hand) for hand in hands_raw]
            batches.append(LabeledHandBatch(hands=hands, is_human=not is_bot))
        return batches
