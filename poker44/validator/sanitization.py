"""Utilities for producing miner-visible Poker44 payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_LEAKAGE_KEYS = {
    "label",
    "label_flag",
    "is_bot",
    "bot_family_id",
    "bot_version",
}
_MINER_ACTION_WINDOW = 12
_DEFAULT_MAX_SEATS = 6
_SANITIZED_SB = 0.01
_SANITIZED_BB = 0.02
_SANITIZED_ANTE = 0.0
_MAX_NORMALIZED_STACK_BB = 500.0
_MAX_NORMALIZED_ACTION_BB = 200.0
_MAX_NORMALIZED_POT_BB = 1000.0
_ALLOWED_ACTION_TYPES = {
    "small_blind",
    "big_blind",
    "ante",
    "check",
    "call",
    "bet",
    "raise",
    "fold",
    "all_in",
}


def _round_bounded(value: float, *, lower: float = 0.0, upper: float) -> float:
    return round(max(lower, min(upper, float(value))), 2)


def _to_bb_units(value: Any, bb: float, *, upper: float) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if bb <= 0:
        return 0.0
    return _round_bounded(numeric / bb, upper=upper)


def _from_bb_units(bb_value: float, *, sanitized_bb: float = _SANITIZED_BB) -> float:
    return round(max(0.0, float(bb_value)) * sanitized_bb, 4)


def _sanitize_seat(value: Any, *, max_seats: int) -> int:
    try:
        seat = int(value)
    except (TypeError, ValueError):
        return 0
    return seat if 1 <= seat <= max_seats else 0


def _sanitize_action_type(value: Any) -> str:
    action_type = str(value or "").strip().lower()
    if action_type in _ALLOWED_ACTION_TYPES:
        return action_type
    if "raise" in action_type:
        return "raise"
    if "bet" in action_type:
        return "bet"
    if "call" in action_type:
        return "call"
    if "check" in action_type:
        return "check"
    if "fold" in action_type or action_type == "muck":
        return "fold"
    return "other"


def strip_leakage_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _LEAKAGE_KEYS:
                continue
            cleaned[key] = strip_leakage_fields(item)
        return cleaned
    if isinstance(value, list):
        return [strip_leakage_fields(item) for item in value]
    return value


def sanitize_hand_for_miner(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep behaviorally useful structure while suppressing direct identity leakage."""
    cleaned = strip_leakage_fields(hand_payload)
    if not isinstance(cleaned, dict):
        return {}

    metadata = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
    players_raw = cleaned.get("players") if isinstance(cleaned.get("players"), list) else []
    actions_raw = cleaned.get("actions") if isinstance(cleaned.get("actions"), list) else []
    outcome = cleaned.get("outcome") if isinstance(cleaned.get("outcome"), dict) else {}

    max_seats = max(
        _DEFAULT_MAX_SEATS,
        _sanitize_seat(metadata.get("max_seats"), max_seats=10),
    )
    source_bb = float(metadata.get("bb", 0.0) or 0.0)

    seat_to_stack_bb: Dict[int, float] = {}
    for player in players_raw:
        if not isinstance(player, dict):
            continue
        seat_i = _sanitize_seat(player.get("seat"), max_seats=max_seats)
        if seat_i == 0:
            continue
        seat_to_stack_bb[seat_i] = _to_bb_units(
            player.get("starting_stack", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_STACK_BB,
        )

    sanitized_players: List[Dict[str, Any]] = [
        {
            "player_uid": f"seat_{seat_i}",
            "seat": seat_i,
            "starting_stack": _from_bb_units(starting_stack_bb),
            "hole_cards": None,
            "showed_hand": False,
        }
        for seat_i, starting_stack_bb in sorted(seat_to_stack_bb.items())
    ]

    raw_actions: List[Dict[str, Any]] = []
    for action in actions_raw:
        if not isinstance(action, dict):
            continue
        amount_bb = _to_bb_units(
            action.get("amount", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_ACTION_BB,
        )
        raise_to_bb = _to_bb_units(
            action.get("raise_to"),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        call_to_bb = _to_bb_units(
            action.get("call_to"),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        pot_before_bb = _to_bb_units(
            action.get("pot_before", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        pot_after_bb = _to_bb_units(
            action.get("pot_after", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        raw_actions.append(
            {
                "action_id": "",
                "street": str(action.get("street", "")),
                "actor_seat": _sanitize_seat(action.get("actor_seat"), max_seats=max_seats),
                "action_type": _sanitize_action_type(action.get("action_type")),
                "amount": _from_bb_units(amount_bb),
                "raise_to": None if raise_to_bb <= 0 else _from_bb_units(raise_to_bb),
                "call_to": None if call_to_bb <= 0 else _from_bb_units(call_to_bb),
                "normalized_amount_bb": amount_bb,
                "pot_before": _from_bb_units(pot_before_bb),
                "pot_after": _from_bb_units(pot_after_bb),
            }
        )

    sanitized_actions: List[Dict[str, Any]] = []
    if raw_actions:
        last_idx = len(raw_actions) - 1
        if len(raw_actions) == 1:
            indices = [0] * _MINER_ACTION_WINDOW
        else:
            indices = [
                int(round(i * last_idx / (_MINER_ACTION_WINDOW - 1)))
                for i in range(_MINER_ACTION_WINDOW)
            ]
        sanitized_actions = [dict(raw_actions[i]) for i in indices]

    for idx, action in enumerate(sanitized_actions, start=1):
        action["action_id"] = str(idx)

    return {
        "metadata": {
            "game_type": str(metadata.get("game_type", "")),
            "limit_type": str(metadata.get("limit_type", "")),
            "max_seats": max_seats,
            "hero_seat": 0,
            "hand_ended_on_street": "",
            "button_seat": 0,
            "sb": _SANITIZED_SB,
            "bb": _SANITIZED_BB,
            "ante": _SANITIZED_ANTE,
            "rng_seed_commitment": None,
        },
        "players": sanitized_players,
        "streets": [
            {
                "street": str(street.get("street", "")),
                "board_cards": [],
            }
            for street in (cleaned.get("streets") or [])
            if isinstance(street, dict)
        ],
        "actions": sanitized_actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": 0.0,
            "rake": 0.0,
            "result_reason": "",
            "showdown": False,
        },
    }


def sanitized_chunk_signature(
    hands: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Coarse miner-visible behavior signature for shortcut analysis and chunk matching."""
    if not hands:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sanitized_hands = [sanitize_hand_for_miner(hand) for hand in hands]
    total_calls = 0
    total_checks = 0
    total_raises = 0
    total_folds = 0
    total_actions = 0
    total_streets = 0
    total_players = 0
    total_action_amount = 0.0
    total_action_pot_after = 0.0
    for hand in sanitized_hands:
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        total_players += len(players)
        total_actions += len(actions)
        for action in actions:
            action_type = action.get("action_type")
            total_action_amount += float(action.get("normalized_amount_bb", 0.0) or 0.0)
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

    n = float(len(sanitized_hands))
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
