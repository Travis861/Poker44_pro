"""
Quick consistency checker for Poker44 hand JSON files.

Validates that hands under hands_generator/bot_hands/bot_hands.json and
hands_generator/human_hands/human_hands.json match the schema in poker44/core/hand_json.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path for Poker44 imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.core.hand_json import V0_JSON_HAND


# --------- Validation helpers ---------

def _expected_top_keys() -> set:
    return set(V0_JSON_HAND.keys())


def _expected_metadata_keys() -> set:
    return set(V0_JSON_HAND["metadata"].keys())


def _expected_outcome_keys() -> set:
    return set(V0_JSON_HAND["outcome"].keys())


def _expected_player_keys() -> set:
    return set(V0_JSON_HAND["players"][0].keys())


def _expected_street_keys() -> set:
    return set(V0_JSON_HAND["streets"][0].keys())


def _expected_action_keys() -> set:
    return set(V0_JSON_HAND["actions"][0].keys())


def _hand_ended_from_board(streets: List[Dict[str, Any]]) -> str:
    if not streets:
        return "preflop"
    last_board = streets[-1]["board_cards"]
    if len(last_board) == 5:
        return "river"
    if len(last_board) == 4:
        return "turn"
    if len(last_board) == 3:
        return "flop"
    return "preflop"


def validate_hand(hand: Dict[str, Any], idx: int, source: str) -> List[str]:
    errors: List[str] = []

    top_keys = set(hand.keys())
    if top_keys != _expected_top_keys():
        errors.append(f"[{source} #{idx}] top-level keys mismatch. expected={sorted(_expected_top_keys())} got={sorted(top_keys)}")

    meta = hand.get("metadata", {})
    if set(meta.keys()) != _expected_metadata_keys():
        errors.append(f"[{source} #{idx}] metadata keys mismatch. expected={sorted(_expected_metadata_keys())} got={sorted(meta.keys())}")

    outcome = hand.get("outcome", {})
    if set(outcome.keys()) != _expected_outcome_keys():
        errors.append(f"[{source} #{idx}] outcome keys mismatch. expected={sorted(_expected_outcome_keys())} got={sorted(outcome.keys())}")
    else:
        # Ensure payouts + rake ≈ total_pot when provided
        total_pot = outcome.get("total_pot")
        rake = outcome.get("rake", 0)
        if total_pot is not None:
            payouts_sum = sum(outcome.get("payouts", {}).values())
            if round(payouts_sum + rake, 2) != round(total_pot, 2):
                errors.append(
                    f"[{source} #{idx}] total_pot mismatch. payouts+rake={round(payouts_sum+rake,2)} total_pot={total_pot}"
                )

    players = hand.get("players", [])
    for p_idx, player in enumerate(players):
        if set(player.keys()) != _expected_player_keys():
            errors.append(f"[{source} #{idx}] player[{p_idx}] keys mismatch. expected={sorted(_expected_player_keys())} got={sorted(player.keys())}")
        # Hole cards, if present, should be a 2-card list
        if player.get("hole_cards") is not None:
            hc = player["hole_cards"]
            if not isinstance(hc, list) or len(hc) != 2:
                errors.append(f"[{source} #{idx}] player[{p_idx}] hole_cards malformed: {hc}")

    streets = hand.get("streets", [])
    for s_idx, street in enumerate(streets):
        if set(street.keys()) != _expected_street_keys():
            errors.append(f"[{source} #{idx}] street[{s_idx}] keys mismatch. expected={sorted(_expected_street_keys())} got={sorted(street.keys())}")
        board_cards = street.get("board_cards", [])
        # Basic board sanity: flop=3, turn=4, river=5 cards
        if street.get("street") == "flop" and len(board_cards) != 3:
            errors.append(f"[{source} #{idx}] street[{s_idx}] flop must have 3 cards, got {len(board_cards)}")
        if street.get("street") == "turn" and len(board_cards) != 4:
            errors.append(f"[{source} #{idx}] street[{s_idx}] turn must have 4 cards, got {len(board_cards)}")
        if street.get("street") == "river" and len(board_cards) != 5:
            errors.append(f"[{source} #{idx}] street[{s_idx}] river must have 5 cards, got {len(board_cards)}")

    actions = hand.get("actions", [])
    for a_idx, action in enumerate(actions):
        if set(action.keys()) != _expected_action_keys():
            errors.append(f"[{source} #{idx}] action[{a_idx}] keys mismatch. expected={sorted(_expected_action_keys())} got={sorted(action.keys())}")

    # Metadata consistency: hand_ended_on_street should match board length
    expected_end = _hand_ended_from_board(streets)
    if meta.get("hand_ended_on_street") != expected_end:
        errors.append(
            f"[{source} #{idx}] metadata.hand_ended_on_street mismatch. expected={expected_end} got={meta.get('hand_ended_on_street')}"
        )

    return errors


# --------- Runner ---------

def load_hands(json_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not json_path.exists():
        return [], [f"Missing file: {json_path}"]
    try:
        data = json.loads(json_path.read_text())
        if not isinstance(data, list):
            return [], [f"{json_path}: top-level JSON is not a list"]
        return data, []
    except Exception as exc:  # pragma: no cover - quick CLI script
        return [], [f"Failed to parse {json_path}: {exc}"]


def check_file(label: str, json_path: Path) -> List[str]:
    hands, load_errors = load_hands(json_path)
    errors: List[str] = []
    errors.extend(load_errors)
    if load_errors:
        return errors

    for idx, hand in enumerate(hands):
        errors.extend(validate_hand(hand, idx, f"{label}:{json_path.name}"))
    return errors


def main() -> int:
    base = Path(__file__).parent
    paths = {
        "bot": base / "bot_hands" / "bot_hands.json",
        "human": base / "human_hands" / "human_hands.json",
    }

    all_errors: List[str] = []
    for label, path in paths.items():
        errs = check_file(label, path)
        if errs:
            all_errors.extend(errs)

    if all_errors:
        print("\n✗ Inconsistencies found:")
        for err in all_errors:
            print(" -", err)
        print(f"\nTotal errors: {len(all_errors)}")
        return 1

    print("✓ No format inconsistencies detected for bot and human hand files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
