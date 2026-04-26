"""
Parser that converts the PokerStars text hand history in `data.txt` into the
Poker44 canonical hand JSON structure defined in `poker44/core/hand_json.py`.

Usage:
    python hands_generator/parse_to_poker44.py
"""

from __future__ import annotations

import json
import re
import hashlib
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from poker44.core.hand_json import V0_JSON_HAND


# Change this salt to any long random string and keep it secret!
# It ensures the same input username always produces the same anonymized ID
SALT = "poker_anonymizer_2025_secret_salt_change_me"


def parse_metadata(header: str, table: str) -> Dict[str, Any]:
    header_re = re.compile(
        r"PokerStars Zoom Hand #(?P<hand_id>\d+):\s+Hold'em No Limit "
        r"\(€(?P<sb>[\d.]+)/€(?P<bb>[\d.]+)\)\s+-\s+"
        r"(?P<date>\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2})\s+(?P<tz>\w+)"
    )
    table_re = re.compile(
        r"Table '([^']+)' (?P<max>\d+)-max Seat #(?P<button>\d+) is the button"
    )

    match = header_re.search(header)
    if not match:
        raise ValueError(f"Could not parse header line: {header}")

    sb = float(match.group("sb"))
    bb = float(match.group("bb"))
    table_match = table_re.search(table)
    max_seats = int(table_match.group("max")) if table_match else 0
    button_seat = int(table_match.group("button")) if table_match else 0

    return {
        "game_type": "Hold'em",
        "limit_type": "No Limit",
        "max_seats": max_seats,
        "hero_seat": None,
        "hand_ended_on_street": None,
        "button_seat": button_seat,
        "sb": sb,
        "bb": bb,
        "ante": 0.0,
        "rng_seed_commitment": None,
    }


def parse_players(lines: Iterable[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    players: List[Dict[str, Any]] = []
    seat_lookup: Dict[str, int] = {}
    seat_re = re.compile(r"Seat (\d+): ([^(]+) \(€([\d.]+) in chips\)")

    for line in lines:
        match = seat_re.match(line.strip())
        if not match:
            break
        seat = int(match.group(1))
        name = match.group(2).strip()
        stack = float(match.group(3))
        players.append(
            {
                "player_uid": name,
                "seat": seat,
                "starting_stack": stack,
                "hole_cards": None,
                "showed_hand": False,
            }
        )
        seat_lookup[name] = seat
    return players, seat_lookup


def normalize(board_cards: List[str]) -> List[str]:
    return [card.strip() for card in board_cards if card.strip()]


def parse_board(line: str) -> List[str]:
    match = re.search(r"\[([^\]]+)\]", line)
    if not match:
        return []
    return normalize(match.group(1).split())


def parse_action_line(
    line: str,
    street: str,
    seat_lookup: Dict[str, int],
    bb: float,
    pot: float,
    current_bet: float,
    action_id: int,
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    line = line.strip()
    amount = 0.0
    actor = ""
    action_type = None

    raise_to = None
    call_to = None

    if "posts small blind" in line:
        match = re.match(r"([^:]+): posts small blind €([\d.]+)", line)
        if match:
            actor, amount = match.group(1).strip(), float(match.group(2))
            action_type = "small_blind"
    elif "posts big blind" in line:
        match = re.match(r"([^:]+): posts big blind €([\d.]+)", line)
        if match:
            actor, amount = match.group(1).strip(), float(match.group(2))
            action_type = "big_blind"
    elif ": calls" in line:
        match = re.match(r"([^:]+): calls €([\d.]+)", line)
        if match:
            actor, amount = match.group(1).strip(), float(match.group(2))
            action_type = "call"
    elif ": raises" in line:
        match = re.match(r"([^:]+): raises €([\d.]+) to €([\d.]+)", line)
        if match:
            actor, amount = match.group(1).strip(), float(match.group(2))
            raise_to = float(match.group(3))
            action_type = "raise"
    elif ": bets" in line:
        match = re.match(r"([^:]+): bets €([\d.]+)", line)
        if match:
            actor, amount = match.group(1).strip(), float(match.group(2))
            action_type = "bet"
    elif ": checks" in line:
        actor = line.split(":")[0].strip()
        action_type = "check"
    elif ": folds" in line:
        actor = line.split(":")[0].strip()
        action_type = "fold"

    if not action_type:
        return None, pot, current_bet

    pot_before = pot
    if action_type not in {"fold", "check"}:
        pot += amount

    if action_type == "big_blind":
        current_bet = max(current_bet, amount)
    elif action_type == "bet":
        current_bet = amount
    elif action_type == "raise":
        current_bet = raise_to if raise_to is not None else current_bet
    elif action_type == "call":
        call_to = current_bet

    action = {
        "action_id": str(action_id),
        "street": street,
        "actor_seat": seat_lookup.get(actor, 0),
        "action_type": action_type,
        "amount": round(amount, 2),
        "raise_to": round(raise_to, 2) if raise_to is not None else None,
        "call_to": round(call_to, 2) if call_to is not None else None,
        "normalized_amount_bb": round(amount / bb, 4) if bb else 0.0,
        "pot_before": round(pot_before, 2),
        "pot_after": round(pot, 2),
    }
    return action, pot, current_bet


def parse_summary(
    lines: List[str],
) -> Tuple[List[str], Dict[str, float], float, float, List[str], Dict[str, Optional[bool]]]:
    winners: List[str] = []
    payouts: Dict[str, float] = {}
    rake = 0.0
    total_pot = 0.0
    board_cards: List[str] = []
    show_info: Dict[str, Optional[bool]] = {}

    pot_re = re.compile(r"Total pot €([\d.]+)(?: \| Rake €([\d.]+))?")
    winner_re = re.compile(r"(?:Seat \d+: |^)(.+?)(?: \(button\)|\(small blind\)|\(big blind\)|\(dealer\))?\s*(?:showed .+? and )?(?:collected|won)(?: from pot)?\s*\(?€([\d.]+)\)?")

    for line in lines:
        line = line.strip()
        pot_match = pot_re.match(line)
        if pot_match:
            total_pot = float(pot_match.group(1))
            if pot_match.group(2):
                rake = float(pot_match.group(2))
            continue

        if line.startswith("Board "):
            board_cards = parse_board(line)
            continue

        show_match = re.match(r"Seat \d+: ([^(]+).*showed", line)
        if show_match:
            name = show_match.group(1).strip()
            show_info[name] = True

        winner_match = winner_re.match(line)
        if winner_match:
            name = winner_match.group(1).strip()
            amount = float(winner_match.group(2))
            winners.append(name)
            payouts[name] = amount

    return winners, payouts, rake, total_pot, board_cards, show_info


def build_streets(board_cards: List[str]) -> List[Dict[str, Any]]:
    streets: List[Dict[str, Any]] = []
    if len(board_cards) >= 3:
        streets.append({"street": "flop", "board_cards": board_cards[:3]})
    if len(board_cards) >= 4:
        streets.append({"street": "turn", "board_cards": board_cards[:4]})
    if len(board_cards) == 5:
        streets.append({"street": "river", "board_cards": board_cards})
    return streets


def parse_hand(raw_hand: str) -> Optional[Dict[str, Any]]:
    cleaned = raw_hand.replace("\ufeff", "").strip()
    if not cleaned:
        return None

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    metadata = parse_metadata(lines[0], lines[1])

    # Players live between the table line and the blind postings.
    player_lines: List[str] = []
    action_start_idx = 2
    for idx in range(2, len(lines)):
        if lines[idx].startswith("Seat "):
            player_lines.append(lines[idx])
            action_start_idx = idx + 1
        else:
            action_start_idx = idx
            break

    players, seat_lookup = parse_players(player_lines)

    actions: List[Dict[str, Any]] = []
    board_cards: List[str] = []
    hole_cards: Dict[str, List[str]] = {}
    showed_hand: Dict[str, bool] = {}
    hero_name: Optional[str] = None
    showdown = False
    street = "preflop"
    pot = 0.0
    action_id = 1
    current_bet = 0.0

    idx = action_start_idx
    while idx < len(lines):
        line = lines[idx]
        idx += 1

        if line.startswith("*** HOLE CARDS ***"):
            street = "preflop"
            continue
        if line.startswith("*** FLOP ***"):
            board_cards = parse_board(line)
            street = "flop"
            current_bet = 0.0
            continue
        if line.startswith("*** TURN ***"):
            turn_board = parse_board(line)
            board_cards = turn_board or board_cards
            street = "turn"
            current_bet = 0.0
            continue
        if line.startswith("*** RIVER ***"):
            river_board = parse_board(line)
            board_cards = river_board or board_cards
            street = "river"
            current_bet = 0.0
            continue
        if line.startswith("*** SHOW DOWN ***"):
            street = "showdown"
            showdown = True
            continue
        if line.startswith("*** SUMMARY ***"):
            break

        # Hero cards
        if m := re.match(r"Dealt to ([^[]+) \[([^\]]+)\]", line):
            hero_name = m.group(1).strip()
            hole_cards[hero_name] = normalize(m.group(2).split())
            continue

        # Showdown shows
        if m := re.match(r"([^:]+): shows \[([^\]]+)\]", line):
            name = m.group(1).strip()
            hole_cards[name] = normalize(m.group(2).split())
            showed_hand[name] = True
            showdown = True
            continue

        if m := re.match(r"([^:]+): doesn't show hand", line):
            name = m.group(1).strip()
            showed_hand[name] = False
            continue

        # Uncalled bet return
        if m := re.match(r"Uncalled bet \(€([\d.]+)\) returned to ([^\)]+)", line):
            amount = float(m.group(1))
            player = m.group(2).strip()
            pot_before = pot
            pot -= amount
            actions.append({
                "action_id": str(action_id),
                "street": street,
                "actor_seat": seat_lookup.get(player, 0),
                "action_type": "uncalled_bet_return",
                "amount": round(amount, 2),
                "raise_to": None,
                "call_to": None,
                "normalized_amount_bb": round(amount / metadata["bb"], 4),
                "pot_before": round(pot_before, 2),
                "pot_after": round(pot, 2),
            })
            action_id += 1
            continue

        action, pot, current_bet = parse_action_line(
            line, street, seat_lookup, metadata["bb"], pot, current_bet, action_id
        )
        if action:
            actions.append(action)
            action_id += 1

    summary_lines = lines[idx:] if idx < len(lines) else []
    winners, payouts, rake, total_pot, summary_board, summary_show_info = parse_summary(summary_lines)
    if summary_board:
        board_cards = summary_board
    for name, info in summary_show_info.items():
        if name not in showed_hand and info is not None:
            showed_hand[name] = bool(info)
        if info is True:
            showdown = True

    streets = build_streets(board_cards)

    for player in players:
        player_name = player["player_uid"]
        player["hole_cards"] = hole_cards.get(player_name) or None
        if player["hole_cards"] is not None:
            player["showed_hand"] = True
        else:
            player["showed_hand"] = bool(showed_hand.get(player_name, False))

    outcome = {
        "winners": winners,
        "payouts": payouts,
        "total_pot": total_pot,
        "rake": rake,
        "result_reason": "showdown" if showdown else "fold",
        "showdown": showdown,
    }

    metadata["hero_seat"] = seat_lookup.get(hero_name, None) if hero_name else None
    if len(board_cards) == 5:
        metadata["hand_ended_on_street"] = "river"
    elif len(board_cards) == 4:
        metadata["hand_ended_on_street"] = "turn"
    elif len(board_cards) == 3:
        metadata["hand_ended_on_street"] = "flop"
    else:
        metadata["hand_ended_on_street"] = "preflop"

    hand = copy.deepcopy(V0_JSON_HAND)
    hand["metadata"] = metadata
    hand["players"] = players
    hand["streets"] = streets
    hand["actions"] = actions
    hand["outcome"] = outcome
    hand["label"] = "human"
    assert_hand_format(hand)
    return hand


def assert_hand_format(hand: Dict[str, Any]) -> None:
    expected_top_keys = set(V0_JSON_HAND.keys())
    actual_top_keys = set(hand.keys())
    if actual_top_keys != expected_top_keys:
        raise AssertionError(
            f"Hand keys mismatch. expected={sorted(expected_top_keys)} actual={sorted(actual_top_keys)}"
        )

    expected_metadata_keys = set(V0_JSON_HAND["metadata"].keys())
    if set(hand["metadata"].keys()) != expected_metadata_keys:
        raise AssertionError(
            f"Metadata keys mismatch. expected={sorted(expected_metadata_keys)} actual={sorted(hand['metadata'].keys())}"
        )

    expected_outcome_keys = set(V0_JSON_HAND["outcome"].keys())
    if set(hand["outcome"].keys()) != expected_outcome_keys:
        raise AssertionError(
            f"Outcome keys mismatch. expected={sorted(expected_outcome_keys)} actual={sorted(hand['outcome'].keys())}"
        )

    if hand["players"]:
        expected_player_keys = set(V0_JSON_HAND["players"][0].keys())
        for idx, player in enumerate(hand["players"]):
            if set(player.keys()) != expected_player_keys:
                raise AssertionError(
                    f"Player keys mismatch at index {idx}. expected={sorted(expected_player_keys)} actual={sorted(player.keys())}"
                )

    if hand["streets"]:
        expected_street_keys = set(V0_JSON_HAND["streets"][0].keys())
        for idx, street in enumerate(hand["streets"]):
            if set(street.keys()) != expected_street_keys:
                raise AssertionError(
                    f"Street keys mismatch at index {idx}. expected={sorted(expected_street_keys)} actual={sorted(street.keys())}"
                )

    if hand["actions"]:
        expected_action_keys = set(V0_JSON_HAND["actions"][0].keys())
        for idx, action in enumerate(hand["actions"]):
            if set(action.keys()) != expected_action_keys:
                raise AssertionError(
                    f"Action keys mismatch at index {idx}. expected={sorted(expected_action_keys)} actual={sorted(action.keys())}"
                )


def build_global_player_mapping(hands: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for hand in hands:
        for player in hand["players"]:
            uid = player["player_uid"]
            if uid not in mapping:
                digest = hashlib.sha256((SALT + uid).encode("utf-8")).hexdigest()
                mapping[uid] = f"p_{digest}"
    return mapping


def anonymize_all_hands(hands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hands:
        return hands

    mapping = build_global_player_mapping(hands)

    for hand in hands:
        # Players
        for player in hand["players"]:
            player["player_uid"] = mapping.get(player["player_uid"], player["player_uid"])

        # Outcome
        outcome = hand["outcome"]
        if outcome["winners"]:
            outcome["winners"] = [mapping.get(w, w) for w in outcome["winners"]]
        if outcome["payouts"]:
            outcome["payouts"] = {mapping.get(k, k): v for k, v in outcome["payouts"].items()}

    return hands


def split_hands(content: str) -> List[str]:
    content = content.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n(?=PokerStars Zoom Hand #)", content.strip())
    return [part for part in parts if part.strip()]


def parse_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    hands_raw = split_hands(text)
    parsed: List[Dict[str, Any]] = []
    for raw in hands_raw:
        hand = parse_hand(raw)
        if hand:
            parsed.append(hand)
    return parsed


def main() -> None:
    input_path = Path(__file__).with_name("data.txt")
    output_path = Path(__file__).with_name("human_hands.json")

    hands = parse_file(input_path)
    hands = anonymize_all_hands(hands)

    output_path.write_text(json.dumps(hands, indent=2), encoding="utf-8")
    print(f"Parsed {len(hands)} hands into {output_path}")


if __name__ == "__main__":
    main()
