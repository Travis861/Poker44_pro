import copy
import gzip
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hands_generator.bot_hands.sandbox_poker_bot import SandboxPokerBot, BotProfile, GameState, LegalActions, Street, ActionType, BotDecision
from poker44.core.hand_json import V0_JSON_HAND

# Fixed base seed for reproducibility across validators.
BOT_RNG_SEED = 424242
HERO_UID = f"p_{hashlib.sha256('hero_player_fixed_2025_secret'.encode()).hexdigest()}"
PUBLIC_HUMAN_HANDS_PATH = Path(__file__).resolve().parents[1] / "human_hands" / "poker_hands_combined.json.gz"
DEFAULT_STAKE_DISTRIBUTION: List[Tuple[float, float, int]] = [
    (0.02, 0.05, 4660),
    (0.05, 0.10, 337),
    (0.01, 0.02, 3),
]
DEFAULT_PLAYER_COUNT_DISTRIBUTION: List[Tuple[int, int]] = [
    (6, 4203),
    (5, 601),
    (4, 191),
    (3, 2),
    (2, 2),
]
DEFAULT_STACK_BB_RANGE: Tuple[float, float] = (60.0, 160.0)


def _weighted_choice(weighted_values, rng: random.Random):
    values = [value for value, _weight in weighted_values]
    weights = [weight for _value, weight in weighted_values]
    return rng.choices(values, weights=weights, k=1)[0]


def _build_reference_distribution(hands: List[Dict[str, Any]]) -> Dict[str, List]:
    stake_counts: Dict[Tuple[float, float], int] = {}
    player_counts: Dict[int, int] = {}

    for hand in hands:
        metadata = hand.get("metadata") or {}
        sb = round(float(metadata.get("sb", 0.02) or 0.02), 2)
        bb = round(float(metadata.get("bb", 0.05) or 0.05), 2)
        stake_counts[(sb, bb)] = stake_counts.get((sb, bb), 0) + 1

        n_players = len(hand.get("players") or [])
        if n_players >= 2:
            player_counts[n_players] = player_counts.get(n_players, 0) + 1

    stakes = [(sb, bb, count) for (sb, bb), count in stake_counts.items()]
    players = list(player_counts.items())
    return {
        "stakes": stakes or DEFAULT_STAKE_DISTRIBUTION,
        "player_counts": players or DEFAULT_PLAYER_COUNT_DISTRIBUTION,
    }


def _load_reference_distribution() -> Dict[str, List]:
    if not PUBLIC_HUMAN_HANDS_PATH.exists():
        return {
            "stakes": DEFAULT_STAKE_DISTRIBUTION,
            "player_counts": DEFAULT_PLAYER_COUNT_DISTRIBUTION,
        }

    opener = gzip.open if PUBLIC_HUMAN_HANDS_PATH.suffix == ".gz" else open
    try:
        with opener(PUBLIC_HUMAN_HANDS_PATH, "rt", encoding="utf-8") as f:
            hands = json.load(f)
    except Exception:
        return {
            "stakes": DEFAULT_STAKE_DISTRIBUTION,
            "player_counts": DEFAULT_PLAYER_COUNT_DISTRIBUTION,
        }
    return _build_reference_distribution(hands)


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _mutate_profile(profile: BotProfile, rng: random.Random) -> BotProfile:
    """Create a nearby profile variant so bots are not carbon copies."""
    return BotProfile(
        name=f"{profile.name}_v{rng.randint(1, 9999)}",
        tightness=_clamp_float(profile.tightness + rng.uniform(-0.08, 0.08), 0.28, 0.80),
        aggression=_clamp_float(profile.aggression + rng.uniform(-0.10, 0.10), 0.32, 0.94),
        bluff_freq=_clamp_float(profile.bluff_freq + rng.uniform(-0.03, 0.03), 0.0, 0.20),
        max_risk_fraction_of_stack=_clamp_float(
            profile.max_risk_fraction_of_stack + rng.uniform(-0.05, 0.05), 0.10, 0.32
        ),
        tilt_factor=_clamp_float(profile.tilt_factor + rng.uniform(-0.08, 0.08), 0.0, 0.35),
        bet_pot_fraction_small=_clamp_float(
            profile.bet_pot_fraction_small + rng.uniform(-0.07, 0.07), 0.18, 0.58
        ),
        bet_pot_fraction_medium=_clamp_float(
            profile.bet_pot_fraction_medium + rng.uniform(-0.10, 0.10), 0.30, 0.96
        ),
        bet_pot_fraction_large=_clamp_float(
            profile.bet_pot_fraction_large + rng.uniform(-0.12, 0.12), 0.50, 1.28
        ),
        preflop_defend_bias=_clamp_float(
            profile.preflop_defend_bias + rng.uniform(-0.25, 0.25), -1.0, 1.0
        ),
        postflop_continue_bias=_clamp_float(
            profile.postflop_continue_bias + rng.uniform(-0.25, 0.25), -1.0, 1.0
        ),
        trap_frequency=_clamp_float(
            profile.trap_frequency + rng.uniform(-0.20, 0.20), -1.0, 1.0
        ),
    )

@dataclass
class Player:
    uid: str
    seat: int
    stack: float
    is_bot: bool
    bot_instance: Optional[SandboxPokerBot] = None
    folded: bool = False
    invested_this_street: float = 0.0
    total_invested: float = 0.0
    hands_played: int = 0
    hole_cards: Optional[List[str]] = None
    hand_strength: Optional[float] = None

class TableSession:
    def __init__(
        self,
        table_id: str,
        sb: float = 0.02,
        bb: float = 0.05,
        max_seats: int = 6,
        rake_rate: float = 0.05,
        bot_profiles: List[BotProfile] = None,
        target_player_count: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        self.rng = rng or random.Random()
        self.table_id = table_id
        self.sb = sb
        self.bb = bb
        self.max_seats = max_seats
        self.rake_rate = rake_rate
        self.bot_profiles = bot_profiles or []
        self.target_player_count = max(2, min(max_seats, target_player_count or max_seats))
        self.players: List[Optional[Player]] = [None] * max_seats
        self.button_position = 0
        self.hero_seat: Optional[int] = None
        self.hand_number = 0
        self.suits = ['s', 'h', 'd', 'c']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        # Anonymized bot name pool
        self.available_names = []
        for i in range(1, 1001):
            hash_input = f"bot_seed_{i}_{BOT_RNG_SEED}"
            player_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            # Store raw hex (no prefix); prefix added at seat assignment time
            self.available_names.append(player_hash)
        self.rng.shuffle(self.available_names)

    def initialize_table(self):
        # Randomize hero seat to mirror human variance
        self.hero_seat = self.rng.randint(1, self.max_seats)
        hero_profile = _mutate_profile(self.rng.choice(self.bot_profiles), self.rng)
        hero = Player(
            uid=HERO_UID,
            seat=self.hero_seat,
            stack=self._sample_stack(),
            is_bot=True,
            hands_played=0
        )
        hero.bot_instance = SandboxPokerBot(hero_profile, rng_seed=self.rng.randint(0, 10**9))
        self.players[self.hero_seat - 1] = hero
        
        # Fill remaining seats
        num_bots = max(1, self.target_player_count - 1)
        available_seats = [s for s in range(1, self.max_seats + 1) if s != self.hero_seat]
        self.rng.shuffle(available_seats)
        for seat in available_seats[: num_bots]:
            self._add_player_to_seat(seat)
        
        occupied = [i for i, p in enumerate(self.players) if p is not None]
        self.button_position = self.rng.choice(occupied)

    def _sample_stack(self) -> float:
        stack_bb = self.rng.uniform(*DEFAULT_STACK_BB_RANGE)
        return round(stack_bb * self.bb, 2)

    def _add_player_to_seat(self, seat_index: int):
        if seat_index == self.hero_seat or self.players[seat_index - 1] is not None:
            return
        if not self.available_names:
            return
        uid = self.available_names.pop()
        if uid.startswith("p_"):
            uid = uid[2:]
        profile = _mutate_profile(self.rng.choice(self.bot_profiles), self.rng)
        player = Player(
            uid=f"p_{uid}",
            seat=seat_index,
            stack=self._sample_stack(),
            is_bot=True
        )
        player.bot_instance = SandboxPokerBot(profile, rng_seed=self.rng.randint(0, 10**9))
        self.players[seat_index - 1] = player

    def _remove_player(self, seat_index: int):
        if seat_index == self.hero_seat or self.players[seat_index - 1] is None:
            return
        uid = self.players[seat_index - 1].uid
        if uid.startswith("p_"):
            uid = uid[2:]
        self.available_names.append(uid)
        self.players[seat_index - 1] = None

    def rotate_button(self):
        occupied = [i for i, p in enumerate(self.players) if p is not None]
        if not occupied:
            return
        try:
            curr_idx = occupied.index(self.button_position)
        except ValueError:
            curr_idx = 0
        next_idx = (curr_idx + 1) % len(occupied)
        self.button_position = occupied[next_idx]

    def handle_player_changes(self):
        hero_idx = self.hero_seat - 1 if self.hero_seat else 0
        occupied_non_hero = [i for i in range(self.max_seats) if i != hero_idx and self.players[i] is not None]
        empty_seats = [i + 1 for i in range(self.max_seats) if i != hero_idx and self.players[i] is None]
        current_players = len(self.get_active_players())
        desired_players = self.target_player_count
        
        # Leave
        leave_prob = 0.03 if current_players <= desired_players else 0.15
        if len(occupied_non_hero) > 1 and self.rng.random() < leave_prob:
            weights = [3.0 if self.players[i].stack < self.bb * 3 else 1.0 for i in occupied_non_hero]
            leaving_idx = self.rng.choices(occupied_non_hero, weights=weights, k=1)[0]
            self._remove_player(leaving_idx + 1)
        
        # Join
        join_prob = 0.18 if current_players < desired_players else 0.04
        if empty_seats and self.rng.random() < join_prob:
            join_seat = self.rng.choice(empty_seats)
            self._add_player_to_seat(join_seat)

    def get_active_players(self) -> List[Player]:
        return [p for p in self.players if p is not None]

class PokerHandGenerator:
    def __init__(
        self,
        sb=0.02,
        bb=0.05,
        max_seats=6,
        rake_rate=0.05,
        reference_hands: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        self.seed = BOT_RNG_SEED if seed is None else seed
        self.rng = random.Random(self.seed)
        self.sb = sb
        self.bb = bb
        self.max_seats = max_seats
        self.rake_rate = rake_rate
        self.reference_distribution = (
            _build_reference_distribution(reference_hands)
            if reference_hands
            else _load_reference_distribution()
        )
        self.hand_counter = 258890000000
        self.suits = ['s', 'h', 'd', 'c']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    def _sample_table_config(self) -> Tuple[float, float, int]:
        stakes = self.reference_distribution["stakes"]
        player_counts = self.reference_distribution["player_counts"]

        stake_choice = _weighted_choice([((sb, bb), count) for sb, bb, count in stakes], self.rng)
        target_players = _weighted_choice(player_counts, self.rng)
        sb, bb = stake_choice
        return sb, bb, int(target_players)

    def _hand_is_consistent(self, hand: Dict[str, Any]) -> bool:
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        outcome = hand.get("outcome") or {}
        metadata = hand.get("metadata") or {}

        if not players or not actions:
            return False

        seats = sorted(p.get("seat") for p in players if p.get("seat") is not None)
        if seats != list(range(1, len(seats) + 1)):
            return False

        seat_set = set(seats)
        hero_seat = metadata.get("hero_seat")
        button_seat = metadata.get("button_seat")
        if hero_seat not in seat_set or button_seat not in seat_set:
            return False

        try:
            total_pot = round(float(outcome.get("total_pot", 0.0)), 2)
            rake = round(float(outcome.get("rake", 0.0)), 2)
            payout_sum = round(sum(float(v) for v in (outcome.get("payouts") or {}).values()), 2)
        except Exception:
            return False
        if round(payout_sum + rake, 2) != total_pot:
            return False

        last_pot_after = None
        for action in actions:
            actor_seat = action.get("actor_seat")
            if actor_seat not in seat_set:
                return False
            pot_before = round(float(action.get("pot_before", 0.0)), 2)
            pot_after = round(float(action.get("pot_after", 0.0)), 2)
            if pot_before < 0 or pot_after < 0:
                return False
            if last_pot_after is not None and round(last_pot_after, 2) != pot_before:
                return False
            if action.get("action_type") == "uncalled_bet_return":
                if pot_after > pot_before:
                    return False
            elif pot_after < pot_before:
                return False
            last_pot_after = pot_after

        return True

    def generate_hands(
        self,
        num_hands_to_play: int,
        num_hands_to_select: int,
        bot_profiles: List[BotProfile],
        output_file: str = "bot_hands.json",
        hands_per_session: int = 50
    ) -> List[Dict[str, Any]]:
        """
        NEW: Play num_hands_to_play hands, randomly select num_hands_to_select.
        e.g., play 1000, select 100 randomly.
        """
        all_hands = []
        hands_generated = 0
        session_count = 0
        
        print(f"\n=== Generating {num_hands_to_play} hands (will select {num_hands_to_select} randomly) ===")
        
        while hands_generated < num_hands_to_play:
            session_count += 1
            table_id = f"Table_{session_count}"
            sb, bb, target_players = self._sample_table_config()
            min_session_hands = min(20, hands_per_session, num_hands_to_play - hands_generated)
            max_session_hands = min(hands_per_session, num_hands_to_play - hands_generated)
            session = TableSession(
                table_id=table_id,
                sb=sb,
                bb=bb,
                max_seats=self.max_seats,
                rake_rate=self.rake_rate,
                bot_profiles=bot_profiles,
                target_player_count=target_players,
                rng=self.rng,
            )
            session.initialize_table()
            session_length = self.rng.randint(min_session_hands, max_session_hands)
            
            print(
                f"Session {session_count}: {table_id} ({session_length} hands) | "
                f"stakes {sb:.2f}/{bb:.2f} | target_players {target_players}"
            )
            
            for hand_in_session in range(session_length):
                hand = self._generate_single_hand(session)
                if hand:
                    all_hands.append(hand)
                    hands_generated += 1
            if hands_generated % 100 == 0:
                print(f"  Generated {hands_generated}/{num_hands_to_play} hands...")
                
                session.rotate_button()
                if 0 < hand_in_session < session_length - 1:
                    session.handle_player_changes()
        
        # RANDOMLY SELECT subset
        if num_hands_to_select < len(all_hands):
            print(f"\n=== Randomly selecting {num_hands_to_select} from {len(all_hands)} hands ===")
            selected_hands = self.rng.sample(all_hands, num_hands_to_select)
        else:
            selected_hands = all_hands
        
        with open(output_file, 'w') as f:
            json.dump(selected_hands, f, indent=2)
        
        print(f"\n✓ Saved {len(selected_hands)} hands to {output_file}")
        return selected_hands

    def _create_shuffled_deck(self) -> List[str]:
        """Create a fresh 52-card deck and shuffle it."""
        deck = [f"{rank}{suit}" for rank in self.ranks for suit in self.suits]
        self.rng.shuffle(deck)
        return deck

    def _deal_cards(self, num: int, deck: Optional[List[str]] = None) -> List[str]:
        """
        Deal unique cards from the deck if provided; otherwise fall back to random draws.
        Passing a deck guarantees no duplicates within a hand.
        """
        if deck is not None:
            if len(deck) < num:
                raise ValueError("Not enough cards left in deck to deal the requested number.")
            return [deck.pop() for _ in range(num)]

        cards = []
        for _ in range(num):
            rank = self.rng.choice(self.ranks)
            suit = self.rng.choice(self.suits)
            cards.append(f"{rank}{suit}")
        return cards

    def _generate_single_hand(self, session: TableSession) -> Optional[Dict[str, Any]]:
        # Ensure hero exists and has chips (no hero → no hand)
        hero_idx = session.hero_seat - 1 if session.hero_seat else 0
        if session.players[hero_idx] is None:
            hero_profile = self.rng.choice(session.bot_profiles)
            hero_player = Player(
                uid=HERO_UID,
                seat=session.hero_seat or 1,
                stack=round(self.rng.uniform(8.0, 12.0), 2),
                is_bot=True,
                hands_played=0,
            )
            hero_player.bot_instance = SandboxPokerBot(
                hero_profile, rng_seed=self.rng.randint(0, 10**9)
            )
            session.players[hero_idx] = hero_player

        hero = session.players[hero_idx]
        if hero.stack <= 0:
            hero.stack = session._sample_stack()

        # Remove busted NON-hero players before starting the hand
        for idx, player in enumerate(session.players):
            if idx == hero_idx:
                continue
            if player is not None and player.stack <= 0:
                session.players[idx] = None

        active_players = session.get_active_players()
        if len(active_players) < 2:
            return None
        
        deck = self._create_shuffled_deck()
        
        self.hand_counter += self.rng.randint(1, 100)
        hand_id = str(self.hand_counter)
        session.hand_number += 1
        
        # Deal hole cards
        for p in active_players:
            p.hole_cards = self._deal_cards(2, deck)
            # Let bot calculate strength from CSV
            p.hand_strength = None  # Will be set by bot
            p.folded = False
            p.invested_this_street = 0.0
            p.total_invested = 0.0
            p.hands_played += 1
        
        pot_true = 0.0  # actual chip accounting
        pot_shown = 0.0  # displayed in JSON (match human semantics)
        action_id = 0
        actions = []
        streets_shown = []
        
        # Determine positions
        occupied_seats = [i for i, p in enumerate(session.players) if p is not None]
        if session.button_position not in occupied_seats:
            session.button_position = occupied_seats[0]
        button_idx = occupied_seats.index(session.button_position)
        sb_idx = occupied_seats[(button_idx + 1) % len(occupied_seats)]
        bb_idx = occupied_seats[(button_idx + 2) % len(occupied_seats)]
        
        sb_player = session.players[sb_idx]
        bb_player = session.players[bb_idx]
        
        # Post blinds
        action_id += 1
        sb_amt = min(self.sb, sb_player.stack)
        pot_shown, pot_true = self._add_action(
            actions,
            action_id,
            "preflop",
            sb_player.seat,
            "small_blind",
            sb_amt,
            pot_shown,
            pot_true,
        )
        sb_player.stack -= sb_amt
        sb_player.invested_this_street = sb_amt
        sb_player.total_invested = sb_amt
        
        action_id += 1
        bb_amt = min(self.bb, bb_player.stack)
        pot_shown, pot_true = self._add_action(
            actions,
            action_id,
            "preflop",
            bb_player.seat,
            "big_blind",
            bb_amt,
            pot_shown,
            pot_true,
        )
        bb_player.stack -= bb_amt
        bb_player.invested_this_street = bb_amt
        bb_player.total_invested = bb_amt
        
        current_level = bb_amt
        
        # Preflop
        action_id, pot_shown, pot_true = self._run_betting_round(
            session, active_players, "preflop", action_id, pot_shown, pot_true, current_level, bb_idx, actions, occupied_seats
        )
        still_in = [p for p in active_players if not p.folded]
        if len(still_in) <= 1:
            return self._finalize_hand(session.button_position + 1, active_players, actions, streets_shown, pot_shown, hero)
        
        for p in active_players:
            p.invested_this_street = 0.0
        
        # Flop
        flop = self._deal_cards(3, deck)
        streets_shown.append({"street": "flop", "board_cards": flop})
        action_id, pot_shown, pot_true = self._run_betting_round(
            session, active_players, "flop", action_id, pot_shown, pot_true, 0.0, bb_idx, actions, occupied_seats
        )
        still_in = [p for p in active_players if not p.folded]
        if len(still_in) <= 1:
            return self._finalize_hand(session.button_position + 1, active_players, actions, streets_shown, pot_shown, hero)
        
        for p in active_players:
            p.invested_this_street = 0.0
        
        # Turn
        turn = self._deal_cards(1, deck)
        streets_shown.append({"street": "turn", "board_cards": flop + turn})
        action_id, pot_shown, pot_true = self._run_betting_round(
            session, active_players, "turn", action_id, pot_shown, pot_true, 0.0, bb_idx, actions, occupied_seats
        )
        still_in = [p for p in active_players if not p.folded]
        if len(still_in) <= 1:
            return self._finalize_hand(session.button_position + 1, active_players, actions, streets_shown, pot_shown, hero)
        
        for p in active_players:
            p.invested_this_street = 0.0
        
        # River
        river = self._deal_cards(1, deck)
        streets_shown.append({"street": "river", "board_cards": flop + turn + river})
        action_id, pot_shown, pot_true = self._run_betting_round(
            session, active_players, "river", action_id, pot_shown, pot_true, 0.0, bb_idx, actions, occupied_seats
        )
        
        return self._finalize_hand(session.button_position + 1, active_players, actions, streets_shown, pot_shown, hero)

    def _run_betting_round(self, session, players, street, action_id, pot_shown, pot_true, current_level, bb_seat_idx, actions, occupied_seats):
        active = [p for p in players if not p.folded and p.stack > 0]
        if len(active) <= 1:
            return action_id, pot_shown, pot_true
        
        if street == "preflop":
            start_idx = occupied_seats.index(bb_seat_idx)
            order = occupied_seats[start_idx + 1:] + occupied_seats[:start_idx + 1]
        else:
            order = occupied_seats[:]
        
        action_order = []
        for idx in order:
            player = session.players[idx]
            if player and not player.folded:
                action_order.append(player)
        
        to_act = set(range(len(action_order)))
        last_aggressor = None
        last_aggressor_true_amt = 0.0
        last_aggressor_shown_amt = 0.0
        count = 0
        max_actions = 60
        
        while to_act and count < max_actions:
            count += 1
            for i in list(to_act):
                player = action_order[i]
                if player.folded or player.stack <= 0:
                    to_act.discard(i)
                    continue
                
                to_call = max(0, current_level - player.invested_this_street)
                # PASS HOLE CARDS to bot for better decisions
                decision = self._get_player_decision(player, street, to_call, pot_true, players, current_level)
                action_id += 1
                pre_action_invested = player.invested_this_street
                
                if decision.action == ActionType.FOLD:
                    player.folded = True
                    pot_shown, pot_true = self._add_action(
                        actions, action_id, street, player.seat, "fold", 0, pot_shown, pot_true
                    )
                    to_act.discard(i)
                elif decision.action == ActionType.CHECK:
                    pot_shown, pot_true = self._add_action(
                        actions, action_id, street, player.seat, "check", 0, pot_shown, pot_true
                    )
                    to_act.discard(i)
                elif decision.action == ActionType.CALL:
                    amt = min(to_call, player.stack)
                    player.stack -= amt
                    player.invested_this_street += amt
                    player.total_invested += amt
                    call_target = current_level if to_call > 0 else None
                    pot_shown, pot_true = self._add_action(
                        actions,
                        action_id,
                        street,
                        player.seat,
                        "call",
                        amt,
                        pot_shown,
                        pot_true,
                        raise_to=None,
                        call_to=call_target,
                        delta_true=amt,
                    )
                    last_aggressor = None
                    last_aggressor_true_amt = 0.0
                    last_aggressor_shown_amt = 0.0
                    to_act.discard(i)
                elif decision.action in (ActionType.BET, ActionType.RAISE):
                    raw_amt = min(decision.amount / 100.0, player.stack)
                    if current_level == 0:
                        # Fresh bet
                        bet_size = raw_amt
                        player.stack -= bet_size
                        player.invested_this_street += bet_size
                        player.total_invested += bet_size
                        current_level = bet_size
                        pot_shown, pot_true = self._add_action(
                            actions,
                            action_id,
                            street,
                            player.seat,
                            "bet",
                            bet_size,
                            pot_shown,
                            pot_true,
                            raise_to=None,
                            call_to=None,
                            delta_true=bet_size,
                        )
                        last_aggressor = i
                        last_aggressor_true_amt = bet_size
                        last_aggressor_shown_amt = bet_size
                    else:
                        # Raise: store only raise increment in amount/pot display
                        raise_increment = raw_amt
                        new_level = current_level + raise_increment
                        to_invest = to_call + raise_increment
                        player.stack -= min(player.stack, to_invest)
                        player.invested_this_street += to_invest
                        player.total_invested += to_invest
                        current_level = new_level
                        pot_shown, pot_true = self._add_action(
                            actions,
                            action_id,
                            street,
                            player.seat,
                            "raise",
                            raise_increment,
                            pot_shown,
                            pot_true,
                            raise_to=current_level,
                            call_to=None,
                            delta_true=to_invest,
                        )
                        last_aggressor = i
                        last_aggressor_true_amt = to_invest
                        last_aggressor_shown_amt = raise_increment
                    to_act = {j for j in range(len(action_order)) if j != i and not action_order[j].folded}
                
                if len([p for p in players if not p.folded]) <= 1:
                    if last_aggressor is not None and last_aggressor_shown_amt > 0:
                        pot_shown, pot_true = self._add_action(
                            actions,
                            action_id + 1,
                            street,
                            action_order[last_aggressor].seat,
                            "uncalled_bet_return",
                            last_aggressor_shown_amt,
                            pot_shown,
                            pot_true,
                            delta_shown=-last_aggressor_shown_amt,
                            delta_true=-last_aggressor_true_amt,
                        )
                        action_id += 1
                    return action_id, pot_shown, pot_true
        
        if last_aggressor is not None and last_aggressor_shown_amt > 0:
            pot_shown, pot_true = self._add_action(
                actions,
                action_id + 1,
                street,
                action_order[last_aggressor].seat,
                "uncalled_bet_return",
                last_aggressor_shown_amt,
                pot_shown,
                pot_true,
                delta_shown=-last_aggressor_shown_amt,
                delta_true=-last_aggressor_true_amt,
            )
            action_id += 1
        return action_id, pot_shown, pot_true

    def _get_player_decision(self, player, street, to_call, pot, all_players, current_bet):
        legal = self._get_legal_actions(player, to_call, pot, current_bet)
        if player.bot_instance:
            # ENHANCED: Pass hole cards for CSV lookup
            state = GameState(
                hand_id="temp",
                player_id=player.uid,
                street=Street(street),
                position_index=player.seat - 1,
                num_players=len([p for p in all_players if not p.folded]),
                stack=int(player.stack * 100),
                pot=int(pot * 100),
                to_call=int(to_call * 100),
                big_blind=int(self.bb * 100),
                hand_strength=player.hand_strength,
                hole_cards=player.hole_cards,  # NEW: Pass hole cards
            )
            decision = player.bot_instance.act(state, legal)
            return decision
        
        return BotDecision(ActionType.CHECK)

    def _get_legal_actions(self, player, to_call, pot, current_bet):
        can_check = to_call == 0
        can_call = to_call > 0 and to_call <= player.stack
        can_fold = to_call > 0
        min_bet = max(self.bb, pot * 0.25) if can_check else 0
        max_bet = player.stack if can_check else 0
        can_bet = can_check and player.stack >= min_bet
        min_raise = to_call + max(self.bb, current_bet * 0.5) if to_call > 0 else self.bb
        max_raise = player.stack
        can_raise = to_call > 0 and player.stack >= min_raise
        
        return LegalActions(
            can_fold=can_fold,
            can_check=can_check,
            can_call=can_call,
            call_amount=int(to_call * 100),
            can_bet=can_bet,
            min_bet=int(min_bet * 100),
            max_bet=int(max_bet * 100),
            can_raise=can_raise,
            min_raise=int(min_raise * 100),
            max_raise=int(max_raise * 100)
        )

    def _add_action(self, actions, action_id, street, seat, action_type, amount, pot_shown, pot_true, raise_to=None, call_to=None, delta_shown=None, delta_true=None):
        delta_s = delta_shown if delta_shown is not None else amount
        delta_t = delta_true if delta_true is not None else delta_s
        pot_before = pot_shown
        pot_after = pot_shown + delta_s
        actions.append({
            "action_id": str(action_id),
            "street": street,
            "actor_seat": seat,
            "action_type": action_type,
            "amount": float(round(amount, 2)),
            "raise_to": None if raise_to is None else round(raise_to, 2),
            "call_to": None if call_to is None else round(call_to, 2),
            "normalized_amount_bb": round(amount / self.bb, 1),
            "pot_before": float(round(pot_before, 2)),
            "pot_after": float(round(pot_after, 2)),
        })
        return pot_after, pot_true + delta_t

    def _finalize_hand(self, button_seat, players, actions, streets_shown, pot_shown, hero):
        still_in = [p for p in players if not p.folded]
        showdown = len(still_in) > 1
        if showdown:
            winner = self.rng.choice(still_in)
            reason = "showdown"
        else:
            winner = still_in[0]
            reason = "fold"
        
        rake = round(pot_shown * self.rake_rate, 2)
        payout = round(pot_shown - rake, 2)
        winner.stack += payout

        # Determine which players must reveal hole cards (hero always knows own cards)
        revealed_uids = {p.uid for p in still_in} if showdown else set()
        
        hand = copy.deepcopy(V0_JSON_HAND)
        hand["metadata"] = {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": self.max_seats,
            "hero_seat": hero.seat,
            "hand_ended_on_street": None,
            # Use the actual computed button seat so blinds/metadata stay consistent
            "button_seat": button_seat,
            "sb": self.sb,
            "bb": self.bb,
            "ante": 0.0,
            "rng_seed_commitment": None,
        }
        hand["players"] = [
            {
                "player_uid": p.uid,
                "seat": p.seat,
                "starting_stack": round(p.stack + p.total_invested, 2),
                "hole_cards": p.hole_cards if (p.uid == HERO_UID or p.uid in revealed_uids) else None,
                "showed_hand": bool(p.hole_cards is not None and (p.uid == HERO_UID or p.uid in revealed_uids)),
            }
            for p in sorted(players, key=lambda x: x.seat)
        ]
        hand["streets"] = streets_shown
        hand["actions"] = actions
        hand["outcome"] = {
            "winners": [winner.uid],
            "payouts": {winner.uid: payout},
            "total_pot": round(pot_shown, 2),
            "rake": rake,
            "result_reason": reason,
            "showdown": reason == "showdown",
        }
        hand["label"] = "bot"
        if streets_shown:
            last_board = streets_shown[-1]["board_cards"]
            if len(last_board) == 5:
                hand["metadata"]["hand_ended_on_street"] = "river"
            elif len(last_board) == 4:
                hand["metadata"]["hand_ended_on_street"] = "turn"
            else:
                hand["metadata"]["hand_ended_on_street"] = "flop"
        else:
            hand["metadata"]["hand_ended_on_street"] = "preflop"
        # Canonicalize so button is always seat 1 (match human data rotation)
        hand = self._rotate_to_button_one(hand)
        hand = self._contiguize_seats(hand)
        if not self._hand_is_consistent(hand):
            return None
        return hand

    def _rotate_to_button_one(self, hand: Dict[str, Any]) -> Dict[str, Any]:
        button = hand["metadata"].get("button_seat", 1)
        max_seats = hand["metadata"].get("max_seats", 6)
        shift = (button - 1) % max_seats
        if shift == 0:
            return hand

        def rotate_seat(seat: int) -> int:
            return ((seat - 1 - shift) % max_seats) + 1

        # Rotate players and recompute hero seat
        rotated_players = []
        new_hero_seat = None
        for p in hand["players"]:
            new_seat = rotate_seat(p["seat"])
            if p["player_uid"] == HERO_UID:
                new_hero_seat = new_seat
            p_rot = dict(p)
            p_rot["seat"] = new_seat
            rotated_players.append(p_rot)
        rotated_players.sort(key=lambda x: x["seat"])
        hand["players"] = rotated_players

        # Rotate actions
        for action in hand["actions"]:
            action["actor_seat"] = rotate_seat(action["actor_seat"])

        # Update metadata
        hand["metadata"]["button_seat"] = 1
        if new_hero_seat is not None:
            hand["metadata"]["hero_seat"] = new_hero_seat

        return hand

    def _contiguize_seats(self, hand: Dict[str, Any]) -> Dict[str, Any]:
        seats = sorted({p["seat"] for p in hand["players"]})
        if seats == list(range(1, len(seats) + 1)):
            return hand  # already contiguous

        seat_map = {old: idx + 1 for idx, old in enumerate(seats)}

        # Update players
        for p in hand["players"]:
            p["seat"] = seat_map[p["seat"]]
        hand["players"].sort(key=lambda x: x["seat"])

        # Update actions
        for a in hand["actions"]:
            a["actor_seat"] = seat_map.get(a["actor_seat"], a["actor_seat"])

        # Update metadata seats
        if hand["metadata"].get("hero_seat"):
            hand["metadata"]["hero_seat"] = seat_map.get(hand["metadata"]["hero_seat"], hand["metadata"]["hero_seat"])
        if hand["metadata"].get("button_seat"):
            hand["metadata"]["button_seat"] = seat_map.get(hand["metadata"]["button_seat"], hand["metadata"]["button_seat"])

        return hand

def main():
    profiles = [
        BotProfile(name="tight_aggressive", tightness=0.70, aggression=0.75, bluff_freq=0.05),
        BotProfile(name="loose_aggressive", tightness=0.40, aggression=0.80, bluff_freq=0.12),
        BotProfile(name="tight_passive", tightness=0.68, aggression=0.35, bluff_freq=0.03),
        BotProfile(name="loose_passive", tightness=0.42, aggression=0.30, bluff_freq=0.08),
        BotProfile(name="balanced", tightness=0.55, aggression=0.55, bluff_freq=0.08),
    ]
    
    generator = PokerHandGenerator()
    output_path = Path(__file__).with_name("bot_hands.json")

    generator.generate_hands(
        num_hands_to_play=1000,
        num_hands_to_select=100,
        bot_profiles=profiles,
        output_file=str(output_path),
        hands_per_session=50
    )
    
    print("\n✓ Generation complete!")

if __name__ == "__main__":
    main()
