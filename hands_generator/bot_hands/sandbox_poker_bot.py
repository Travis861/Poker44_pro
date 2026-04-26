"""
sandbox_poker_bot.py

A RULE-BASED poker bot intended ONLY for a controlled / sandbox environment
to generate labeled behavior for anti-bot detection datasets.

- Uses simple 'if' logic and lightweight heuristics.
- Produces decision logs (features + action + score).
- Supports multiple behavior profiles (tight/aggressive, loose/passive, etc.).
- Not designed or intended for use on real-money platforms or against humans outside a sandbox.

You will need to wire:
- GameState input (current street, pot, stacks, position, legal actions)
- Action output (fold/call/raise size)
to your environment.

Author: (you)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import json
import pandas as pd

# ----------------------------
# Types / Contracts
# ----------------------------

class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass
class LegalActions:
    """
    A minimal representation of what the environment allows at this decision point.
    - If can_check is True, "check" is a valid action.
    - If can_call is True, "call" is valid (call_amount > 0).
    - If can_bet is True, "bet" is valid (min_bet <= amount <= max_bet).
    - If can_raise is True, "raise" is valid (min_raise <= amount <= max_raise).
    """
    can_fold: bool
    can_check: bool
    can_call: bool
    call_amount: int

    can_bet: bool
    min_bet: int
    max_bet: int

    can_raise: bool
    min_raise: int
    max_raise: int


@dataclass
class GameState:
    """
    Minimal state required for heuristic decisions.
    Extend as needed.

    NOTE: In real poker, good play needs hand strength estimation.
    For sandbox dataset generation, we can accept an environment-provided
    'hand_strength' signal in [0, 1] (e.g., from a simulator / oracle).

    If you do NOT want an oracle, set hand_strength=None and the bot will
    rely more on position + pot odds + randomness.
    """
    hand_id: str
    player_id: str

    street: Street
    position_index: int         # 0 = earliest, higher = later position
    num_players: int

    stack: int                  # current bot stack
    pot: int                    # current pot size
    to_call: int                # amount needed to call (0 if check possible)
    big_blind: int

    # Optional "oracle" signal for sandbox only (0..1).
    # Example: 0.15 weak, 0.5 medium, 0.85 strong.
    hand_strength: Optional[float] = None

    # Optional meta signals (useful for dataset richness)
    last_action_was_aggressive: bool = False
    opponent_aggro_score: Optional[float] = None
    # hole cards for better decision making
    hole_cards: Optional[List[str]] = None

@dataclass
class BotProfile:
    """
    Behavior knobs (use these to create many bot families/patterns).
    """
    name: str = "balanced_v0"

    # Preflop tendencies
    tightness: float = 0.55          # higher = plays fewer hands
    aggression: float = 0.55         # higher = bets/raises more
    bluff_freq: float = 0.08         # chance to bluff when weak

    # Risk controls
    max_risk_fraction_of_stack: float = 0.18  # won't commit more than this fraction w/out strong signal
    tilt_factor: float = 0.0                 # increases aggression after losses (sandbox signal if you want)

    # Sizing behavior
    bet_pot_fraction_small: float = 0.33
    bet_pot_fraction_medium: float = 0.55
    bet_pot_fraction_large: float = 0.80
    # Structural behavior controls used to shape how often hands end early.
    preflop_defend_bias: float = 0.0      # lower -> fold more often vs opens/3bets
    postflop_continue_bias: float = 0.0   # lower -> fold/check more often after flop
    trap_frequency: float = 0.0           # lower -> fewer slowplays/call-downs with strong hands


@dataclass
class BotDecision:
    action: ActionType
    amount: int = 0  # used for bet/raise; 0 for fold/check/call where irrelevant
    meta: Dict[str, Any] = None


# ----------------------------
# Enhanced Sandbox Bot
# ----------------------------

class SandboxPokerBot:
    """
    Rule-based sandbox bot.
    - Decision logic is intentionally simple.
    - Use profiles to generate many different patterns.
    """

    def __init__(self, profile: BotProfile, rng_seed: Optional[int] = None):
        self.profile = profile
        self.rng = random.Random(rng_seed)
        self.session_stats = {
            "hands_seen": 0,
            "hands_played": 0,
            "aggressive_actions": 0,
        }
        
        # Load hand strengths from CSV (Method I)
        self.starting_strengths = self._load_hand_strengths()
        
        # print(f"STARTING STRENGTH : {self.starting_strengths}")
        
        
    def _load_hand_strengths(self) -> Dict[str, float]:
        """Load pre-computed hand strengths from CSV."""
        try:
            csv_path = Path(__file__).resolve().with_name("hole_strengths.csv")
            if not csv_path.exists():
                return {}
            calculated_df = pd.read_csv(csv_path)
            holes = calculated_df.Holes
            strengths = calculated_df.Strengths
            return dict(zip(holes, strengths))
        except Exception:
            return {}

    def _rank_to_numeric(self, rank: str) -> int:
        """Convert rank to numeric value."""
        if rank.isnumeric():
            return int(rank)
        return [10, 11, 12, 13, 14]['TJQKA'.index(rank)]
    
    def _hole_list_to_key(self, hole: List[str]) -> str:
        """Convert hole cards to CSV lookup key."""
        if not hole or len(hole) != 2:
            return None
        card1, card2 = hole[0], hole[1]
        rank1, suit1 = card1[0], card1[1]
        rank2, suit2 = card2[0], card2[1]
        numeric1 = self._rank_to_numeric(rank1)
        numeric2 = self._rank_to_numeric(rank2)
        
        if numeric1 == numeric2:
            # Pocket pair: always use 'o' convention
            return rank1 + rank2 + 'o'  # rank1 == rank2, order doesn't matter
        else:
            # Non-pair: determine suited/offsuit and order higher rank first
            suited = suit1 == suit2
            suit_string = 's' if suited else 'o'
            
            if numeric1 >= numeric2:
                return rank1 + rank2 + suit_string
            else:
                return rank2 + rank1 + suit_string
        
    def _get_hand_strength_from_csv(self, hole_cards: List[str]) -> Optional[float]:
        """Get hand strength from pre-computed CSV."""
        
        key = str(self._hole_list_to_key(hole_cards))
        # print(f"Hole cards: {hole_cards}, converted keys: {key}")
        if key and key in self.starting_strengths:
            # print(f"strength: {self.starting_strengths[key]}")
            return self.starting_strengths[key]
        return None

    # --------- Public API ---------

    def act(self, state: GameState, legal: LegalActions) -> BotDecision:
        """
        Main decision entrypoint.
        """
        self.session_stats["hands_seen"] += 1
        
        # Try to get CSV-based strength first (Method I)
        if state.hole_cards:
            csv_strength = self._get_hand_strength_from_csv(state.hole_cards)
            if csv_strength is not None:
                state.hand_strength = csv_strength
        
        # Compute features used by simple if-logic
        pos_factor = self._position_factor(state.position_index, state.num_players)
        pot_odds = self._pot_odds(state.to_call, state.pot)
        hs = state.hand_strength  # None allowed

        strength_bucket = self._bucket_strength(hs, pos_factor)

        # Decide action via street-specific logic
        if state.street == Street.PREFLOP:
            decision = self._decide_preflop(state, legal, strength_bucket, pos_factor, pot_odds)
        else:
            decision = self._decide_postflop(state, legal, strength_bucket, pos_factor, pot_odds)

        # Add meta for logging/dataset
        decision.meta = decision.meta or {}
        decision.meta.update({
            "profile": self.profile.name,
            "pos_factor": round(pos_factor, 3),
            "pot_odds": round(pot_odds, 3),
            "hand_strength": None if hs is None else round(hs, 3),
            "strength_bucket": strength_bucket,
            "street": state.street.value,
            "stack": state.stack,
            "pot": state.pot,
            "to_call": state.to_call,
        })

        # Update session stats
        if decision.action in (ActionType.BET, ActionType.RAISE):
            self.session_stats["aggressive_actions"] += 1
            
        return decision

    def export_session_stats(self) -> Dict[str, Any]:
        return dict(self.session_stats)


    # --------- Core Logic (IF-based) ---------

    def _decide_preflop(
        self,
        state: GameState,
        legal: LegalActions,
        strength_bucket: str,
        pos_factor: float,
        pot_odds: float
    ) -> BotDecision:
        """
        Very simplified preflop rules.
        Buckets: "weak", "medium", "strong".
        Uses tightness/aggression + position factor to vary behavior.
        """
        # If no call amount (can check), treat as opening opportunity
        opening = (state.to_call == 0 and legal.can_check)
        defend_bias = self._bounded_bias(self.profile.preflop_defend_bias, scale=0.20)

        # Base willingness to play hands (tightness)
        # Later position => slightly looser
        play_threshold = self.profile.tightness - (0.10 * pos_factor) - (0.08 * defend_bias)
        
        # If we have actual hand strength from CSV, use it
        hs = state.hand_strength
        if hs is None:
            pseudo_strength = self.rng.random() * 0.65 + 0.35 * pos_factor
            if pseudo_strength < play_threshold and legal.can_fold:
                return BotDecision(ActionType.FOLD, 0, {"reason": "preflop_pseudo_fold"})
            strength_bucket = "medium"
            hs = pseudo_strength
        
        # WEAK hands
        if strength_bucket == "weak":
            # Method I: Consider pot odds and threat level
            if state.to_call > 0 and legal.can_fold:
                threat_level = state.to_call / max(1, state.big_blind)
                if threat_level > 4:  # Big raise
                    return BotDecision(ActionType.FOLD, 0, {"reason": "weak_fold_big_raise"})
                
                # Good pot odds + late position = defend occasionally
                defend_prob = 0.30 + (0.18 * defend_bias)
                if pot_odds < 0.20 and pos_factor > 0.60 and self.rng.random() < defend_prob:
                    return BotDecision(ActionType.CALL if legal.can_call else ActionType.FOLD, 
                                     state.to_call, {"reason": "weak_defend_good_odds"})
                return BotDecision(ActionType.FOLD, 0, {"reason": "weak_fold"})
            # Opening: usually check if possible, otherwise fold
            if opening and legal.can_check:
                return BotDecision(ActionType.CHECK, 0, {"reason": "weak_check_opening"})
        
        # MEDIUM hands
        if strength_bucket == "medium":
            if opening:
                # Open sometimes depending on aggression and position
                open_prob = (
                    0.26
                    + 0.46 * self.profile.aggression * (0.45 + 0.55 * pos_factor)
                    + (0.10 * defend_bias)
                )
                if legal.can_bet and self.rng.random() < open_prob:
                    amt = self._size_open_raise(state, legal)
                    return BotDecision(ActionType.BET, amt, {"reason": "medium_open_bet"})
                return BotDecision(ActionType.CHECK, 0, {"reason": "medium_check_opening"})

            # Facing a raise: continue selectively, not by default.
            if legal.can_call:
                if self._risk_too_high(state.to_call, state.stack) and legal.can_fold:
                    return BotDecision(ActionType.FOLD, 0, {"reason": "medium_fold_risk_cap"})

                threat_level = state.to_call / max(1, state.big_blind)
                raise_prob = max(0.0, 0.24 + 0.22 * self.profile.aggression + (0.12 * defend_bias))
                if legal.can_raise and hs > 0.60 and threat_level <= 5 and self.rng.random() < raise_prob:
                    amt = self._size_raise(state, legal, large=False)
                    return BotDecision(ActionType.RAISE, amt, {"reason": "medium_preflop_reraise"})

                call_prob = max(0.12, 0.44 + (0.22 * defend_bias))
                if (((pot_odds < 0.24 and threat_level <= 5) or hs > 0.56) and self.rng.random() < call_prob):
                    return BotDecision(ActionType.CALL, state.to_call, {"reason": "medium_call"})
                if legal.can_fold:
                    return BotDecision(ActionType.FOLD, 0, {"reason": "medium_fold_bad_odds"})
        
        # STRONG hands
        if strength_bucket == "strong":
            # Strong hands: prefer aggression
            if opening:
                if legal.can_bet:
                    amt = self._size_open_raise(state, legal, strong=True)
                    return BotDecision(ActionType.BET, amt, {"reason": "strong_open_bet"})
                return BotDecision(ActionType.CHECK, 0, {"reason": "strong_check_fallback"})

            # Facing action: raise sometimes, otherwise call
            if legal.can_raise and self.rng.random() < (0.60 + 0.20 * self.profile.aggression):
                amt = self._size_raise(state, legal, large=True)
                return BotDecision(ActionType.RAISE, amt, {"reason": "strong_reraise"})
            if legal.can_call and not self._risk_too_high(state.to_call, state.stack):
                return BotDecision(ActionType.CALL, state.to_call, {"reason": "strong_call"})
            if legal.can_fold:
                return BotDecision(ActionType.FOLD, 0, {"reason": "strong_fold_risk_cap"})

        # Fallbacks
        if legal.can_check:
            return BotDecision(ActionType.CHECK, 0, {"reason": "preflop_fallback_check"})
        if legal.can_call:
            return BotDecision(ActionType.CALL, state.to_call, {"reason": "preflop_fallback_call"})
        return BotDecision(ActionType.FOLD, 0, {"reason": "preflop_fallback_fold"})

    def _decide_postflop(
        self,
        state: GameState,
        legal: LegalActions,
        strength_bucket: str,
        pos_factor: float,
        pot_odds: float
    ) -> BotDecision:
        """
        Simplified postflop logic:
        - Weak: check/fold; occasional bluff.
        - Medium: pot-odds aware calls; selective aggression.
        - Strong: value bet/raise.
        """
        # Preflop hole-card strength should only weakly inform postflop lines.
        hs = self._effective_postflop_strength(state, pos_factor)
        strength_bucket = self._bucket_strength(hs, pos_factor)
        continue_bias = self._bounded_bias(self.profile.postflop_continue_bias, scale=0.22)
        trap_bias = self._bounded_bias(self.profile.trap_frequency, scale=0.18)

        # Determine if we are facing a bet
        facing_bet = (state.to_call > 0 and legal.can_call)
        
        # Calculate threat level (Method I concept)
        threat_level = 0
        if facing_bet:
            threat_level = state.to_call / max(1, state.big_blind)
        
        # WEAK hands
        if strength_bucket == "weak":
            # Bluff opportunity
            if not facing_bet and (legal.can_bet or legal.can_raise):
                bluff_prob = self.profile.bluff_freq * (0.42 + 0.44 * pos_factor) * (1.0 + 0.25 * continue_bias)
                if self.rng.random() < bluff_prob:
                    amt = self._size_bet(state, legal, small=True)
                    act = ActionType.BET if legal.can_bet else ActionType.RAISE
                    return BotDecision(act, amt, {"reason": "weak_bluff"})
            
            # Facing bet: fold unless great odds
            if facing_bet:
                # Method I: Reduce strength when facing aggression
                if threat_level > 4:
                    adjusted_strength = max(0, hs - 0.17)
                else:
                    adjusted_strength = hs
                
                # Good pot odds = consider calling rarely, mostly on flop in position.
                peel_prob = max(0.02, 0.08 + 0.08 * pos_factor + (0.14 * continue_bias))
                if (
                    state.street == Street.FLOP
                    and pot_odds < 0.13
                    and adjusted_strength >= pot_odds + 0.12
                    and self.rng.random() < peel_prob
                ):
                    return BotDecision(ActionType.CALL, state.to_call, {"reason": "weak_peel_good_odds"})
                
                if legal.can_fold:
                    return BotDecision(ActionType.FOLD, 0, {"reason": "weak_fold_postflop"})
            
            if legal.can_check:
                return BotDecision(ActionType.CHECK, 0, {"reason": "weak_check"})
        
        # MEDIUM hands
        if strength_bucket == "medium":
            if facing_bet:
                if self._risk_too_high(state.to_call, state.stack) and legal.can_fold:
                    return BotDecision(ActionType.FOLD, 0, {"reason": "medium_fold_risk_cap_postflop"})

                required_edge = 0.08 if state.street == Street.FLOP else 0.14
                if hs >= pot_odds + required_edge:
                    raise_prob = max(
                        0.02,
                        0.12 + 0.18 * self.profile.aggression * pos_factor + (0.12 * continue_bias),
                    )
                    if legal.can_raise and hs > 0.62 and self.rng.random() < raise_prob:
                        amt = self._size_raise(state, legal, large=False)
                        return BotDecision(ActionType.RAISE, amt, {"reason": "medium_raise_semibluff"})
                    call_prob = max(0.08, 0.18 + 0.12 * pos_factor + (0.20 * continue_bias))
                    if state.street in (Street.TURN, Street.RIVER):
                        call_prob -= 0.04
                    if self.rng.random() < call_prob:
                        return BotDecision(ActionType.CALL, state.to_call, {"reason": "medium_call_postflop"})
                if legal.can_fold:
                    return BotDecision(ActionType.FOLD, 0, {"reason": "medium_fold_bad_odds"})
            
            # Not facing bet: bet for value
            value_bet_prob = max(0.04, 0.10 + 0.18 * self.profile.aggression + (0.10 * continue_bias))
            if legal.can_bet and self.rng.random() < value_bet_prob:
                amt = self._size_bet(state, legal, small=False)
                return BotDecision(ActionType.BET, amt, {"reason": "medium_value_bet"})
            
            if legal.can_check:
                return BotDecision(ActionType.CHECK, 0, {"reason": "medium_check"})
        
        # STRONG hands
        if strength_bucket == "strong":
            # Method I: Bet deeper with strong hands on turn/river
            is_late_street = state.street in (Street.TURN, Street.RIVER)
            
            raise_prob = max(0.18, 0.48 + 0.18 * self.profile.aggression + (0.12 * continue_bias) - (0.10 * trap_bias))
            if facing_bet and legal.can_raise and self.rng.random() < raise_prob:
                amt = self._size_raise(state, legal, large=True)
                return BotDecision(ActionType.RAISE, amt, {"reason": "strong_raise_value"})
            
            if facing_bet and legal.can_call and not self._risk_too_high(state.to_call, state.stack):
                call_prob = (0.24 if is_late_street else 0.28) + (0.16 * trap_bias) + (0.08 * continue_bias)
                if self.rng.random() < call_prob:
                    return BotDecision(ActionType.CALL, state.to_call, {"reason": "strong_call_trap"})
            if facing_bet and legal.can_fold and self._risk_too_high(state.to_call, state.stack):
                return BotDecision(ActionType.FOLD, 0, {"reason": "strong_fold_risk_cap_postflop"})
            
            if facing_bet and legal.can_call and trap_bias > -0.05:
                return BotDecision(ActionType.CALL, state.to_call, {"reason": "strong_call_trap"})
            
            if not facing_bet and legal.can_bet:
                # Bet bigger on later streets with strong hands
                large = is_late_street and hs > 0.75
                amt = self._size_bet(state, legal, large=large)
                return BotDecision(ActionType.BET, amt, {"reason": "strong_value_bet"})
        
        # Fallbacks
        if legal.can_check:
            return BotDecision(ActionType.CHECK, 0, {"reason": "postflop_fallback_check"})
        if legal.can_call:
            return BotDecision(ActionType.CALL, state.to_call, {"reason": "postflop_fallback_call"})
        return BotDecision(ActionType.FOLD, 0, {"reason": "postflop_fallback_fold"})

    # --------- Helpers ---------
    def _position_factor(self, position_index: int, num_players: int) -> float:
        if num_players <= 1:
            return 0.5
        return max(0.0, min(1.0, position_index / (num_players - 1)))

    def _pot_odds(self, to_call: int, pot: int) -> float:
        """
        Simple pot odds approximation: to_call / (pot + to_call).
        Lower is better for calling.
        """
        denom = pot + max(0, to_call)
        if denom <= 0:
            return 1.0
        return max(0.0, min(1.0, to_call / denom))

    def _bucket_strength(self, hand_strength: Optional[float], pos_factor: float) -> str:
        """
        Convert a 0..1 strength into buckets.
        If hand_strength is None, return 'medium' by default.
        """
        if hand_strength is None:
            return "medium"
        # Slightly "inflate" playable strength in later position (sandbox realism)
        adj = min(1.0, max(0.0, hand_strength + 0.06 * (pos_factor - 0.5)))
        if adj < 0.38:
            return "weak"
        if adj < 0.72:
            return "medium"
        return "strong"

    def _bounded_bias(self, value: float, *, scale: float) -> float:
        return max(-scale, min(scale, float(value or 0.0)))

    def _effective_postflop_strength(self, state: GameState, pos_factor: float) -> float:
        """Blend preflop strength back toward the mean so later streets are less deterministic."""
        hs = state.hand_strength
        continue_bias = self._bounded_bias(self.profile.postflop_continue_bias, scale=0.20)
        if hs is None:
            baseline = 0.24 + 0.34 * self.rng.random() + 0.06 * pos_factor + (0.10 * continue_bias)
            return max(0.0, min(1.0, baseline))

        street_anchor = {
            Street.FLOP: 0.46,
            Street.TURN: 0.42,
            Street.RIVER: 0.38,
        }.get(state.street, 0.50)
        street_noise = {
            Street.FLOP: 0.10,
            Street.TURN: 0.11,
            Street.RIVER: 0.12,
        }.get(state.street, 0.10)
        blended = (0.42 * hs) + (0.46 * street_anchor) + (0.06 * pos_factor) + (0.12 * continue_bias)
        blended += self.rng.uniform(-street_noise, street_noise)
        return max(0.0, min(1.0, blended))

    def _risk_too_high(self, amount: int, stack: int) -> bool:
        """
        Enforce a simple risk cap: don't invest too much of stack without strong signal.
        """
        if stack <= 0:
            return True
        return (amount / stack) > self.profile.max_risk_fraction_of_stack

    def _clamp(self, x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, x))

    def _size_open_raise(self, state: GameState, legal: LegalActions, strong: bool = False) -> int:
        """
        Sandbox sizing for opening bet. Uses big blind and pot.
        """
        # Typical open sizes: 2.2bb to 3.5bb (very simplified)
        bb = max(1, state.big_blind)
        base = int((2.5 if not strong else 3.0) * bb)
        # Add small randomness so bots aren't identical
        base += self.rng.randint(0, int(0.6 * bb))
        if legal.can_bet:
            return self._clamp(base, legal.min_bet, legal.max_bet)
        return base

    def _size_bet(self, state: GameState, legal: LegalActions, small: bool = False, large: bool = False) -> int:
        """
        Postflop bet sizing as fraction of pot with jitter.
        """
        if not legal.can_bet:
            return 0
        pot = max(1, state.pot)
        if small:
            frac = self.profile.bet_pot_fraction_small
        elif large:
            frac = self.profile.bet_pot_fraction_large
        else:
            frac = self.profile.bet_pot_fraction_medium

        amt = int(pot * frac)
        amt += self.rng.randint(0, max(1, int(0.08 * pot)))
        return self._clamp(amt, legal.min_bet, legal.max_bet)

    def _size_raise(self, state: GameState, legal: LegalActions, large: bool = False) -> int:
        """
        Raise sizing based on call amount + pot fraction.
        """
        if not legal.can_raise:
            return 0
        pot = max(1, state.pot)
        # Simple: raise bigger when "large", otherwise moderate
        extra = int(pot * (0.68 if large else 0.48))
        amt = state.to_call + extra
        amt += self.rng.randint(0, max(1, int(0.08 * pot)))
        return self._clamp(amt, legal.min_raise, legal.max_raise)


# ----------------------------
# Example usage (sandbox)
# ----------------------------

def example():
    profile = BotProfile(
        name="balanced_v0",
        tightness=0.58,
        aggression=0.52,
        bluff_freq=0.06,
        max_risk_fraction_of_stack=0.20,
    )
    bot = SandboxPokerBot(profile, rng_seed=42)

    # Fake state snapshot (you would build this from your environment)
    state = GameState(
        hand_id="H123",
        player_id="BOT_001",
        street=Street.FLOP,
        position_index=4,
        num_players=6,
        stack=10000,
        pot=1200,
        to_call=300,
        big_blind=100,
        hand_strength=0.62,  # sandbox "oracle" example
        last_action_was_aggressive=True,
    )

    legal = LegalActions(
        can_fold=True,
        can_check=False,
        can_call=True,
        call_amount=300,
        can_bet=False,
        min_bet=0,
        max_bet=0,
        can_raise=True,
        min_raise=800,
        max_raise=6000,
    )

    decision = bot.act(state, legal)
    print("Decision:", decision.action.value, decision.amount)
    print("Meta:", json.dumps(decision.meta, indent=2))


if __name__ == "__main__":
    example()
