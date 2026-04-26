from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT.parent / "Poker44-subnet"
UPSTREAM_GENERATOR = UPSTREAM_ROOT / "hands_generator" / "bot_hands" / "generate_poker_data.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic bot hands using the upstream Poker44 generator.")
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "generated_bot_hands.json"))
    parser.add_argument("--num-hands-to-play", type=int, default=40000)
    parser.add_argument("--num-hands-to-select", type=int, default=32000)
    parser.add_argument("--hands-per-session", type=int, default=50)
    parser.add_argument("--seed", type=int, default=424242)
    return parser.parse_args()


def _load_upstream_module():
    if not UPSTREAM_GENERATOR.exists():
        raise FileNotFoundError(
            f"Expected upstream generator at {UPSTREAM_GENERATOR}, but it was not found."
        )
    spec = importlib.util.spec_from_file_location("poker44_upstream_bot_generator", UPSTREAM_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load upstream generator module from {UPSTREAM_GENERATOR}")
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = _load_upstream_module()

    generator = module.PokerHandGenerator(seed=args.seed)
    profiles = [
        module.BotProfile(name="tight_aggressive", tightness=0.70, aggression=0.75, bluff_freq=0.05),
        module.BotProfile(name="loose_aggressive", tightness=0.40, aggression=0.80, bluff_freq=0.12),
        module.BotProfile(name="tight_passive", tightness=0.68, aggression=0.35, bluff_freq=0.03),
        module.BotProfile(name="loose_passive", tightness=0.42, aggression=0.30, bluff_freq=0.08),
        module.BotProfile(name="balanced", tightness=0.55, aggression=0.55, bluff_freq=0.08),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hands = generator.generate_hands(
        num_hands_to_play=args.num_hands_to_play,
        num_hands_to_select=args.num_hands_to_select,
        bot_profiles=profiles,
        output_file=str(output_path),
        hands_per_session=args.hands_per_session,
    )

    if not output_path.exists():
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(hands, handle)

    print(f"Generated {len(hands)} bot hands at {output_path}")


if __name__ == "__main__":
    main()
