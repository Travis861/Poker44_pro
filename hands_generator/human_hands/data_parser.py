"""Concatenate all PokerStars hand history .txt files into one dataset.

This script scans `hands_generator/human_hands/poker_hands` (2023/2024/2025
subfolders), appends every `.txt` file in deterministic order, and writes the
combined result to `massive_data.txt` alongside the existing `data.txt`.
"""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).parent
SOURCE_ROOT = BASE_DIR / "poker_hands"
OUTPUT_PATH = BASE_DIR / "massive_data.txt"


def iter_hand_files() -> list[Path]:
    """Return all .txt files under SOURCE_ROOT in a stable, sorted order."""

    files: list[Path] = []
    if not SOURCE_ROOT.exists():
        return files

    for path in sorted(SOURCE_ROOT.rglob("*.txt")):
        if path.is_file():
            files.append(path)
    return files


def build_massive_file() -> None:
    hand_files = iter_hand_files()
    if not hand_files:
        print(f"No .txt files found under {SOURCE_ROOT}")
        return

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for idx, path in enumerate(hand_files):
            text = path.read_text(encoding="utf-8")
            # Ensure there is a blank line between files to keep hand splits intact.
            out_f.write(text.rstrip())
            if idx != len(hand_files) - 1:
                out_f.write("\n\n")

    print(f"Wrote {len(hand_files)} files into {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    build_massive_file()
