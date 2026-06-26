"""Report why multi-turn turns ended without an executable tool call.

Each multi-turn turn that decodes to no call ends with a `handler_log` entry in
the result file's `inference_log` (see BaseHandler._empty_response_reason and the
AWMFormatHandler override). This scans result files and tallies those terminal
reasons so an eval run's "Empty response" turns can be told apart — a benign
"task complete" finish vs. a truncated generation vs. an undecodable tool call.

Usage:
    python -m bfcl_eval.scripts.empty_reason_report <result-file-or-dir> [--json]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# (substring to match in the handler_log content, category). Order matters: the
# specific AWM reasons are checked before the generic default, which is a prefix
# of none of them. "Successfully decoded model response." is per-step and skipped.
_PATTERNS = [
    ("final answer with no tool call", "task_complete"),
    ("truncated mid-reasoning", "truncated"),
    ("Empty model output", "empty_output"),
    ("undecodable tool call", "undecodable_call"),
    ("Empty response from the model", "generic_empty"),
    ("Error decoding the model response", "decode_error"),
    ("forced to quit after", "force_quit"),
]
_BENIGN = {"task_complete"}

_LABELS = {
    "task_complete": "task_complete (final answer, no tool call)",
    "truncated": "truncated (hit generation length cap)",
    "empty_output": "empty_output (no content)",
    "undecodable_call": "undecodable_call (direct/malformed call)",
    "generic_empty": "generic_empty (unclassified)",
    "decode_error": "decode_error (decode raised)",
    "force_quit": "force_quit (step cap)",
}


def _classify(content: str) -> str | None:
    for needle, category in _PATTERNS:
        if needle in content:
            return category
    return None


def _handler_log_contents(turn) -> list[str]:
    """All handler_log `content` strings inside one inference_log turn.

    A turn is a dict of `step_N -> [entries]`; state-log turns are bare lists.
    """
    steps = turn.values() if isinstance(turn, dict) else [turn]
    contents = []
    for step in steps:
        if not isinstance(step, list):
            continue
        for entry in step:
            if isinstance(entry, dict) and entry.get("role") == "handler_log":
                contents.append(entry.get("content", ""))
    return contents


def _iter_entries(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A result .json file or a directory of them.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    root = Path(args.path)
    files = sorted(root.rglob("*_result.json")) if root.is_dir() else [root]

    counts = Counter()
    n_episodes = n_turns = 0
    non_benign = []  # (scenario_id, turn_index, category)

    for file in files:
        for entry in _iter_entries(file):
            inference_log = entry.get("inference_log")
            if not inference_log:
                continue
            n_episodes += 1
            for turn_index, turn in enumerate(inference_log):
                n_turns += 1
                category = None
                for content in _handler_log_contents(turn):
                    hit = _classify(content)
                    if hit:
                        category = hit  # last terminal message wins
                if category:
                    counts[category] += 1
                    if category not in _BENIGN:
                        non_benign.append((entry["id"], turn_index, category))

    if args.json:
        print(json.dumps(
            {
                "files": len(files),
                "episodes": n_episodes,
                "turns": n_turns,
                "counts": dict(counts),
                "non_benign": [
                    {"id": i, "turn": t, "category": c} for i, t, c in non_benign
                ],
            },
            indent=2,
        ))
        return

    print(f"Empty/terminal-turn report  "
          f"({len(files)} file(s), {n_episodes} episodes, {n_turns} turns)\n")
    width = max((len(_LABELS[c]) for c in counts), default=len("terminal turns total"))
    for category in sorted(counts, key=lambda c: -counts[c]):
        label = _LABELS.get(category, category)
        print(f"  {label:<{width}}  {counts[category]:>5}")
    print(f"  {'terminal turns total':<{width}}  {sum(counts.values()):>5}")

    if non_benign:
        print(f"\nNon-benign terminations ({len(non_benign)}):")
        for scenario_id, turn_index, category in non_benign:
            print(f"  {scenario_id}  turn{turn_index}  {category}")


if __name__ == "__main__":
    main()
