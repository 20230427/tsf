#!/usr/bin/env python
"""Apply a validation-only selection result to its dataset config.

Reads a ``checkpoints/selection_<name>.json`` produced by
``run_selection_protocol.py`` and writes the **validation-selected** switch
values back into the config YAML named in that file (comments preserved via
line-level editing). This removes the human-in-the-loop, test-informed choice
(e.g. Solar's RevIN flag) so the batch is fully hands-off and the reported
configuration is the one chosen on validation data alone.

Usage:
    python scripts/apply_selection.py checkpoints/selection_solar.json
    python scripts/apply_selection.py checkpoints/selection_*.json --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yaml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


def update_key(lines, section, key, val):
    """Set ``section.key`` to ``val`` in YAML ``lines`` (list of strings).

    Returns (changed, old_value_str_or_None). Preserves any trailing comment.
    If the key is absent from the section, inserts it right after the header.
    """
    header = re.compile(rf"^{re.escape(section)}:\s*(#.*)?$")
    kv = re.compile(rf"^(\s+){re.escape(key)}:\s*([^\s#]+)?[ \t]*(#.*)?$")
    header_idx = None
    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        if header.match(s):
            header_idx = i
            continue
        if header_idx is not None:
            # a new top-level section (non-indented, non-blank, non-comment) ends the block
            if re.match(r"^[^\s#]", s):
                break
            m = kv.match(s)
            if m:
                indent, old, comment = m.group(1), m.group(2), m.group(3) or ""
                pad = "  " if comment else ""
                lines[i] = f"{indent}{key}: {yaml_val(val)}{pad}{comment}\n".rstrip() + "\n"
                return True, old
    if header_idx is None:
        raise KeyError(f"section '{section}:' not found")
    # key missing -> insert after header
    lines.insert(header_idx + 1, f"  {key}: {yaml_val(val)}\n")
    return True, None


def apply_one(json_path: Path, dry_run: bool) -> None:
    rep = json.loads(json_path.read_text(encoding="utf-8"))
    selection = rep.get("selection")
    protocol = rep.get("protocol")
    models = rep.get("models")
    if not isinstance(selection, dict) or not isinstance(protocol, dict) or not isinstance(models, dict):
        raise ValueError(f"{json_path} is not a complete selection-protocol report")
    if selection.get("test_metrics_used") is not False:
        raise ValueError(f"{json_path} does not certify validation-only selection")
    if (
        protocol.get("candidate_test_loader_constructed") is not False
        or protocol.get("candidate_test_metrics_emitted") is not False
        or protocol.get("final_test_phase") != "after_configuration_freeze"
    ):
        raise ValueError(f"{json_path} violates the no-test candidate protocol")
    if "test_oracle" in models:
        raise ValueError(f"{json_path} contains a test-oracle model and cannot update configs")
    selected_model = models.get("validation_selected")
    if not isinstance(selected_model, dict) or not isinstance(selected_model.get("candidate"), dict):
        raise ValueError(f"{json_path} lacks a frozen validation-selected candidate")
    if selected_model["candidate"] != selection.get("selected_candidate"):
        raise ValueError(f"{json_path} has inconsistent frozen-candidate records")
    cfg_rel = rep["config"]
    candidate = selected_model["candidate"]
    cfg_path = ROOT / cfg_rel
    if not cfg_path.exists():
        print(f"[skip] {json_path.name}: config {cfg_rel} not found")
        return
    lines = cfg_path.read_text().splitlines(keepends=True)
    changes = []
    for dotted, val in candidate.items():
        section, _, key = dotted.partition(".")
        if not key:  # bare key -> model section (matches run_selection_protocol)
            section, key = "model", section
        changed, old = update_key(lines, section, key, val)
        changes.append(f"{section}.{key}: {old} -> {yaml_val(val)}")
    verb = "would set" if dry_run else "set"
    print(f"[{cfg_rel}] validation-selected ({verb}): " + "; ".join(changes))
    if not dry_run:
        cfg_path.write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selection_json", nargs="+",
                    help="one or more checkpoints/selection_<name>.json files")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the changes without writing the configs")
    args = ap.parse_args()
    paths = []
    for pat in args.selection_json:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.exit("No selection JSON files matched.")
    for p in paths:
        apply_one(Path(p), args.dry_run)


if __name__ == "__main__":
    main()
