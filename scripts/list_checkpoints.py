"""List all Tinker checkpoints created under your API key.

Usage:
    TINKER_API_KEY=... python scripts/list_checkpoints.py
    TINKER_API_KEY=... python scripts/list_checkpoints.py --type state   # only durable state ckpts
    TINKER_API_KEY=... python scripts/list_checkpoints.py --json         # raw dump

Pages through RestClient.list_user_checkpoints() (100/page) and prints newest-first with size + expiry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import tinker


def _human_size(n: int | None) -> str:
    if not n:
        return "-"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f}{unit}" if unit == "B" else f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def _fmt_time(t) -> str:
    if t is None:
        return "-"
    if isinstance(t, (int, float)):
        return dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    return str(t)[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default=None, help="filter by checkpoint_type (e.g. state, sampler)")
    ap.add_argument("--no-expiry", action="store_true",
                    help="show only checkpoints with no expiry set (persist until deleted)")
    ap.add_argument("--json", action="store_true", help="dump raw records as JSON")
    args = ap.parse_args()

    rest = tinker.ServiceClient().create_rest_client()

    ckpts = []
    offset = 0
    while True:
        page = rest.list_user_checkpoints(limit=100, offset=offset).result()
        batch = page.checkpoints
        ckpts.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    if args.type:
        ckpts = [c for c in ckpts if c.checkpoint_type == args.type]
    if args.no_expiry:
        ckpts = [c for c in ckpts if c.expires_at is None]

    if args.json:
        print(json.dumps([c.model_dump() if hasattr(c, "model_dump") else vars(c) for c in ckpts],
                         default=str, indent=2))
        return

    print(f"{len(ckpts)} checkpoint(s)\n")
    print(f"{'time':16}  {'type':10}  {'size':>8}  {'expires':16}  tinker_path")
    print("-" * 110)
    for c in sorted(ckpts, key=lambda c: str(c.time), reverse=True):
        print(f"{_fmt_time(c.time):16}  {str(c.checkpoint_type):10}  {_human_size(c.size_bytes):>8}  "
              f"{_fmt_time(c.expires_at):16}  {c.tinker_path}")


if __name__ == "__main__":
    main()
