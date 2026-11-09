"""KTRL command-line entry point."""

import argparse
import json
from pathlib import Path

from .configs.recipe import load_plan


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ktrl")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="resolve a KTRL recipe")
    run.add_argument("--config", required=True, type=Path, help="KTRL YAML recipe")
    run.add_argument("--dry-run", action="store_true", help="print the resolved plan as JSON")
    run.add_argument("--steps", type=int, help="override the planned update limit (1 or 2)")
    run.add_argument("--profiling", choices=("light", "full"), help="override the profiling preset")
    run.add_argument("--output-dir", type=Path, help="fresh qjh005 results/ktrl-m4-* directory")
    rollout = commands.add_parser("rollout", help="run one KT-SGLang request on qjh005 and verify Graph replay")
    rollout.add_argument("--config", type=Path, required=True)
    rollout.add_argument("--output-dir", type=Path, required=True, help="new directory under kt-baseline/results")
    rollout.add_argument("--max-new-tokens", type=int, choices=range(2, 65), metavar="2..64", default=32)
    args = parser.parse_args(argv)
    if args.command == "rollout":
        from .workers.rollout.sglang.smoke import run_one

        try:
            result = run_one(args.config, args.output_dir, args.max_new_tokens)
        except ValueError as error:
            rollout.error(str(error))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.dry_run:
        if args.steps != 1 or args.output_dir is None or args.profiling != 'light':
            run.error("M4 execution requires --steps 1 --profiling light --output-dir <fresh directory>")
        from .trainer.online import run as run_online
        run_online(args.config, args.output_dir)
        return
    try:
        plan = load_plan(args.config, steps=args.steps, profiling=args.profiling)
    except (ValueError, OSError) as error:
        run.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
