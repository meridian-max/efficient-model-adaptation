import argparse
import json
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Compare frozen, full and low-rank adaptation on CPU")
    parser.add_argument("--output", default="runs/benchmark.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 31])
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    result = run_experiment(args.output, seeds=args.seeds, rank=args.rank, steps=args.steps)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
