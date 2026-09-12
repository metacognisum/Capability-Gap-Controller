import argparse
import json
from pathlib import Path

from .demo import run_demo
from .storage import CapabilityMap


def main():
    parser = argparse.ArgumentParser(description="Capability Gap Controller research preview")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run offline recovery and held-out skill-adaptation examples")
    demo.add_argument("--output", type=Path, required=True, help="New output directory")
    inspect = commands.add_parser("inspect", help="Read an existing capability map")
    inspect.add_argument("database", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        result = run_demo(args.output)
    else:
        if not args.database.is_file():
            parser.error("Database does not exist")
        store = CapabilityMap(args.database)
        result = {"capabilities": store.summary(), "clusters": store.clusters(), "adaptations": store.adaptations()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
