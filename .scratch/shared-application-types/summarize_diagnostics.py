import argparse
import re
from collections import Counter
from pathlib import Path


def counts(path: Path) -> Counter[str]:
    return Counter(
        re.findall(r"^(src/[^:\n]+):\d+: error:", path.read_text(), re.MULTILINE)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before, after = counts(args.before), counts(args.after)
    print("| Application file | Before | After |")
    print("| --- | ---: | ---: |")
    for path in sorted(before.keys() | after.keys()):
        print(
            f"| `{path.removeprefix('src/pupil_labs/neon_player/')}` | {before[path]} | {after[path]} |"  # noqa: E501
        )
    print(f"| Total | {before.total()} | {after.total()} |")


if __name__ == "__main__":
    main()
