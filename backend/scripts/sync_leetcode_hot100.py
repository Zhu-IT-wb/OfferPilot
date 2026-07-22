import argparse
import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from app.services.leetcode_catalog import HOT100_SOURCE_URL, parse_hot100_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the versioned LeetCode Hot 100 snapshot.")
    parser.add_argument("--html", type=Path, help="Read a previously downloaded LeetCode page.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "app" / "data" / "leetcode_hot100.json",
    )
    args = parser.parse_args()

    html = args.html.read_text(encoding="utf-8") if args.html else _download_hot100_page()
    snapshot = parse_hot100_html(html, source_version=date.today().isoformat())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Wrote {len(snapshot.problems)} problems to {output}")


def _download_hot100_page() -> str:
    request = Request(
        HOT100_SOURCE_URL,
        headers={"User-Agent": "OfferPilot Hot100 snapshot updater/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
