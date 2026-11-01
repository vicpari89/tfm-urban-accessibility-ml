"""Download the raw files (the restricted Zenodo record) into data/raw/, for the GitHub Actions workflow.

Needs the environment variable ZENODO_TOKEN (a Zenodo personal access token of an account with access).
Each required file (REQUIRED_FILES of pipeline.py) is matched by its file name with the files of the record.
"""

import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

from urban_accessibility_ml.pipeline import REQUIRED_FILES, ZENODO_RECORD_URL

RECORD_ID = ZENODO_RECORD_URL.rstrip("/").rsplit("/", 1)[-1]
API = f"https://zenodo.org/api/records/{RECORD_ID}"


def request(url: str, token: str):
    return urlopen(Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=120)


def main() -> None:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN is not set (repository secret).")

    with request(f"{API}/files", token) as response:
        entries = json.load(response).get("entries", [])
    keys = {Path(e["key"]).name: e["key"] for e in entries}

    missing = [f.name for f in REQUIRED_FILES if f.name not in keys]
    if missing:
        sys.exit(f"Not in the Zenodo record: {missing}. Files seen: {sorted(keys)}")

    for file in REQUIRED_FILES:
        file.parent.mkdir(parents=True, exist_ok=True)
        with request(f"{API}/files/{quote(keys[file.name])}/content", token) as response:
            file.write_bytes(response.read())
        print(f"{file.name}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
