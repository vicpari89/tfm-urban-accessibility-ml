"""Download the raw files (the restricted Zenodo record) into data/raw/, for the GitHub Actions workflow.

Needs the environment variable ZENODO_TOKEN (a Zenodo personal access token of an account with access).
Each required file (REQUIRED_FILES of pipeline.py) is matched by its file name with the files of the record,
including the ones inside a zip.
"""

import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

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

    # file name -> content, from the loose files and from inside the zips of the record (the record has the zip and 2 loose files)
    contents: dict[str, bytes] = {}
    for entry in entries:
        with request(f"{API}/files/{quote(entry['key'])}/content", token) as response:
            data = response.read()
        if entry["key"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                contents.update({Path(n).name: zf.read(n) for n in zf.namelist() if not n.endswith("/")})
        else:
            contents[Path(entry["key"]).name] = data

    missing = [f.name for f in REQUIRED_FILES if f.name not in contents]
    if missing:
        sys.exit(f"Not in the Zenodo record: {missing}. Files seen: {sorted(contents)}")

    for file in REQUIRED_FILES:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(contents[file.name])
        print(f"{file.name}: {file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
