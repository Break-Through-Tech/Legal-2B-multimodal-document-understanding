"""Step 2's output files: what is already done, the settings check, and removing personal data (issue #10).

No torch here, so it can be tested on any machine (tests/test_extraction.py).
"""

import json
from pathlib import Path

# Pages that show real people. The model still reads them, but their reply is not saved,
# because the results are committed to a public repo.
REMOVED_IDS = {36}  # a phone photo of a real 2023 petition: signers' names, addresses and birth dates
REMOVED_NOTE = "reply removed: it held the names and addresses of real people (see extraction/README.md)"


def remove_personal_data(result: dict) -> dict:
    """The result with its reply blanked if the page is in REMOVED_IDS. Other results are unchanged."""
    if result["id"] in REMOVED_IDS:
        return {**result, "output": None, "raw": None, "error": REMOVED_NOTE}
    return result


def done_ids(path: Path) -> set[int]:
    """Ids already in the output file, so a rerun can skip them."""
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                # Most likely the run was killed while writing this line.
                raise SystemExit(f"Line {n} of {path.name} is not a full record. Delete that line and run again.")
    return ids


def check_settings(out_path: Path, run_path: Path, settings: dict) -> None:
    """Stop if the output file was started with a different model or settings.

    It compares the values in `settings` only. It does not check package versions or the
    schema files, so after changing those, move the old files away and start over.
    """
    if out_path.exists() and not run_path.exists():
        raise SystemExit(f"{out_path.name} exists but {run_path.name} doesn't, so its settings are unknown. "
                         "Move it away to start over.")
    if not run_path.exists():
        return
    old = json.loads(run_path.read_text())
    changed = sorted(k for k in settings if old.get(k) != settings[k])
    if changed:
        raise SystemExit(f"{run_path.name} was made with a different {', '.join(changed)}. "
                         "Move the old output files away to start over.")
