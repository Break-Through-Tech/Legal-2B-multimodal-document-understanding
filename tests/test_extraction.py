"""Tests for extraction/: the prompt, reply parsing, the output files, and the committed results.

No GPU or model needed.
"""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analysis.channels import RESULTS_DIR
from analysis.detection import REVIEW
from extraction.outputs import REMOVED_IDS, REMOVED_NOTE, check_settings, done_ids, remove_personal_data
from extraction.prompt import build_prompt, load_schema, parse_reply, shrink_to

LEGAL_TYPES = ["COMMERCIAL_LEASE_AGREEMENT", "PATENT", "PETITION_FORM", "PROXY_VOTING", "REAL_ESTATE"]


@pytest.mark.parametrize("doc_type", LEGAL_TYPES)
def test_every_legal_type_has_a_schema(doc_type):
    assert load_schema(doc_type)["type"] == "object"


def test_prompt_names_the_type_and_includes_the_whole_schema():
    schema = load_schema("PATENT")
    prompt = build_prompt("PATENT", schema)
    assert "patent document" in prompt
    assert json.dumps(schema) in prompt


@pytest.mark.parametrize("reply", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Here is the JSON:\n```\n{"a": 1}\n```\nDone.',
    'Sure! {"a": 1} Hope this helps.',
])
def test_parse_reply_finds_the_json(reply):
    assert parse_reply(reply) == {"a": 1}


@pytest.mark.parametrize("reply", ["", "no json here", '{"a": 1', "[1, 2]", '{"a": 1} and {"b": 2}'])
def test_parse_reply_returns_none_without_one_valid_object(reply):
    assert parse_reply(reply) is None


def test_shrink_keeps_the_shape_and_caps_the_pixels():
    small = shrink_to(Image.new("L", (100, 200)), max_pixels=50_000)
    assert small.size == (100, 200) and small.mode == "RGB"  # already small enough, only converted
    big = shrink_to(Image.new("RGB", (2000, 4000)), max_pixels=50_000)
    assert big.width * big.height <= 50_000
    assert abs(big.height / big.width - 2) < 0.05


# Resuming a run (extraction/outputs.py)

def test_done_ids_reads_the_ids_and_skips_blank_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    assert done_ids(path) == set()
    path.write_text('{"id": 1}\n\n{"id": 2}\n')
    assert done_ids(path) == {1, 2}


def test_done_ids_stops_on_a_half_written_line(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": 1}\n{"id": 2, "raw": "cut o')
    with pytest.raises(SystemExit, match="Line 2"):
        done_ids(path)


def test_check_settings(tmp_path):
    out_path, run_path = tmp_path / "out.jsonl", tmp_path / "out_run.json"
    settings = {"model": "m", "max_pixels": 100}
    check_settings(out_path, run_path, settings)  # nothing there yet
    run_path.write_text(json.dumps({**settings, "started": "today"}))
    out_path.write_text('{"id": 1}\n')
    check_settings(out_path, run_path, settings)  # same settings: fine
    with pytest.raises(SystemExit, match="max_pixels"):
        check_settings(out_path, run_path, {**settings, "max_pixels": 200})
    run_path.unlink()
    with pytest.raises(SystemExit, match="settings are unknown"):
        check_settings(out_path, run_path, settings)


def test_remove_personal_data_blanks_only_the_listed_pages():
    reply = {"predicted_type": "PETITION_FORM", "output": {"name": "x"}, "raw": '{"name": "x"}', "error": None}
    removed = remove_personal_data({"id": next(iter(REMOVED_IDS)), **reply})
    assert removed["output"] is None and removed["raw"] is None and removed["error"] == REMOVED_NOTE
    kept = {"id": -1, **reply}
    assert remove_personal_data(kept) == kept


# The committed results (results/extraction/)

KEYS = ["id", "predicted_type", "output", "raw", "input_tokens", "output_tokens", "hit_token_limit", "seconds", "error"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_replies(split):
    return read_jsonl(RESULTS_DIR / "extraction" / f"qwen3-vl-4b-instruct_{split}.jsonl")


def test_no_committed_file_keeps_a_removed_reply():
    """Every model's output file, not just this one, since the repo is public."""
    paths = list((RESULTS_DIR / "extraction").glob("*.jsonl"))
    assert paths
    for path in paths:
        for r in read_jsonl(path):
            if r["id"] in REMOVED_IDS:
                assert r["output"] is None and r["raw"] is None, f"{path.name} keeps the reply for {r['id']}"


def test_routing_matches_the_detection_results():
    routes = pd.read_csv(RESULTS_DIR / "extraction" / "routing.csv").set_index("id")
    scores = pd.read_csv(RESULTS_DIR / "detection_scores.csv")
    scores = scores[scores["channel"] == "vision_colqwen2"]
    either = scores[scores["detector"] == "either"].set_index("id").loc[routes.index]
    assert (routes["predicted_type"] == either["predicted_type"]).all()
    assert (routes["route"] == np.where(either["flagged"], REVIEW, either["predicted_type"])).all()
    for detector in ["confidence", "distance"]:
        expected = scores[scores["detector"] == detector].set_index("id").loc[routes.index, "score"]
        assert np.allclose(routes[f"{detector}_score"], expected)


@pytest.mark.parametrize("split", ["val", "test"])
def test_replies_cover_exactly_the_routed_documents(split):
    routes = pd.read_csv(RESULTS_DIR / "extraction" / "routing.csv")
    routed = routes[(routes["split"] == split) & (routes["route"] != REVIEW)].set_index("id")["route"]
    replies = load_replies(split)
    ids = [r["id"] for r in replies]
    assert len(ids) == len(set(ids)), "an id appears twice"
    assert sorted(ids) == sorted(routed.index)  # order can change when a page is retried
    for r in replies:
        assert list(r) == KEYS
        assert r["predicted_type"] == routed[r["id"]]
        if r["id"] in REMOVED_IDS:
            assert r["output"] is None and r["raw"] is None and r["error"] == REMOVED_NOTE
        else:
            assert r["output"] == parse_reply(r["raw"]) and r["error"] is None
