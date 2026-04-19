"""
Tests for ingestion/ingest.py

Covers: checkpoint round-trip, stream_reviews line parsing, embed truncation + renormalisation.
Does NOT test ingest() end-to-end — that requires a live ChromaDB + model (integration concern).
"""
import json
import pickle
from unittest.mock import MagicMock

import numpy as np

from config import settings
from ingestion.ingest import embed, load_checkpoint, save_checkpoint, stream_reviews


# ── checkpoint ────────────────────────────────────────────────────────────────


def test_load_checkpoint_returns_zero_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("ingestion.ingest.CHECKPOINT_FILE", tmp_path / "checkpoint.pkl")
    assert load_checkpoint() == 0


def test_load_checkpoint_returns_saved_value(tmp_path, monkeypatch):
    cp = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr("ingestion.ingest.CHECKPOINT_FILE", cp)
    save_checkpoint(42)
    assert load_checkpoint() == 42


def test_save_checkpoint_writes_to_disk(tmp_path, monkeypatch):
    cp = tmp_path / "checkpoint.pkl"
    monkeypatch.setattr("ingestion.ingest.CHECKPOINT_FILE", cp)
    save_checkpoint(99)
    assert pickle.loads(cp.read_bytes()) == 99


# ── stream_reviews ─────────────────────────────────────────────────────────────


def test_stream_reviews_yields_parsed_dicts(tmp_path):
    reviews = [
        {"review_id": "r1", "business_id": "b1", "stars": 5, "text": "Great"},
        {"review_id": "r2", "business_id": "b1", "stars": 1, "text": "Bad"},
    ]
    review_file = tmp_path / "reviews.json"
    review_file.write_text("\n".join(json.dumps(r) for r in reviews))

    results = list(stream_reviews(str(review_file)))

    assert len(results) == 2
    assert results[0]["review_id"] == "r1"
    assert results[1]["stars"] == 1


def test_stream_reviews_skips_blank_lines(tmp_path):
    review_file = tmp_path / "reviews.json"
    review_file.write_text('{"review_id": "r1", "stars": 5, "text": "ok"}\n\n')

    results = list(stream_reviews(str(review_file)))

    assert len(results) == 1


# ── embed ──────────────────────────────────────────────────────────────────────


def test_embed_truncates_to_configured_dimensions():
    model = MagicMock()
    rng = np.random.default_rng(42)
    model.encode.return_value = rng.standard_normal((2, 768)).astype(np.float32)

    result = embed(model, ["text1", "text2"])

    assert len(result) == 2
    assert len(result[0]) == settings.embed_dimensions


def test_embed_returns_unit_vectors():
    model = MagicMock()
    rng = np.random.default_rng(42)
    model.encode.return_value = rng.standard_normal((3, 768)).astype(np.float32)

    result = embed(model, ["a", "b", "c"])

    for vec in result:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_handles_zero_vector_without_dividing_by_zero():
    model = MagicMock()
    model.encode.return_value = np.zeros((1, 768), dtype=np.float32)

    result = embed(model, ["empty"])

    assert len(result[0]) == settings.embed_dimensions
    assert all(v == 0.0 for v in result[0])
