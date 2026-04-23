"""
Tests for api/retriever.py

Mocks both the EmbeddingModel and ChromaDB collection.
Does NOT make real embedding or vector search calls.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.retriever import retrieve


def _mock_collection(docs, metas, distances):
    col = MagicMock()
    col.count.return_value = 100
    col.query.return_value = {
        "ids": [["id1", "id2", "id3"][: len(docs)]],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }
    return col


def _mock_embed(vectors=None):
    """Patch embed() to return a fixed query vector."""
    vec = vectors or [[0.1] * 256]
    return patch("api.retriever.embed", return_value=vec)


def _mock_model():
    return patch("api.retriever._get_model", return_value=MagicMock())


def _mock_coll(col):
    return patch("api.retriever._get_collection", return_value=col)


# ── happy path ─────────────────────────────────────────────────────────────────


def test_retrieve_returns_snippets():
    docs = ["Great jazz brunch!", "Amazing atmosphere."]
    metas = [
        {"business_id": "biz_a", "stars": 5.0, "date": "2023-01-01"},
        {"business_id": "biz_b", "stars": 4.0, "date": "2023-02-01"},
    ]
    dists = [0.1, 0.3]
    col = _mock_collection(docs, metas, dists)

    with _mock_model(), _mock_coll(col), _mock_embed():
        result = retrieve("jazz brunch spot")

    assert len(result) == 2
    assert result[0]["business_id"] == "biz_a"
    assert result[0]["text"] == "Great jazz brunch!"
    assert result[0]["stars"] == 5.0
    assert result[0]["distance"] == 0.1


def test_retrieve_with_business_ids_passes_where_filter():
    col = _mock_collection(["review text"], [{"business_id": "biz_a", "stars": 4.0, "date": "2023-01-01"}], [0.2])

    with _mock_model(), _mock_coll(col), _mock_embed():
        retrieve("brunch spot", business_ids=["biz_a", "biz_b"])

    call_kwargs = col.query.call_args[1]
    assert "where" in call_kwargs
    assert call_kwargs["where"] == {"business_id": {"$in": ["biz_a", "biz_b"]}}


def test_retrieve_without_business_ids_omits_where_filter():
    col = _mock_collection(["review"], [{"business_id": "biz_a", "stars": 4.0, "date": "2023-01-01"}], [0.2])

    with _mock_model(), _mock_coll(col), _mock_embed():
        retrieve("late night spot", business_ids=None)

    call_kwargs = col.query.call_args[1]
    assert "where" not in call_kwargs


def test_retrieve_uses_query_prefix_in_embedding():
    col = _mock_collection([], [], [])
    col.count.return_value = 0
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    with _mock_model(), _mock_coll(col):
        with patch("api.retriever.embed", return_value=[[0.1] * 256]) as mock_embed:
            retrieve("jazz brunch")

    embedded_text = mock_embed.call_args[0][1][0]
    assert embedded_text.startswith("search_query: ")
    assert "jazz brunch" in embedded_text


def test_retrieve_respects_top_k():
    col = _mock_collection(["r1", "r2"], [
        {"business_id": "a", "stars": 4.0, "date": "2023-01-01"},
        {"business_id": "b", "stars": 3.0, "date": "2023-01-02"},
    ], [0.1, 0.2])
    col.count.return_value = 100

    with _mock_model(), _mock_coll(col), _mock_embed():
        retrieve("query", top_k=5)

    call_kwargs = col.query.call_args[1]
    assert call_kwargs["n_results"] == 5


def test_retrieve_caps_n_results_to_collection_size():
    col = _mock_collection(["r1"], [{"business_id": "a", "stars": 4.0, "date": "2023-01-01"}], [0.1])
    col.count.return_value = 3  # collection smaller than top_k

    with _mock_model(), _mock_coll(col), _mock_embed():
        retrieve("query", top_k=20)

    call_kwargs = col.query.call_args[1]
    assert call_kwargs["n_results"] == 3


def test_retrieve_empty_collection_returns_empty_list():
    col = MagicMock()
    col.count.return_value = 0
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    with _mock_model(), _mock_coll(col), _mock_embed():
        result = retrieve("any query")

    assert result == []
