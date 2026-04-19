"""
Tests for api/retriever.py

Uses a mock ChromaDB collection so tests run without a live database.
"""
from unittest.mock import MagicMock, patch

from api.retriever import retrieve_reviews


def test_retrieve_reviews_returns_list_of_dicts():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["Great food!", "Terrible service."],
        "metadatas": [{"stars": 5.0}, {"stars": 1.0}],
    }
    with patch("api.retriever.get_collection", return_value=mock_collection):
        results = retrieve_reviews("biz_001", n_results=2)

    assert len(results) == 2
    assert results[0] == {"text": "Great food!", "stars": 5.0}
    assert results[1] == {"text": "Terrible service.", "stars": 1.0}


def test_retrieve_reviews_returns_empty_list_when_no_results():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": [], "metadatas": []}
    with patch("api.retriever.get_collection", return_value=mock_collection):
        results = retrieve_reviews("biz_unknown")

    assert results == []


def test_retrieve_reviews_passes_correct_query_to_chromadb():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": [], "metadatas": []}
    with patch("api.retriever.get_collection", return_value=mock_collection):
        retrieve_reviews("biz_42", n_results=10)

    call_kwargs = mock_collection.get.call_args.kwargs
    assert call_kwargs["where"] == {"business_id": "biz_42"}
    assert call_kwargs["limit"] == 10
    assert call_kwargs["include"] == ["documents", "metadatas"]
