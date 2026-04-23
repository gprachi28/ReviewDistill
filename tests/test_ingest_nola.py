"""
Tests for ingestion/ingest_nola.py

Covers: attribute parsing, business filter logic, SQLite schema creation and inserts.
Does NOT test ChromaDB embedding end-to-end (integration concern).
"""
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from ingestion.ingest_nola import (
    _parse_bool,
    _parse_dict,
    _parse_str,
    create_schema,
    ingest_businesses,
    parse_business,
)


# ── _parse_bool ────────────────────────────────────────────────────────────────


def test_parse_bool_true():
    assert _parse_bool("True") == 1


def test_parse_bool_false():
    assert _parse_bool("False") == 0


def test_parse_bool_none_input():
    assert _parse_bool(None) is None


def test_parse_bool_unknown_string():
    assert _parse_bool("maybe") is None


# ── _parse_str ─────────────────────────────────────────────────────────────────


def test_parse_str_u_prefix():
    assert _parse_str("u'loud'") == "loud"


def test_parse_str_single_quoted():
    assert _parse_str("'quiet'") == "quiet"


def test_parse_str_plain_string():
    # Already clean — pass through unchanged
    assert _parse_str("average") == "average"


def test_parse_str_none():
    assert _parse_str(None) is None


# ── _parse_dict ────────────────────────────────────────────────────────────────


def test_parse_dict_returns_json_string():
    raw = "{'romantic': True, 'casual': False}"
    result = _parse_dict(raw)
    parsed = json.loads(result)
    assert parsed["romantic"] is True
    assert parsed["casual"] is False


def test_parse_dict_none():
    assert _parse_dict(None) is None


def test_parse_dict_invalid_returns_none():
    assert _parse_dict("not a dict at all") is None


# ── parse_business ─────────────────────────────────────────────────────────────


VALID_RECORD = {
    "business_id": "abc123",
    "name": "Test Place",
    "city": "New Orleans",
    "stars": 4.0,
    "review_count": 100,
    "categories": "Restaurants, Cajun/Creole",
    "latitude": 29.95,
    "longitude": -90.07,
    "attributes": {
        "RestaurantsPriceRange2": "2",
        "NoiseLevel": "u'loud'",
        "Alcohol": "u'full_bar'",
        "RestaurantsGoodForGroups": "True",
        "OutdoorSeating": "False",
        "Ambience": "{'romantic': False, 'casual': True}",
        "GoodForMeal": "{'brunch': True, 'dinner': True}",
        "Music": "{'live': True, 'dj': False}",
        "BusinessParking": "{'street': True, 'garage': False}",
    },
}


def test_parse_business_valid():
    result = parse_business(VALID_RECORD)
    assert result is not None
    assert result["business_id"] == "abc123"
    assert result["noise_level"] == "loud"
    assert result["alcohol"] == "full_bar"
    assert result["good_for_groups"] == 1
    assert result["outdoor_seating"] == 0
    assert result["price_range"] == 2
    assert json.loads(result["ambience"])["casual"] is True
    assert json.loads(result["good_for_meal"])["brunch"] is True


def test_parse_business_wrong_city():
    record = {**VALID_RECORD, "city": "Las Vegas"}
    assert parse_business(record) is None


def test_parse_business_not_a_restaurant():
    record = {**VALID_RECORD, "categories": "Hair Salons, Beauty"}
    assert parse_business(record) is None


def test_parse_business_too_few_reviews():
    record = {**VALID_RECORD, "review_count": 50}
    assert parse_business(record) is None


def test_parse_business_missing_attributes():
    record = {**VALID_RECORD, "attributes": None}
    result = parse_business(record)
    assert result is not None
    assert result["noise_level"] is None
    assert result["price_range"] is None


# ── SQLite schema ──────────────────────────────────────────────────────────────


def test_create_schema_creates_businesses_table():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "businesses" in tables
    conn.close()


def test_create_schema_is_idempotent():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    create_schema(conn)  # should not raise
    conn.close()


def test_create_schema_creates_indexes():
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_biz_noise" in indexes
    assert "idx_biz_groups" in indexes
    assert "idx_biz_price" in indexes
    conn.close()


# ── ingest_businesses ──────────────────────────────────────────────────────────


def test_ingest_businesses_inserts_nola_rows(tmp_path):
    """Full ingest_businesses run against a fake JSONL business file."""
    biz_file = tmp_path / "business.json"
    records = [
        {
            "business_id": "nola_001",
            "name": "Bayou Spot",
            "city": "New Orleans",
            "stars": 4.5,
            "review_count": 200,
            "categories": "Restaurants, Cajun/Creole",
            "latitude": 29.95,
            "longitude": -90.07,
            "attributes": {
                "NoiseLevel": "u'loud'",
                "RestaurantsPriceRange2": "2",
                "RestaurantsGoodForGroups": "True",
                "OutdoorSeating": "True",
                "Alcohol": "u'full_bar'",
            },
        },
        # filtered out — wrong city
        {
            "business_id": "notme_001",
            "name": "Vegas Diner",
            "city": "Las Vegas",
            "stars": 3.0,
            "review_count": 300,
            "categories": "Restaurants",
            "attributes": {},
        },
    ]
    biz_file.write_text("\n".join(json.dumps(r) for r in records))

    db_path = str(tmp_path / "test.db")
    with patch("ingestion.ingest_nola.settings") as mock_settings:
        mock_settings.sqlite_path = db_path
        ids = ingest_businesses(str(biz_file))

    assert ids == {"nola_001"}
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT business_id, noise_level FROM businesses").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0] == ("nola_001", "loud")


def test_ingest_businesses_skips_when_table_populated(tmp_path):
    """If the table already has rows, ingest_businesses returns existing IDs without re-inserting."""
    biz_file = tmp_path / "business.json"
    biz_file.write_text("")  # empty — should not be read

    db_path = str(tmp_path / "test.db")
    # Pre-populate the table
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE businesses (
        business_id TEXT PRIMARY KEY, name TEXT NOT NULL, stars REAL NOT NULL,
        review_count INTEGER NOT NULL, price_range INTEGER, noise_level TEXT,
        alcohol TEXT, attire TEXT, wifi TEXT, smoking TEXT,
        good_for_groups INTEGER, takes_reservations INTEGER, outdoor_seating INTEGER,
        good_for_kids INTEGER, good_for_dancing INTEGER, happy_hour INTEGER,
        has_tv INTEGER, caters INTEGER, wheelchair_accessible INTEGER,
        dogs_allowed INTEGER, byob INTEGER, corkage INTEGER,
        ambience TEXT, good_for_meal TEXT, music TEXT, parking TEXT,
        categories TEXT NOT NULL, latitude REAL, longitude REAL
    )""")
    conn.execute("INSERT INTO businesses (business_id, name, stars, review_count, categories) "
                 "VALUES ('existing_001', 'Old Place', 4.0, 100, 'Restaurants')")
    conn.commit()
    conn.close()

    with patch("ingestion.ingest_nola.settings") as mock_settings:
        mock_settings.sqlite_path = db_path
        ids = ingest_businesses(str(biz_file))

    assert ids == {"existing_001"}
