"""
Tests for api/sql_filter.py

Uses an in-memory SQLite database seeded with fixture businesses.
"""
import json
import sqlite3

import pytest

from api.sql_filter import filter_businesses


# ── fixtures ───────────────────────────────────────────────────────────────────


def _create_db(tmp_path) -> str:
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE businesses (
            business_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            stars REAL NOT NULL,
            review_count INTEGER NOT NULL,
            price_range INTEGER,
            noise_level TEXT,
            alcohol TEXT,
            attire TEXT,
            wifi TEXT,
            smoking TEXT,
            good_for_groups INTEGER,
            takes_reservations INTEGER,
            outdoor_seating INTEGER,
            good_for_kids INTEGER,
            good_for_dancing INTEGER,
            happy_hour INTEGER,
            has_tv INTEGER,
            caters INTEGER,
            wheelchair_accessible INTEGER,
            dogs_allowed INTEGER,
            byob INTEGER,
            corkage INTEGER,
            ambience TEXT,
            good_for_meal TEXT,
            music TEXT,
            parking TEXT,
            categories TEXT NOT NULL,
            latitude REAL,
            longitude REAL
        );
    """)

    # Helper: build JSON attrs as strings
    def j(d): return json.dumps(d)

    businesses = [
        # fmt: off
        # id, name, stars, rev, price, noise, alcohol, attire, wifi, smoking,
        # groups, reserv, outdoor, kids, dancing, happy, tv, caters, wheelchair,
        # dogs, byob, corkage,
        # ambience, good_for_meal, music, parking, categories, lat, lon

        # biz_a: loud, full_bar, groups, brunch, live, price=2, stars=4.5, no dancing
        ("biz_a", "Bayou Jazz", 4.5, 200, 2, "loud", "full_bar", "casual", None, None,
         1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
         j({"casual": True}), j({"brunch": True, "dinner": True}),
         j({"live": True}), j({"street": True}),
         "Restaurants,Cajun", 29.95, -90.07),

        # biz_b: quiet, beer_and_wine, no groups, no brunch, price=3, stars=4.2, no dancing
        ("biz_b", "The Quiet Creole", 4.2, 150, 3, "quiet", "beer_and_wine", "dressy", None, None,
         0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
         j({"romantic": True}), j({"dinner": True}),
         j({"background_music": True}), j({"garage": True}),
         "Restaurants,Seafood", 29.96, -90.05),

        # biz_c: loud, full_bar, groups, brunch, live, price=2, stars=4.0, dancing=True
        ("biz_c", "Frenchmen Spot", 4.0, 120, 2, "loud", "full_bar", "casual", None, None,
         1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0,
         j({"hipster": True}), j({"brunch": True, "latenight": True}),
         j({"live": True, "dj": True}), j({"street": True}),
         "Restaurants,Bars", 29.96, -90.06),

        # biz_d: average, full_bar, groups, no brunch, price=1, stars=3.8, no dancing
        ("biz_d", "Cheap Eats NOLA", 3.8, 80, 1, "average", "full_bar", "casual", None, None,
         1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
         j({"casual": True}), j({"lunch": True, "dinner": True}),
         j({"background_music": True}), j({"lot": True}),
         "Restaurants,American", 29.97, -90.08),

        # biz_e: loud, full_bar, groups, brunch, live, price=2, stars=4.3, no dancing
        # — adds the 3rd match for loud / brunch / live / stars>=4.2 filters
        ("biz_e", "Jazz Corner", 4.3, 180, 2, "loud", "full_bar", "casual", None, None,
         1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
         j({"trendy": True}), j({"brunch": True, "dinner": True}),
         j({"live": True}), j({"street": True}),
         "Restaurants,Cajun", 29.94, -90.07),

        # biz_f: quiet, no alcohol, no groups, price=3, stars=3.7, no dancing
        ("biz_f", "Solo Diner", 3.7, 60, 3, "quiet", "none", "dressy", None, None,
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         j({"intimate": True}), j({"dinner": True}),
         j({"background_music": True}), j({"valet": True}),
         "Restaurants,French", 29.93, -90.04),

        # biz_g: average, no alcohol, no groups, price=2, stars=3.5, no dancing
        ("biz_g", "Corner Cafe", 3.5, 55, 2, "average", "none", "casual", None, None,
         0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,
         j({"casual": True}), j({"lunch": True, "breakfast": True}),
         j({"background_music": True}), j({"street": True}),
         "Restaurants,Breakfast", 29.92, -90.09),
        # fmt: on
    ]

    conn.executemany(
        "INSERT INTO businesses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        businesses,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def db(tmp_path):
    return _create_db(tmp_path)


# ── basic filters ──────────────────────────────────────────────────────────────


def test_scalar_filter(db):
    result = filter_businesses({"noise_level": "loud"}, db_path=db)
    # biz_a, biz_c, biz_e are all loud (3 results — no fallback)
    assert set(result) == {"biz_a", "biz_c", "biz_e"}


def test_scalar_list_filter(db):
    result = filter_businesses({"noise_level": ["loud", "average"]}, db_path=db)
    assert set(result) == {"biz_a", "biz_c", "biz_d", "biz_e", "biz_g"}


def test_boolean_true_filter(db):
    result = filter_businesses({"good_for_groups": True}, db_path=db)
    # biz_a, biz_c, biz_d, biz_e
    assert set(result) == {"biz_a", "biz_c", "biz_d", "biz_e"}


def test_boolean_false_filter(db):
    result = filter_businesses({"good_for_groups": False}, db_path=db)
    # biz_b, biz_f, biz_g (3 results — no fallback)
    assert set(result) == {"biz_b", "biz_f", "biz_g"}


def test_range_lte_filter(db):
    result = filter_businesses({"price_range": {"lte": 2}}, db_path=db)
    assert set(result) == {"biz_a", "biz_c", "biz_d", "biz_e", "biz_g"}


def test_range_gte_filter(db):
    result = filter_businesses({"stars": {"gte": 4.2}}, db_path=db)
    # biz_a=4.5, biz_b=4.2, biz_e=4.3
    assert set(result) == {"biz_a", "biz_b", "biz_e"}


def test_json_subkey_filter(db):
    result = filter_businesses({"good_for_meal": {"brunch": True}}, db_path=db)
    # biz_a, biz_c, biz_e
    assert set(result) == {"biz_a", "biz_c", "biz_e"}


def test_json_subkey_music_live(db):
    result = filter_businesses({"music": {"live": True}}, db_path=db)
    # biz_a, biz_c, biz_e
    assert set(result) == {"biz_a", "biz_c", "biz_e"}


def test_combined_filters(db):
    # noise_level=loud AND good_for_groups=True → biz_a, biz_c, biz_e (3) — no fallback
    result = filter_businesses(
        {"noise_level": "loud", "good_for_groups": True},
        db_path=db,
    )
    assert set(result) == {"biz_a", "biz_c", "biz_e"}


# ── empty / no-op cases ────────────────────────────────────────────────────────


def test_empty_filters_returns_none(db):
    assert filter_businesses({}, db_path=db) is None


def test_null_scalar_value_triggers_fallback(db):
    # noise_level = NULL never matches (SQL = NULL != IS NULL) → sparse fallback → None
    result = filter_businesses({"noise_level": None}, db_path=db)
    assert result is None


# ── sparse fallback ────────────────────────────────────────────────────────────


def test_sparse_fallback_relaxes_lowest_confidence(db):
    # good_for_dancing=True (confidence=2) + good_for_groups=True (confidence=2)
    # → only biz_c has dancing=True AND groups=True (1 result < 3)
    # Drop first constraint (good_for_dancing): good_for_groups=True → biz_a, biz_c, biz_d, biz_e (4)
    result = filter_businesses(
        {"good_for_dancing": True, "good_for_groups": True},
        db_path=db,
    )
    assert result is not None
    assert len(result) >= 3
    # good_for_groups=True constraint is still applied
    assert "biz_b" not in result
    assert "biz_f" not in result


def test_sparse_fallback_drops_json_subkey_before_boolean(db):
    # JSON sub-key (confidence=1) dropped before boolean (confidence=2)
    # good_for_meal.brunch=True AND good_for_dancing=True → only biz_c (1 result)
    # Drop brunch (confidence=1): good_for_dancing=True → only biz_c (still 1 result < 3)
    # Can't relax further (single condition left) → None
    result = filter_businesses(
        {"good_for_meal": {"brunch": True}, "good_for_dancing": True},
        db_path=db,
    )
    assert result is None


def test_sparse_fallback_returns_none_when_still_too_few(db):
    # alcohol=beer_and_wine → only biz_b (1 result, single constraint)
    # Can't relax → None
    result = filter_businesses({"alcohol": "beer_and_wine"}, db_path=db)
    assert result is None


def test_no_fallback_when_enough_results(db):
    # full_bar → biz_a, biz_c, biz_d, biz_e (4 results ≥ 3)
    result = filter_businesses({"alcohol": "full_bar"}, db_path=db)
    assert result is not None
    assert len(result) == 4
