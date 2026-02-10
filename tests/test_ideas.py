"""Tests for the project ideas module."""

import json
import os
import sqlite3
import tempfile

from click.testing import CliRunner

from hn_intel.db import init_db, upsert_blogs, insert_post
from hn_intel.ideas import (
    extract_pain_signals,
    extract_signal_keywords,
    score_ideas,
    build_justification,
    cluster_signals,
    generate_ideas,
)
from hn_intel.reports import generate_ideas_report
from hn_intel.cli import main


def _mem_db():
    """Create an in-memory SQLite database with schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def _seed_blogs(conn, blogs=None):
    """Insert default test blogs and return {name: id} mapping."""
    if blogs is None:
        blogs = [
            {"name": "Alpha Blog", "feed_url": "https://alpha.com/feed", "site_url": "https://alpha.com"},
            {"name": "Beta Blog", "feed_url": "https://beta.com/feed", "site_url": "https://beta.com"},
            {"name": "Gamma Blog", "feed_url": "https://gamma.com/feed", "site_url": "https://gamma.com"},
        ]
    upsert_blogs(conn, blogs)
    rows = conn.execute("SELECT id, name FROM blogs ORDER BY id").fetchall()
    return {row["name"]: row["id"] for row in rows}


def _seed_pain_posts(conn):
    """Insert posts containing various pain-point language.

    Returns {blog_name: blog_id} mapping.
    """
    ids = _seed_blogs(conn)

    posts = [
        # Wish signals
        (ids["Alpha Blog"], {
            "title": "Thoughts on developer tooling",
            "description": "<p>I wish someone would build a better debugger for distributed systems. "
                           "The current tools are inadequate.</p>",
            "url": "https://alpha.com/wish-post",
            "published": "2024-03-15",
            "author": "Alice",
        }),
        # Frustration signals
        (ids["Beta Blog"], {
            "title": "My experience with CI pipelines",
            "description": "<p>It is incredibly frustrating to deal with flaky CI tests. "
                           "Every team I work with faces this pain point daily.</p>",
            "url": "https://beta.com/frustration-post",
            "published": "2024-04-10",
            "author": "Bob",
        }),
        # Gap signals
        (ids["Gamma Blog"], {
            "title": "The state of observability",
            "description": "<p>There is still no good way to correlate logs across microservices "
                           "without setting up expensive infrastructure.</p>",
            "url": "https://gamma.com/gap-post",
            "published": "2024-04-20",
            "author": "Carol",
        }),
        # Difficulty signals
        (ids["Alpha Blog"], {
            "title": "Database migrations in production",
            "description": "<p>It is hard to run zero-downtime database migrations. "
                           "Most tools assume you can take the database offline.</p>",
            "url": "https://alpha.com/difficulty-post",
            "published": "2024-05-01",
            "author": "Alice",
        }),
        # Opportunity signals
        (ids["Beta Blog"], {
            "title": "The future of edge computing",
            "description": "<p>There is a huge opportunity in lightweight edge runtimes. "
                           "WebAssembly opens up new possibilities.</p>",
            "url": "https://beta.com/opportunity-post",
            "published": "2024-05-10",
            "author": "Bob",
        }),
        # Broken signals
        (ids["Gamma Blog"], {
            "title": "DNS resolution woes",
            "description": "<p>DNS caching in containers is broken in many orchestrators. "
                           "Pods frequently fail to resolve internal services.</p>",
            "url": "https://gamma.com/broken-post",
            "published": "2024-05-15",
            "author": "Carol",
        }),
        # Multiple signals in one post (overlap: wish + gap)
        (ids["Alpha Blog"], {
            "title": "Log analysis tools",
            "description": "<p>I wish there was a lightweight log analysis tool. "
                           "There is no good way to search logs without deploying Elasticsearch.</p>",
            "url": "https://alpha.com/multi-signal",
            "published": "2024-05-20",
            "author": "Alice",
        }),
    ]

    for blog_id, entry in posts:
        insert_post(conn, blog_id, entry)

    return ids


def _seed_neutral_posts(conn):
    """Insert posts with no pain-point language."""
    ids = _seed_blogs(conn)
    posts = [
        (ids["Alpha Blog"], {
            "title": "Introduction to Python",
            "description": "<p>Python is a versatile programming language used in many domains.</p>",
            "url": "https://alpha.com/neutral-1",
            "published": "2024-01-10",
            "author": "Alice",
        }),
        (ids["Beta Blog"], {
            "title": "Rust performance benchmarks",
            "description": "<p>We measured Rust against Go in several compute workloads.</p>",
            "url": "https://beta.com/neutral-2",
            "published": "2024-02-15",
            "author": "Bob",
        }),
    ]
    for blog_id, entry in posts:
        insert_post(conn, blog_id, entry)
    return ids


# ── extract_pain_signals tests ──


def test_extract_pain_signals_empty_db():
    conn = _mem_db()
    signals = extract_pain_signals(conn)
    assert signals == []
    conn.close()


def test_extract_pain_signals_no_signals():
    conn = _mem_db()
    _seed_neutral_posts(conn)
    signals = extract_pain_signals(conn)
    assert signals == []
    conn.close()


def test_extract_pain_signals_finds_wishes():
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)

    wish_signals = [s for s in signals if s["signal_type"] == "wish"]
    assert len(wish_signals) >= 1

    # Verify back-pointer data
    for s in wish_signals:
        assert s["blog_name"] in ("Alpha Blog", "Beta Blog", "Gamma Blog")
        assert s["post_url"].startswith("https://")
        assert s["post_title"] != ""
        assert s["published"] != ""
    conn.close()


def test_extract_pain_signals_finds_frustrations():
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)

    frust_signals = [s for s in signals if s["signal_type"] == "frustration"]
    assert len(frust_signals) >= 1
    assert any("Beta Blog" == s["blog_name"] for s in frust_signals)
    conn.close()


def test_extract_pain_signals_finds_gaps():
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)

    gap_signals = [s for s in signals if s["signal_type"] == "gap"]
    assert len(gap_signals) >= 1
    conn.close()


def test_extract_pain_signals_back_pointers():
    """Every signal must carry full provenance to source blog/post."""
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)

    assert len(signals) > 0
    required_keys = {
        "post_id", "blog_id", "blog_name", "post_title",
        "post_url", "published", "signal_text", "signal_type",
    }
    for s in signals:
        assert required_keys.issubset(s.keys()), f"Missing keys: {required_keys - s.keys()}"
        assert isinstance(s["post_id"], int)
        assert isinstance(s["blog_id"], int)
        assert len(s["signal_text"]) >= 10
    conn.close()


# ── extract_signal_keywords tests ──


def test_extract_signal_keywords_too_few():
    signals = [{"signal_text": "only one signal here"}]
    vectorizer, matrix = extract_signal_keywords(signals)
    assert vectorizer is None
    assert matrix is None


def test_extract_signal_keywords_returns_matrix():
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)
    assert len(signals) >= 2

    vectorizer, matrix = extract_signal_keywords(signals)
    assert vectorizer is not None
    assert matrix is not None
    assert matrix.shape[0] == len(signals)
    conn.close()


# ── score_ideas tests ──


def test_score_ideas_empty():
    result = score_ideas([], [], {})
    assert result == []


def test_score_ideas_with_emerging():
    signals = [
        {
            "blog_name": "Alpha Blog",
            "signal_text": "machine learning deployment is frustrating",
            "published": "2024-05-01",
        },
        {
            "blog_name": "Beta Blog",
            "signal_text": "no good way to handle payments",
            "published": "2024-04-01",
        },
    ]
    emerging = [
        {"keyword": "machine learning", "acceleration": 5.0,
         "recent_score": 0.1, "historical_avg": 0.02},
    ]
    centrality = {}

    scored = score_ideas(signals, emerging, centrality)
    assert len(scored) == 2

    # The ML signal should score higher on the trend dimension
    ml_sig = scored[0]
    pay_sig = scored[1]
    assert ml_sig["score_breakdown"]["trend"] > pay_sig["score_breakdown"]["trend"]
    assert "impact_score" in ml_sig


def test_score_ideas_with_centrality():
    signals = [
        {
            "blog_name": "Famous Blog",
            "signal_text": "I wish there was a better tool",
            "published": "2024-05-01",
        },
        {
            "blog_name": "Unknown Blog",
            "signal_text": "I wish there was a better tool",
            "published": "2024-05-01",
        },
    ]
    centrality = {
        "Famous Blog": {"pagerank": 0.9, "betweenness": 0.5, "in_degree": 10, "out_degree": 5},
        "Unknown Blog": {"pagerank": 0.01, "betweenness": 0.0, "in_degree": 0, "out_degree": 1},
    }

    scored = score_ideas(signals, [], centrality)
    famous = scored[0]
    unknown = scored[1]
    assert famous["score_breakdown"]["authority"] > unknown["score_breakdown"]["authority"]


# ── build_justification tests ──


def test_justification_content():
    idea = {
        "blog_count": 5,
        "signal_count": 8,
        "impact_score": 0.75,
        "pain_type_breakdown": {"gap": 4, "difficulty": 3, "wish": 1},
        "keywords": ["observability", "logging", "monitoring"],
        "sources": [
            {"blog_name": "Alpha Blog", "post_url": "https://alpha.com/1",
             "post_title": "Test", "published": "2024-05-01",
             "signal_text": "test signal", "signal_type": "gap"},
            {"blog_name": "Beta Blog", "post_url": "https://beta.com/1",
             "post_title": "Test2", "published": "2024-04-01",
             "signal_text": "test signal 2", "signal_type": "difficulty"},
            {"blog_name": "Gamma Blog", "post_url": "https://gamma.com/1",
             "post_title": "Test3", "published": "2024-03-01",
             "signal_text": "test signal 3", "signal_type": "wish"},
        ],
    }
    justification = build_justification(idea)

    # Should mention blog names
    assert "Alpha Blog" in justification
    # Should mention pain type breakdown
    assert "gap" in justification
    # Should mention keywords
    assert "observability" in justification
    # Should mention blog count
    assert "5 blogs" in justification
    # Should have an impact statement
    assert "impact" in justification.lower()


def test_justification_single_blog():
    idea = {
        "blog_count": 1,
        "signal_count": 1,
        "impact_score": 0.3,
        "pain_type_breakdown": {"wish": 1},
        "keywords": [],
        "sources": [
            {"blog_name": "Solo Blog", "post_url": "https://solo.com/1",
             "post_title": "One", "published": "2024-05-01",
             "signal_text": "I wish", "signal_type": "wish"},
        ],
    }
    justification = build_justification(idea)
    assert "Solo Blog" in justification


# ── cluster_signals tests ──


def test_cluster_signals_empty():
    result = cluster_signals([], None, None)
    assert result == []


def test_cluster_signals_single():
    signals = [{
        "post_id": 1, "blog_id": 1, "blog_name": "A Blog",
        "post_title": "T", "post_url": "https://a.com/1",
        "published": "2024-05-01", "signal_text": "I wish there was a tool",
        "signal_type": "wish", "impact_score": 0.5,
        "score_breakdown": {"trend": 0.2, "authority": 0.3, "breadth": 0.0, "recency": 0.4},
    }]
    result = cluster_signals(signals, None, None)
    assert len(result) == 1
    assert result[0]["signal_count"] == 1
    assert result[0]["blog_count"] == 1
    assert len(result[0]["sources"]) == 1


def test_cluster_signals_groups_similar():
    """Similar pain signals should cluster together."""
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)
    assert len(signals) >= 2

    vectorizer, matrix = extract_signal_keywords(signals)
    scored = score_ideas(signals, [], {})
    ideas = cluster_signals(scored, vectorizer, matrix)

    assert len(ideas) >= 1
    for idea in ideas:
        assert "idea_id" in idea
        assert "label" in idea
        assert "sources" in idea
        assert "justification" in idea
        assert idea["signal_count"] == len(idea["sources"])
        assert idea["blog_count"] >= 1
    conn.close()


def test_cluster_signals_sources_have_back_pointers():
    """Each source in a clustered idea must have full provenance."""
    conn = _mem_db()
    _seed_pain_posts(conn)
    signals = extract_pain_signals(conn)
    vectorizer, matrix = extract_signal_keywords(signals)
    scored = score_ideas(signals, [], {})
    ideas = cluster_signals(scored, vectorizer, matrix)

    for idea in ideas:
        for src in idea["sources"]:
            assert "blog_name" in src
            assert "post_title" in src
            assert "post_url" in src
            assert "published" in src
            assert "signal_text" in src
            assert "signal_type" in src
    conn.close()


# ── generate_ideas (end-to-end) tests ──


def test_generate_ideas_empty_db():
    conn = _mem_db()
    result = generate_ideas(conn)
    assert result == []
    conn.close()


def test_generate_ideas_end_to_end():
    conn = _mem_db()
    _seed_pain_posts(conn)
    ideas = generate_ideas(conn, top_n=10)

    assert len(ideas) >= 1
    # Ideas should be sorted by impact_score descending
    for i in range(len(ideas) - 1):
        assert ideas[i]["impact_score"] >= ideas[i + 1]["impact_score"]

    # Each idea should have full structure
    for idea in ideas:
        assert idea["label"] != ""
        assert idea["signal_count"] >= 1
        assert idea["blog_count"] >= 1
        assert len(idea["sources"]) >= 1
        assert idea["justification"] != ""
        assert idea["representative_quote"] != ""
    conn.close()


# ── Report generation tests ──


def test_ideas_report_creates_files():
    ideas = [{
        "idea_id": 0,
        "label": "testing tools",
        "impact_score": 0.8,
        "justification": "Multiple blogs raise this pain.",
        "keywords": ["testing", "ci"],
        "signal_count": 3,
        "blog_count": 2,
        "pain_type_breakdown": {"frustration": 2, "gap": 1},
        "representative_quote": "Flaky tests are painful.",
        "sources": [
            {"blog_name": "A Blog", "post_title": "CI Woes", "post_url": "https://a.com/1",
             "published": "2024-05-01", "signal_text": "Flaky tests are painful.",
             "signal_type": "frustration", "impact_score": 0.8,
             "score_breakdown": {"trend": 0.3, "authority": 0.2, "breadth": 0.2, "recency": 0.1}},
        ],
    }]
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path, json_path = generate_ideas_report(ideas, tmpdir)
        assert os.path.exists(md_path)
        assert os.path.exists(json_path)
        assert md_path.endswith("ideas.md")
        assert json_path.endswith("ideas.json")


def test_ideas_report_content():
    ideas = [{
        "idea_id": 0,
        "label": "observability gap",
        "impact_score": 0.72,
        "justification": "3 blogs raise this. Related: logging, monitoring.",
        "keywords": ["logging", "monitoring"],
        "signal_count": 4,
        "blog_count": 3,
        "pain_type_breakdown": {"gap": 3, "wish": 1},
        "representative_quote": "No good way to correlate logs.",
        "sources": [
            {"blog_name": "Alpha Blog", "post_title": "Observability",
             "post_url": "https://alpha.com/obs", "published": "2024-04-20",
             "signal_text": "No good way to correlate logs.",
             "signal_type": "gap", "impact_score": 0.72,
             "score_breakdown": {"trend": 0.4, "authority": 0.2, "breadth": 0.1, "recency": 0.1}},
            {"blog_name": "Beta Blog", "post_title": "Logging Pain",
             "post_url": "https://beta.com/log", "published": "2024-05-01",
             "signal_text": "I wish there was a lightweight log tool.",
             "signal_type": "wish", "impact_score": 0.65,
             "score_breakdown": {"trend": 0.3, "authority": 0.1, "breadth": 0.1, "recency": 0.15}},
        ],
    }]
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path, json_path = generate_ideas_report(ideas, tmpdir)

        with open(md_path, encoding="utf-8") as f:
            md = f.read()
        assert "Project Ideas Report" in md
        assert "Observability Gap" in md
        assert "Alpha Blog" in md
        assert "Beta Blog" in md
        assert "Justification" in md
        assert "Sources" in md
        assert "Key Quotes" in md

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "ideas" in data
        assert len(data["ideas"]) == 1
        assert data["ideas"][0]["label"] == "observability gap"
        assert len(data["ideas"][0]["sources"]) == 2


def test_ideas_report_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path, json_path = generate_ideas_report([], tmpdir)
        assert os.path.exists(md_path)
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        assert "No project ideas detected" in content


# ── CLI tests ──


def test_cli_ideas_help():
    runner = CliRunner()
    result = runner.invoke(main, ["ideas", "--help"])
    assert result.exit_code == 0
    assert "max-features" in result.output
    assert "top-n" in result.output
    assert "period" in result.output
    assert "output-dir" in result.output


def test_cli_main_lists_ideas_command():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "ideas" in result.output
