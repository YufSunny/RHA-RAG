"""Shared pytest fixtures for RHA-RAG tests."""

import pytest


@pytest.fixture
def sample_chart_spec() -> dict:
    """A minimal but valid ChartSpec dict for tests."""
    return {
        "type": "bar",
        "title": "P0420 frequency by code",
        "x_label": "Code",
        "y_label": "Count",
        "data": [
            {"x": "P0420", "y": 1842, "label": "Catalyst Bank 1"},
            {"x": "P0430", "y": 612, "label": "Catalyst Bank 2"},
        ],
        "citation": {
            "filename": "dtc-codes.csv",
            "label": "OBD-II DTC frequency 2023",
        },
        "confidence": "extracted",
    }


@pytest.fixture
def sample_estimated_chart_spec() -> dict:
    """A valid ChartSpec that requires a note (confidence=estimated)."""
    return {
        "type": "line",
        "title": "NEV penetration trend (estimated)",
        "x_label": "Year",
        "y_label": "Share %",
        "data": [
            {"x": "2020", "y": 5.4},
            {"x": "2021", "y": 13.4},
            {"x": "2022", "y": 25.6},
            {"x": "2023", "y": 31.6},
            {"x": "2024", "y": 41.6},
        ],
        "confidence": "estimated",
        "note": "Aggregated from CAAM monthly tables; rounded to annual avg.",
    }