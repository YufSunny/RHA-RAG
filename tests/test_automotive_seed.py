"""Tests for the bundled data/auto-seed/ corpus (Automaker A focus).

Covers:
- All expected files exist.
- Each CSV parses with the documented column shape.
- Each markdown file renders to non-empty plain text.
- Total file count matches the README's manifest.
"""

from pathlib import Path

import pytest

from rha_rag.pipeline import discover_files, load_all_documents

SEED_DIR = Path("data/auto-seed")

EXPECTED_FILES = {
    "README.md",
    "automaker-annual.csv",
    "automaker-quarterly.csv",
    "automaker-models-2025.csv",
    "automaker-market-share.md",
    "automaker-city-ev.md",
}


@pytest.fixture(scope="module")
def seed_dir() -> Path:
    if not SEED_DIR.exists():
        pytest.skip(f"seed dir missing: {SEED_DIR.resolve()}")
    return SEED_DIR


class TestManifest:
    def test_all_expected_files_present(self, seed_dir):
        present = {p.name for p in seed_dir.iterdir() if p.is_file()}
        missing = EXPECTED_FILES - present
        assert not missing, f"missing seed files: {missing}"

    def test_no_unexpected_files(self, seed_dir):
        present = {p.name for p in seed_dir.iterdir() if p.is_file()}
        unexpected = present - EXPECTED_FILES
        named = {n for n in unexpected if not n.startswith(".")}
        assert not named, f"unexpected seed files: {named}"

    def test_discover_files_finds_all(self, seed_dir):
        files = discover_files(str(seed_dir))
        names = {Path(f).name for f in files}
        assert names == EXPECTED_FILES, f"discover_files saw {names}"


class TestCsv:
    def test_automaker_annual_columns(self, seed_dir):
        text = (seed_dir / "automaker-annual.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["year", "production", "sales", "bev_sales",
                          "phev_sales", "china_sales", "overseas_sales"]
        # Body should cover 2019-2025 (7 years).
        assert len(text.splitlines()) >= 8

    def test_automaker_quarterly_columns(self, seed_dir):
        text = (seed_dir / "automaker-quarterly.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["quarter", "sales", "overseas_sales", "note"]
        # Body should cover 2023-Q2 through 2025-Q4 (11 rows).
        assert len(text.splitlines()) >= 12

    def test_automaker_models_2025_columns(self, seed_dir):
        text = (seed_dir / "automaker-models-2025.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["model", "series", "segment", "units_2025", "yoy_pct"]
        # Body should have at least 10 Automaker A models.
        assert len(text.splitlines()) >= 11

    def test_csv_rows_have_parseable_numbers(self, seed_dir):
        text = (seed_dir / "automaker-annual.csv").read_text(encoding="utf-8")
        for line in text.splitlines()[1:]:
            parts = line.split(",")
            assert len(parts) == 7, f"wrong column count: {line!r}"
            int(parts[0])    # year
            int(parts[1])    # production
            int(parts[2])    # sales
            int(parts[3])    # bev_sales
            int(parts[4])    # phev_sales
            int(parts[5])    # china_sales
            int(parts[6])    # overseas_sales

    def test_annual_2024_totals_consistent(self, seed_dir):
        # 2024 bev + phev should equal passenger sales (within rounding).
        text = (seed_dir / "automaker-annual.csv").read_text(encoding="utf-8")
        row_2024 = next(l for l in text.splitlines() if l.startswith("2024,"))
        parts = row_2024.split(",")
        bev, phev = int(parts[3]), int(parts[4])
        # Sum of bev+phev is the passenger sales (commercial separate).
        assert abs(bev + phev - 4250370) < 5, f"2024 BEV+PHEV total off: {bev}+{phev}"


class TestMarkdown:
    @pytest.mark.parametrize("name", [
        "automaker-market-share.md",
        "automaker-city-ev.md",
    ])
    def test_markdown_non_empty(self, seed_dir, name):
        text = (seed_dir / name).read_text(encoding="utf-8").strip()
        assert len(text) > 50, f"{name} suspiciously short"


class TestLoaderIntegration:
    def test_load_all_documents_returns_one_per_file(self, seed_dir):
        docs = load_all_documents([str(seed_dir)])
        # One Document per file (load_text returns single-doc lists).
        assert len(docs) >= len(EXPECTED_FILES)
        sources = {d.metadata["name"] for d in docs}
        assert EXPECTED_FILES.issubset(sources)

    def test_metadata_carries_filename(self, seed_dir):
        docs = load_all_documents([str(seed_dir)])
        for d in docs:
            assert "source" in d.metadata
            assert "name" in d.metadata
            # `source` is a forward-slash path.
            assert "\\" not in d.metadata["source"]
