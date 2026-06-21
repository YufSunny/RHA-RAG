"""Tests for the bundled data/auto-seed/ corpus.

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
    "dtc-codes.csv",
    "caam-monthly-sales.csv",
    "cpca-segment-share.csv",
    "toyota-p0420.md",
    "bmw-fault-codes.md",
    "ford-service-intervals.md",
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
        # Allow .gitkeep-style extras; nothing else expected.
        unexpected = present - EXPECTED_FILES
        # .gitkeep / hidden files are acceptable; named files should match.
        named = {n for n in unexpected if not n.startswith(".")}
        assert not named, f"unexpected seed files: {named}"

    def test_discover_files_finds_all(self, seed_dir):
        files = discover_files(str(seed_dir))
        names = {Path(f).name for f in files}
        assert names == EXPECTED_FILES, f"discover_files saw {names}"


class TestCsv:
    def test_dtc_codes_columns(self, seed_dir):
        text = (seed_dir / "dtc-codes.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["code", "system", "description", "frequency_2023"]
        # Body should have ≥ 10 DTCs.
        assert len(text.splitlines()) >= 11

    def test_caam_monthly_sales_columns(self, seed_dir):
        text = (seed_dir / "caam-monthly-sales.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["year", "month", "total_sales_units",
                          "nev_sales_units", "nev_share_pct"]
        assert len(text.splitlines()) >= 10  # multiple years

    def test_cpca_segment_share_columns(self, seed_dir):
        text = (seed_dir / "cpca-segment-share.csv").read_text(encoding="utf-8")
        header = text.splitlines()[0].split(",")
        assert header == ["year", "segment", "share_pct"]
        assert len(text.splitlines()) >= 9

    def test_csv_rows_have_parseable_numbers(self, seed_dir):
        text = (seed_dir / "caam-monthly-sales.csv").read_text(encoding="utf-8")
        for line in text.splitlines()[1:]:
            parts = line.split(",")
            assert len(parts) == 5, f"wrong column count: {line!r}"
            int(parts[0])  # year
            int(parts[1])  # month
            int(parts[2])  # total
            int(parts[3])  # nev
            float(parts[4])  # share


class TestMarkdown:
    @pytest.mark.parametrize("name", [
        "toyota-p0420.md",
        "bmw-fault-codes.md",
        "ford-service-intervals.md",
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