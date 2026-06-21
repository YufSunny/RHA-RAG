# data/auto-seed — Automaker A anonymized corpus

Small bundled corpus so the demo works on first boot with zero uploads.
Focus: **Automaker A** (anonymized placeholder brand name for a major
Chinese NEV manufacturer) — sales, model lineup, and global market
position.

All numeric data is from **public sources** cited in each file; the
brand label **"Automaker A"** is a generic placeholder. No proprietary
or restricted material is included. Files are loaded by the existing
`rha_rag.pipeline` loader (`.csv` and `.md`).

| File | Source | License |
| --- | --- | --- |
| `automaker-annual.csv` | Public annual reports 2019–2025; Statista compilation | Public summary |
| `automaker-quarterly.csv` | Public monthly disclosures 2023-Q2 → 2025-Q4 | Public summary |
| `automaker-models-2025.csv` | Public model-level retail data 2025 | Public summary |
| `automaker-market-share.md` | CPCA / CAAM retail share + Statista global EV share | Public summary |
| `automaker-city-ev.md` | Public press release dated 30 June 2025 (City-EV product profile) | Public release |

If a source ever moves or becomes restricted, replace the file with a
new public equivalent. Do not commit proprietary dealer data.

> **Anonymization scope.** Product names that were model-specific
> (e.g. Seagull, Dolphin Mini, Song, Qin, Han, Tang, Yuan) have been
> generalized to neutral category labels (City-EV, SUV-Pro, Sedan-Pro,
> SUV-XL, Crossover, Sedan-Full, etc.) so the corpus does not embed any
> original brand identity. Technology terms (e.g. blade battery,
> e-Platform 3.0, DM-i drivetrain) are likewise generalized (LFP
> battery, EV-Platform A, PHEV drivetrain).
