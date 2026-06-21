# data/auto-seed — BYD sales & market share seed corpus

Small bundled corpus so the demo works on first boot with zero uploads.
Focus: **BYD Auto** sales, model lineup, and global market position.
All data is from **public sources** cited in each file; no proprietary
or restricted material is included. Files are loaded by the existing
`rha_rag.pipeline` loader (`.csv` and `.md`).

| File | Source | License |
|---|---|---|
| `byd-annual.csv` | BYD Global annual reports; Statista compilation | Public summary |
| `byd-quarterly.csv` | BYD Global monthly disclosures 2023-2025 | Public summary |
| `byd-models-2025.csv` | BYD Global model-level retail data 2025 | Public summary |
| `byd-market-share.md` | CPCA / CAAM retail share + Statista global EV share | Public summary |
| `byd-seagull.md` | BYD Global press release dated 30 June 2025 (Seagull / Dolphin Mini) | Public release |

If a source ever moves or becomes restricted, replace the file with a
new public equivalent. Do not commit proprietary dealer data.
