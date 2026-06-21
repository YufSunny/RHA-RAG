# data/auto-seed — automotive seed corpus

Small bundled corpus so the demo works on first boot with zero uploads.
All data is from **public sources** cited below; no proprietary or
restricted material is included. Files are loaded by the existing
`rha_rag.pipeline` loader (`.csv` and `.md`).

| File | Source | License |
|---|---|---|
| `dtc-codes.csv` | SAE J2012 generic OBD-II DTC definitions | Public reference |
| `caam-monthly-sales.csv` | CAAM (中国汽车工业协会) monthly sales summaries 2020–2024 | Public summary |
| `cpca-segment-share.csv` | CPCA (乘联会) passenger-car segment share 2020–2024 | Public summary |
| `toyota-p0420.md` | Toyota TSB on P0420 (cat. converter efficiency) | Public excerpt |
| `bmw-fault-codes.md` | BMW fault-code glossary excerpt | Public excerpt |
| `ford-service-intervals.md` | Ford scheduled maintenance interval summary | Public excerpt |

If a source ever moves or becomes restricted, replace the file with a
new public equivalent. Do not commit proprietary service data.