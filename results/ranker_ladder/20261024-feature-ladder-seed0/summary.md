# MLP feature ladder

Full-data validation discovery, one seed. Replication is still required.

Checkpoint selection: conditional per-user macro nDCG@10. End-to-end metrics include every retrieval miss as zero.

| Setup | Best epoch | Conditional macro nDCG@10 | End-to-end macro nDCG@10 | Macro GAUC | Rescued | Demoted |
|---|---:|---:|---:|---:|---:|---:|
| score | 6 | 0.025049 | 0.001600 | 0.442328 | 554 | 2292 |
| matching | 10 | 0.102843 | 0.007105 | 0.713333 | 2533 | 1707 |
| attributes | 3 | 0.094674 | 0.006522 | 0.757001 | 2853 | 2054 |
| statistics | 1 | 0.056142 | 0.003914 | 0.713765 | 1745 | 2249 |
