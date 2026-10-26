# MLP feature ladder

Full-data validation discovery, one seed. Replication is still required.

Checkpoint selection: conditional per-user macro nDCG@10. End-to-end metrics include every retrieval miss as zero.

| Setup | Best epoch | Conditional macro nDCG@10 | End-to-end macro nDCG@10 | Macro GAUC | Rescued | Demoted |
|---|---:|---:|---:|---:|---:|---:|
| matching | 7 | 0.095782 | 0.006660 | 0.710535 | 2467 | 1818 |
| attributes | 1 | 0.101582 | 0.006853 | 0.747259 | 2871 | 1980 |
