# MLP feature ladder

Full-data validation discovery, one seed. Replication is still required.

Checkpoint selection: conditional per-user macro nDCG@10. End-to-end metrics include every retrieval miss as zero.

| Setup | Best epoch | Conditional macro nDCG@10 | End-to-end macro nDCG@10 | Macro GAUC | Rescued | Demoted |
|---|---:|---:|---:|---:|---:|---:|
| matching | 9 | 0.098543 | 0.006764 | 0.712118 | 2429 | 1716 |
| attributes | 3 | 0.091500 | 0.006276 | 0.748599 | 2807 | 2092 |
