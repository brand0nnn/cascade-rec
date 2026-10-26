# Why does the frozen MLP lose targets when K grows?

Validation diagnostics with frozen MLP seeds 0/1/2. No retraining, policy change or Test evaluation.

## Cohorts and interpretation

Uniform cohort: 1,000 random validation events. Rescored union: 5,814 events, including all eligible 200→500 demotions across three seeds. Per-seed demotion events: [3148, 3027, 3094].

Uniform-query score associations describe the random cohort. Overtaker profiles condition on demotion events and count seed–event–candidate occurrences, not equally weighted users or independent unique products. Repeated products/events across seeds can recur. These diagnostics do not estimate recommendation relevance of unobserved alternatives.

## Training versus validation score signal

| Population | Lists | Raw retrieval-score AUC, per-event | Per-user macro AUC |
|---|---:|---:|---:|
| Training: all | 315,114 | 0.080370 | 0.076231 |
| Training: retrieved_positive | 38,793 | 0.652839 | 0.649580 |
| Training: forced_positive | 276,321 | 0.000000 | 0.000000 |
| Validation: retrieved targets at K200 | 16,672 | 0.628975 | 0.625756 |

Training force-added fraction: 87.69%.

AUC here is the probability that the observed target has a higher raw retrieval score than an unobserved alternative within its list, with half credit for ties. Below 0.5 means the score direction is reversed for that constructed task. Training values compare against every valid training alternative; sampling 50 uniform alternatives has the same expected within-list pairwise AUC. Validation uses naturally retrieved targets only.

## Observed score association on uniform validation queries

| Seed | Spearman: raw retrieval vs MLP score, all 500 | Spearman, original 200 | Spearman, added 300 | Added-candidate share of final top 10 |
|---:|---:|---:|---:|---:|
| 0 | -0.402 | -0.084 | -0.296 | 94.7% |
| 1 | -0.372 | -0.076 | -0.264 | 95.6% |
| 2 | -0.400 | -0.076 | -0.294 | 95.2% |

These are means of within-query correlations and top-ten shares across the uniform event sample. Sample per-user macro and median correlations are retained in diagnostics.json. Association combines all varying candidate features; it is not an isolated feature effect.

## Hold the product/history fixed and vary only retrieval score

Use the product at retrieval rank 100 in each uniform query. Replace only numeric input 0 with z=[-3, -2, -1, 0, 1, 2, 3]; all other inputs and weights are unchanged. Raw score = training mean + z × training standard deviation.

| Seed | Mean logit at z=-3 | Mean logit at z=+3 | High minus low logit | Queries where lower retrieval score gets higher MLP score |
|---:|---:|---:|---:|---:|
| 0 | 20.270 | -5.037 | -25.308 | 99.9% |
| 1 | 10.910 | -5.021 | -15.931 | 99.7% |
| 2 | 17.746 | -4.803 | -22.548 | 100.0% |

This is an artificial feature-sensitivity probe, not a new accuracy result. Some altered score/product combinations may not occur naturally. It isolates the network's response to this input over the chosen grid, not the causal effect of the training policy. Scores are logits, not calibrated interaction probabilities.

## Which added products overtake demoted targets?

| Added candidate group | Share available | Share of overtakers | Share of added top-ten slots | Overtaker share / availability share |
|---|---:|---:|---:|---:|
| new-in-period | 8.9% | 15.2% | 27.9% | 1.70 |
| warm-head | 60.9% | 50.1% | 36.9% | 0.82 |
| warm-tail | 30.2% | 34.7% | 35.3% | 1.15 |

Overtakers have score ≥ the target, following the existing pessimistic tie rule; strict/tied counts are stored separately. Top-ten composition uses stable ordering for descriptive slot lists; target metrics retain pessimistic ties. Compare overtaker share against availability before describing overrepresentation. Item labels are descriptive (new-at-T1 first, then existing before-T2 warm head/tail), never model inputs.

### Most frequent leaf categories among overtakers

| Leaf category | Overtaker occurrences | Availability share | Overtaker share |
|---|---:|---:|---:|
| Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 149,090 | 26.6% | 25.6% |
| Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | 88,558 | 20.6% | 15.2% |
| Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | 56,398 | 6.2% | 9.7% |
| Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Screen Protectors | 31,208 | 4.9% | 5.4% |
| Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wall Chargers | 24,400 | 3.2% | 4.2% |
| Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cases | 23,506 | 3.1% | 4.0% |
| Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | 22,873 | 4.4% | 3.9% |
| Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Automobile Chargers | 20,414 | 3.4% | 3.5% |
| Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wireless Chargers | 20,331 | 3.9% | 3.5% |
| Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Charging Stations | 12,884 | 1.7% | 2.2% |

## Real product examples

One seed-0 example per target group, selected nearest the median rank loss among that group's demotions with a title. Examples illustrate the measured pattern; they were not chosen for the most extreme outcome. Alternative products are unobserved, not proven disliked or irrelevant.

### warm-head — event 26725

Target: **Moto G fast / 2020 / Unlocked / Made for US by Motorola / 3/32GB / 16MP Camera / White** (B0B7FJXP3M). MLP rank **4 → 55**, unchanged target logit **-2.5879**. Retrieval rank 55, retrieval score -0.1653.

Last three prior products: ykooe Moto X Pure Edition Case (5.7 inch), (Armor Series) Heavy Duty Protection Hybrid Shockproof Dual Layer Protective Case Cover with Stand for Motorola MotoPure Edition (Blue); LG V30 H932 Dual Camera 4G LTE 64GB Silver (GSM Unlocked) (Renewed); Spigen Rugged Armor Designed for Moto G Stylus Case (2020) / Designed for Moto G Pro Case (2020) - Matte Black.

| Added product | Item group / leaf category | Retrieval rank | Retrieval score | MLP logit | MLP rank at K500 |
|---|---|---:|---:|---:|---:|
| iOttie Velox Magnetic Car Mount Air Vent Phone Holder. Compatible with MagSafe iPhones including iPhone 12, iPhone 13, iPhone 14, iPhone 12/13/14 Pro & Pro Max, iPhone 12/13 mini, iPhone 14 Plus | new-in-period / Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | 418 | -0.2318 | 4.7203 | 1 |
| QHOHQ Case for Motorola Moto G Power 2021 (Not fit G Power 2020) with 2 Pack Screen Protector,[360° Rotating Stand] [5 Times Military Grade Anti-Fall Protection],Transparent PC Back Cover-Black | warm-tail / Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | 421 | -0.2319 | 3.7449 | 2 |
| TOCOL [6 Pack] Compatible with Moto G Power 2021 (Not for Moto G Power 2020) - 3 Pack Tempered Glass Screen Protector and 3 Pack Camera Lens Protector [Easy Installation Frame] HD Clear Case Friendly | warm-head / Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | 434 | -0.2333 | 3.1009 | 3 |
| for Motorola Moto G Play 2021 Case: Dual Layer Protective Heavy Duty Cell Phone Cover Shockproof Rugged with Non Slip Textured Back - Military Protection Bumper Tough - 6.5inch (Black Black) | new-in-period /  | 429 | -0.2330 | 3.0763 | 4 |
| ONOLA Compatible with Moto G Power 2021 Case with HD Screen Protector (2 Packs) [Not for G Power 2020],Moto G Power Case 2021 Full Body 3 in 1 Durable Case for Moto G Power 2021 (Grey-SkyBlue) | warm-tail / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 388 | -0.2291 | 2.4483 | 5 |

### new-in-period — event 274648

Target: **OtterBox IPhone 13 Pro Max & IPhone 12 Pro Max Symmetry Series Case - BLACK, Ultra-Sleek, Wireless Charging Compatible, Raised Edges Protect Camera & Screen** (B0BZT4P292). MLP rank **4 → 50**, unchanged target logit **-4.5124**. Retrieval rank 64, retrieval score -0.1309.

Last three prior products: [10 Pack] Screen Protector for Apple Watch Series 4 44mm / Series 3 42mm, SPARIN TPU Full Coverage Flexible Film with Case Friendly/Ultimate Thin/HD Clear; Miracase Product Great; OtterBox IPhone 13 Pro Max & IPhone 12 Pro Max Defender Series Case - FORT BLUE, Rugged & Durable, With Port Protection, Includes Holster Clip Kickstand.

| Added product | Item group / leaf category | Retrieval rank | Retrieval score | MLP logit | MLP rank at K500 |
|---|---|---:|---:|---:|---:|
| ESR for iPhone 13 Pro Max Phone Case, Military-Grade Protection, Shockproof Air-Guard Corners, Yellowing-Resistant Acrylic Back, Phone Case for iPhone 13 Pro Max, Air Armor Case, Clear Black | new-in-period / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 350 | -0.1846 | 3.1986 | 1 |
| UniqueMe for iPhone 13 Pro/iPhone 13 Pro Max Camera Lens Protector, [Strong Drop Protection] [Not Easy to Fall-off] Metal Ring Tempered Glass Rear Camera Screen Cover, Alpine Green | new-in-period / Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | 434 | -0.1915 | 2.1021 | 2 |
| TETHYS 3 Pack Glass Screen Protector Compatible with iPhone 13 Pro Max 2021 6.7 Inch, Case Friendly Tempered Glass Film 9H Hardness HD Shield (Guidance Frame included) | new-in-period / Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | 431 | -0.1913 | 1.3873 | 3 |
| Saupsitnz Designed for iPhone 13 Pro Max Waterproof Case, Built-in Screen Protector 360° Full Body Protective Dust-Proof Shockproof Waterproof Case for iPhone 13 Pro Max 6.7 inch 2021 (Blue) | new-in-period / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 438 | -0.1918 | 0.8839 | 4 |
| Redpepper for iPhone 12 Pro Max Case with Built-in Screen Protector,Compatible with Magsafe Full Body Heavy Duty Shockproof Case for iPhone 12 Pro Max-6.7inch (Black) | warm-tail / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 432 | -0.1913 | 0.5059 | 5 |

### warm-tail — event 483053

Target: **amFilm (3 Pack) Glass Screen Protector for Samsung Galaxy Note 9, Full Screen Coverage Screen Protector, 3D Curved Tempered Glass, Dot Matrix with Easy Installation Tray** (B08ZN3Z5TD). MLP rank **4 → 55**, unchanged target logit **-3.9999**. Retrieval rank 1, retrieval score 0.0035.

Last three prior products: BENTOBEN Galaxy Note 9 Case, Purple Space Design Slim 2 in 1 Hybrid Soft TPU Bumper Hard PC Cover with PU Faux Leather Heavy Duty Shockproof Protective Phone Case Cover for Samsung Note 9, Purple; OTAO (2 Pack) Tempered Glass Screen Protector for Samsung Galaxy Note 9 3D Curved Dot Matrix with Easy Installation Tray (Full Coverage); Kitoo Designed for Samsung Galaxy Note 9 Case with Kickstand, Military Grade 12ft. Drop Tested - Blue.

| Added product | Item group / leaf category | Retrieval rank | Retrieval score | MLP logit | MLP rank at K500 |
|---|---|---:|---:|---:|---:|
| PIXIU for Galaxy Note 9 case, Unique Pattern Heavy Duty Shockproof 3 in 1 hybird Rubber stury Full-Body Protective case Cover for Samsung Galaxy Note 9 2018 Released Marble | warm-tail / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 436 | -0.1868 | 1.7164 | 1 |
| GPEL Privacy Screen Protector Compatible for Galaxy Note 9 Real Tempered Premium Japanese Asahi Glass w/Applicator Anti Spy Case-Friendly HD Clarity 9H Hardness 99% Touch Accurate | warm-head /  | 485 | -0.1914 | 1.2191 | 2 |
| eSamcore Samsung Galaxy Note 9 Case Ring Holder Kickstand Cases + Dashboard Magnetic Phone Car Mount [RED] | warm-tail / Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | 424 | -0.1862 | 0.2746 | 3 |
| AUPAI Compatible with Samsung Galaxy Note 9 Case, Heavy Duty 15ft Drop Tested Shockproof Cover with Ring Kickstand for Samsung Galaxy Note 9, Rosegold | warm-tail / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 455 | -0.1882 | -0.1633 | 4 |
| SURITCH for Samsung Galaxy Note 9 Marble Case, [Built-in Screen Protector] Natural Marble Full-Body Protection Shockproof Rugged Bumper Protective Cover for Galaxy Note 9 6.4 Inch (Gold Marble) | warm-head / Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | 352 | -0.1788 | -0.2661 | 5 |

## Limits and reproducibility

- All selected cohort K200/K500 ranks must match the saved sweep ranks exactly. Six reproduction checks are saved in diagnostics.json.
- Raw-score distribution changes, score correlations and a one-input probe can triangulate an explanation for this frozen model. They do not establish that force-insertion caused the learned response; a controlled retraining comparison would be needed for that claim.
- This analysis does not change split/labels/sampling/training/checkpoints or the prior candidate-count recommendation. Test remains untouched. Source/config/checkpoint hashes, cohort scores, per-query correlations, rank-band summaries, profiles and probe logits are saved.
