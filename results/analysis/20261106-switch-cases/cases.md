# Val switch cases: targets missed by both retrievers

**Val only, exploratory; model-assisted coding, needs human review.**

Recency = frozen MiniLM, 90-day half-life, natural K200 (`20261103-full-promising-ranking`); Two-tower (text) seed 0, exact search, 500-lists. Double miss = target outside both top 200s. Frozen rule: `config.yaml` (sha256 `5815f7011096`), written before selection; IDs and eligible counts in `selection.json`. Titles are crawl-time metadata. The device-audit gate is blocked, so 'same device ecosystem' only records device families explicit in both titles and asserts no compatibility. 'store change' compares the metadata store string, a proxy for brand. Later reviews are context only and stop at T2. Cases illustrate the aggregate table; they do not prove it.

Val events 232,904; in recency top 200 14,916; in two-tower top 200 16,673; in both 2,796; double misses 204,111 (new 69,185; warm with 0 reviews before T1, outside F1-F3: 0).

## Aggregate (all events, automatic metadata rubric vs last history product)

| Population | Events | Measure | Yes | No | Unknown | Share yes of known |
|---|---|---|---|---|---|---|
| double_miss_all | 204,111 | leaf_category_jump | 164,403 | 31,534 | 8,174 | 0.8391 |
| double_miss_all | 204,111 | top_category_jump | 0 | 195,937 | 8,174 | 0.0 |
| double_miss_all | 204,111 | second_level_jump | 89,249 | 106,688 | 8,174 | 0.4555 |
| double_miss_all | 204,111 | store_change | 201,102 | 2,813 | 196 | 0.9862 |
| double_miss_all | 204,111 | leaf_unseen_in_history | 113,406 | 87,780 | 2,925 | 0.5637 |
| retrieved_by_at_least_one | 28,793 | leaf_category_jump | 15,431 | 12,623 | 739 | 0.55 |
| retrieved_by_at_least_one | 28,793 | top_category_jump | 0 | 28,054 | 739 | 0.0 |
| retrieved_by_at_least_one | 28,793 | second_level_jump | 7,167 | 20,887 | 739 | 0.2555 |
| retrieved_by_at_least_one | 28,793 | store_change | 23,517 | 5,264 | 12 | 0.8171 |
| retrieved_by_at_least_one | 28,793 | leaf_unseen_in_history | 9,083 | 19,515 | 195 | 0.3176 |
| double_miss_F1_new_target | 69,185 | leaf_category_jump | 54,115 | 12,567 | 2,503 | 0.8115 |
| double_miss_F1_new_target | 69,185 | top_category_jump | 0 | 66,682 | 2,503 | 0.0 |
| double_miss_F1_new_target | 69,185 | second_level_jump | 30,753 | 35,929 | 2,503 | 0.4612 |
| double_miss_F1_new_target | 69,185 | store_change | 67,775 | 1,336 | 74 | 0.9807 |
| double_miss_F1_new_target | 69,185 | leaf_unseen_in_history | 33,722 | 34,670 | 793 | 0.4931 |
| double_miss_F2_warm_ge50 | 47,470 | leaf_category_jump | 40,101 | 5,761 | 1,608 | 0.8744 |
| double_miss_F2_warm_ge50 | 47,470 | top_category_jump | 0 | 45,862 | 1,608 | 0.0 |
| double_miss_F2_warm_ge50 | 47,470 | second_level_jump | 20,036 | 25,826 | 1,608 | 0.4369 |
| double_miss_F2_warm_ge50 | 47,470 | store_change | 46,919 | 516 | 35 | 0.9891 |
| double_miss_F2_warm_ge50 | 47,470 | leaf_unseen_in_history | 30,118 | 16,961 | 391 | 0.6397 |
| double_miss_F3_warm_1to49 | 87,456 | leaf_category_jump | 70,187 | 13,206 | 4,063 | 0.8416 |
| double_miss_F3_warm_1to49 | 87,456 | top_category_jump | 0 | 83,393 | 4,063 | 0.0 |
| double_miss_F3_warm_1to49 | 87,456 | second_level_jump | 38,460 | 44,933 | 4,063 | 0.4612 |
| double_miss_F3_warm_1to49 | 87,456 | store_change | 86,408 | 961 | 87 | 0.989 |
| double_miss_F3_warm_1to49 | 87,456 | leaf_unseen_in_history | 49,566 | 36,149 | 1,741 | 0.5783 |
| double_miss_warm_0_before_t1 | 0 | leaf_category_jump | 0 | 0 | 0 | nan |
| double_miss_warm_0_before_t1 | 0 | top_category_jump | 0 | 0 | 0 | nan |
| double_miss_warm_0_before_t1 | 0 | second_level_jump | 0 | 0 | 0 | nan |
| double_miss_warm_0_before_t1 | 0 | store_change | 0 | 0 | 0 | nan |
| double_miss_warm_0_before_t1 | 0 | leaf_unseen_in_history | 0 | 0 | 0 | nan |

## Rubric summary (counts within the 24 drawn cases)

| Bucket | Eligible | Chosen | Ecosystem yes/no/unk | Relation sub/comp/unrel/unk | Leaf jump y/n/unk | Store change y/n/unk | Leaf unseen y/n/unk | Price up/similar/down/unk |
|---|---|---|---|---|---|---|---|---|
| F1_new_target | 69,185 | 8 | 5/0/3 | 1/1/6/0 | 5/1/2 | 8/0/0 | 6/1/1 | 0/3/2/3 |
| F2_warm_ge50 | 47,470 | 8 | 2/1/5 | 2/1/5/0 | 7/1/0 | 8/0/0 | 4/4/0 | 0/3/2/3 |
| F3_warm_1to49 | 87,456 | 8 | 4/1/3 | 3/2/3/0 | 3/4/1 | 8/0/0 | 4/4/0 | 0/4/0/4 |

## Pattern findings (model-assisted, needs human review)

These notes describe the 24 drawn cases and the aggregate table above; they are descriptive, not causal, and say nothing about user intent beyond what titles show.

- **Hypothesis check (frozen operational test): holds.** Among double misses with known categories, 83.9% (164,403 / 195,937) change leaf category from the last history product, versus 55.0% (15,431 / 28,054) among events retrieved by at least one retriever. Caveats: double misses are 87.6% of all Val events, so they are the base case (all-event leaf-jump share 80.3%), and both retrievers are text-similarity retrievers, so lower jump rates among retrieved events are partly built in. The "minority same-category" part also holds in size: 16.1% (31,534) of known double misses keep the last product's leaf. "No text overlap" was not measured; in the cards, same-leaf misses usually share the product type and differ in the device model named in the title.
- **Top-level category is uninformative** here: every known item has top_cat "Cell Phones & Accessories", so the top-level jump is 0% in every population. The supplementary second-level path (for example Cases vs Accessories) changes in 45.6% of known double misses versus 25.6% of retrieved events.
- **Store change is almost universal** among double misses (98.6%, 201,102 / 203,915 known) versus 81.7% among retrieved events; all 24 drawn cases change store. The target leaf never appears in the user's prior history in 56.4% of known double misses versus 31.8% of retrieved events.
- **Device change.** The most common pattern in the cards is a new phone or watch model the history does not name. The target matches the product type, or the brand family, of earlier purchases, while the retrievers return items for the old model: Pixel XL to Pixel 5a case (1756859, recency top 3 = Pixel XL cases), Galaxy S10e holster to Z Flip pouch (2369503), iPhone 12 Pro Max to iPhone 14 Pro lens protector (2323078), iPhone XR case to iPhone 13 mini protector (114423), Pixel 3 charger to Pixel 6 Pro protector (840982), and LG G7 battery to Gear S3 battery (244763). Later same-user reviews often name the new device (114423, 1997091, 244763), as context only.
- **New device category.** Four of 24 targets are Apple Watch bands after phone accessories (47323, 209154, 1532879, 2347616); the smartwatch-band leaf is absent from all four histories. Both retrievers return phone protectors and cases.
- **Generic accessories with no device anchor** (wall or car chargers, stylus pens, mounts, dust plugs, holsters): 764303, 2244325, 825873, 1488653, 1732724, 1837606, 219280. Several follow long gaps (219280: 7.7 years; 1732724: 4.4 years), so the "last product" is a weak anchor.
- **Near misses are rare.** Only 1491415 (two-tower rank 224; recency top 1 is a sibling product from the target's store) and 1732724 (two-tower 323) fall in the two-tower 201-500 band. 482717 is a same-day, same-leaf Apple Watch complement that is still far down both lists. Rubric counts: relation 6 substitute / 4 complement / 14 unrelated; same ecosystem 11 yes (9 of them the same brand or family but a different device or model) / 2 no / 11 unknown. Across all double misses, 10,093 (4.9%) have a two-tower rank of 201-500.
- **Metadata caveats.** Titles are crawl-time listings: some name models released after the review date (2323078, 2572126, 840982 ASIN), and 3 cards have a missing category on the target or last item. Ecosystem codes record only device names in titles; the device-audit gate is blocked, so no compatibility is asserted.

## Case cards

### F1 - instance 47323 (user 6540, history length 2)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2021-11-01 | QHOHQ [3 Pack] Tempered Glass Camera Lens Protector for iPhone 13 Pro 6.1" ＆ iPhone 13 Pro Max 6.7", 9H Hardness, Ultra HD, Anti-Scratch, Easy to Install, Case Friendly [Does not Affect Night Shots] | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Lens Protectors | QHOHQ | 7.99 |
| h2 | 2021-11-04 | FYY Compatible with iPhone 13 Case, [Kickstand Feature] Luxury PU Leather Wallet Case Flip Folio Cover with [Card Slots] and [Note Pockets] Case for iPhone 13 5G 6.1" Canvas Green | Cell Phones & Accessories > Cases, Holsters & Sleeves > Flip Cases | FYY | 14.57 |
| **target** | 2021-12-11 | **Zsuoop Metal Stainless Steel Band Compatible with Apple Watch Bands 38mm 40mm 41mm 42mm 44mm 45mm With Screen Protector Case,Magnetic Adjustable strap for iWatch Series 8/7/6/5/4/SE/3/2/1 Women Men** | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | Zsuoop | 12.99 |

- Target `B09QSM4Q17`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 58,142); recency rank: >200.
- Recency top 3: (1) QHOHQ 3 Pack Screen Protector for iPhone 13 Pro 6.1 Inch with 3 Pack Tempered Glass Camera Lens Protector, Ultra HD, 9H Hardness, Scratch Resistant - Case Friendly [Not fit iPhone 13 Pro Max 6.7"] [QHOHQ]; (2) QHOHQ 3 Pack Screen Protector for iPhone 13 Pro Max 6.7" with 3 Pack Tempered Glass Camera Lens Protector, Ultra HD, 9H Hardness, Scratch Resistant, Easy Install - Case Friendly [QHOHQ]; (3) FYY Compatible with iPhone 13 Pro Max Case, [Kickstand Feature] Luxury PU Leather Wallet Case Flip Folio Cover with [Card Slots] and [Note Pockets] Case for iPhone 13 Pro Max 5G 6.7" Fish Scale Black [FYY]
- Two-tower top 3: (1) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (2) Ailun Screen Protector Compatible for iPhone 8 plus,7 Plus,6s Plus,6 Plus, 5.5 Inch 3Pack Case Friendly Tempered Glass [Ailun]; (3) JETech Screen Protector for iPhone 11 6.1-Inch with Camera Lens Protector, Tempered Glass Film, 2-Pack Each [JETech]
- Same user's later reviews within 30 days (context only, Val period only): 2021-12-11: baozai Compatible with Apple Watch Band with Case, Stainless Steel Band with Glass Screen Protector Case for iWatch Series 8/7/6/5/4/3/2/1/SE/Ultra [baozai]
- **Rubric vs h2 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = similar. Note: Apple brand only (iPhone 13 case -> Apple Watch band; different device). Smartwatch-band leaf absent from history; a second Apple Watch band+case review the same day suggests a newly acquired watch (context only).

### F1 - instance 114423 (user 15860, history length 1)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2021-10-31 | OTOFLY Compatible with iPhone XR Case 6.1 inch,[Silky and Soft Touch Series] Premium Soft Liquid Silicone Rubber Full-Body Protective Bumper Case for iPhone XR (Purple) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | OTOFLY | 13.97 |
| **target** | 2021-12-23 | **Ailun Glass Screen Protector for iPhone 13 mini [5.4 Inch] Display 2021 3 Pack Tempered Glass** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Ailun | 6.88 |

- Target `B09CMJMB7W`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 3,109); recency rank: >200.
- Recency top 3: (1) OTOFLY Compatible with iPhone 12 Mini Case 5.4 inch(2020),[Silky and Soft Touch Series] Premium Soft Liquid Silicone Rubber Full-Body Protective Bumper Case Black [OTOFLY]; (2) OTOFLY Liquid Silicone Gel Rubber Full Body Protection Shockproof Case for iPhone Xs/iPhone X，Anti-Scratch&Fingerprint Basic-Cases，Compatible with iPhone X/iPhone Xs 5.8 inch (2018), (Space Gray) [OTOFLY]; (3) OTOFLY Compatible with iPhone 13 Mini Case,Liquid Silicone Slim Protective Shockproof Phone Case Cover with Anti-Scratch Microfiber Lining, 5.4 inch (Teal) [OTOFLY]
- Two-tower top 3: (1) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]; (2) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (3) TETHYS Glass Screen Protector Designed For iPhone 11 / iPhone XR (6.1") [Edge to Edge Coverage] Full Protection Durable Tempered Glass Compatible iPhone XR/11 [Guidance Frame Include] - Pack of 3 [TETHYS]
- Same user's later reviews within 30 days (context only, Val period only): 2022-01-13: LUPA Legacy iPhone 13 Pro Wallet Case - Case with Card Holder - [Slim + Durable] for Women and Men - iPhone 13 Pro Flip Cell Phone case - Faux Leather - Folio Cover - Burgundy [LUPA Legacy]; 2022-01-13: SPARIN Screen Protector for iPhone 13, 3 Pack 9H Tempered Glass and 2 Pack Camera Lens Protector for iPhone 6.1 Inch 2021, Alignment Tool, Bubble Free [SPARIN]
- **Rubric vs h1 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = down. Note: iPhone XR case -> iPhone 13 mini protector: same family, different model, so not jointly usable from titles. Later reviews name iPhone 13/13 Pro accessories: device-change pattern (context only).

### F1 - instance 209154 (user 28955, history length 3)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2017-04-08 | Speck Products CandyShell Clear Case for Moto Z Force Droid Smartphone, Beaming Orchid Purple Clear | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Speck | 22.57 |
| h2 | 2019-02-04 | Speck Products CandyShell Grip iPhone XR Case, Ultraviolet Purple/Ruby Red | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Speck | - |
| h3 | 2020-11-03 | ORIbox Case Compatible with iPhone 12 Case & Glass Screen Protector for iPhone 12 and iPhone 12 Pro, 3 Packs Anti-Scratch Eye-Protection HD Clear Tempered Glass Screen Protector | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | ORIbox | 17.55 |
| **target** | 2022-01-15 | **Bands Compatible with Apple Watch 38mm 40mm 41mm 42mm 44mm 45mm, 6 Pack Replacement Soft Silicone Sport Accessory Strap for iWatch Series 7/6/5/4/3/2/1 SE Women Men** | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | Distore | 9.99 |

- Target `B09MS51HKB`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 18,912); recency rank: >200.
- Recency top 3: (1) ORIbox Glass Screen Protector for iPhone 12 mini, 3 Packs Anti-Scratch Eye-Protection HD Clear Tempered Glass Screen Protector [ORIbox]; (2) ORIbox Case Compatible with iPhone 12 pro Case, Compatible with iPhone 12 Case, with 4 Corners Shockproof Protection [ORIbox]; (3) ORIbox Case Compatible with iPhone 12/12 pro Case, Super Thin Mobile Phone Cover Case with Matte Finish Coating Grip [ORIbox]
- Two-tower top 3: (1) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]; (2) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (3) Yootech Wireless Charger,10W Max Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [yootech]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h3 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = yes; price jump = similar. Note: Apple brand only (iPhone 12 case/protector -> Apple Watch bands, 14-month gap). New device category; both retrievers return iPhone protectors/cases.

### F1 - instance 741481 (user 102546, history length 5)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2015-03-20 | New Waterproof Shockproof Dirtproof Snowproof Protection Case Cover for Apple iPhone 5 (Teal) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | RedPepper | - |
| h2 | 2019-06-30 | iPhone 5/5S SE Case, EXSEK Hybrid Impact Ultra Slim 3 Color Shockproof Case [Anti-Slip] [Extra Front Raised Lip] Scratch Resistant Soft Gel Bumper Rugged Case for iPhone 5/5S (Wine Red) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | EXSEK | 5.99 |
| h3 | 2019-08-07 | HOMEMO Glass Screen Protector for iPhone 11/iPhone XR 6.1 Inch 3 Pack Tempered Glass | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | HOMEMO | 5.89 |
| h4 | 2022-03-18 | Elecshion (Full-Coverage) Privacy Screen Protector for iPhone 11 and iPhone XR(2 Pack), Anti-spy Tempered Glass Screen Protector for iPhone 11/XR(6.1 ''), Bubble Free, (Case Friendly) | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Elecshion | 8.49 |
| h5 | 2022-05-17 | USB Wall Charger, KENHAO 3-Pack 2.1A/5V Dual Port USB Plug Power Adapter Charging Block Cube Charger Brick Box for iPhone 13/12/11/XR/XS/X/8/7/6 Plus 5/SE/4, iPad, Samsung, Moto, Android Phone | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wall Chargers | KENHAO | - |
| **target** | 2022-05-17 | **Caefax Compatible with iPhone 13 Case (2021), Crystal Clear Militrary Grade Protective Non-Yellowing Slim Thin with Double Shockproof Air Bags** | - | Caefax | - |

- Target `B09BYWWRCC`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 6,515); recency rank: >200.
- Recency top 3: (1) USB Wall Charger, LUOATIP 3-Pack 2.1A/5V Dual Port USB Cube Power Adapter Charger Plug Block Charging Box Brick for iPhone 13 12 11 Pro Max SE XS XR X 8 7 6 6S Plus, Samsung, LG, Moto, Android Phones [LUOATIP]; (2) USB Wall Charger, Power-7 5-Pack 2.1Amp Dual Port USB Cube Power Adapter Charger Plug Block Charging Box Brick for iPhone SE 11 Pro Max XS XR X 8 7 6S 6 Plus, Samsung, LG, Moto, Android Phones [power-7]; (3) USB Wall Charger, Charger Block, 3-Pack Charging Cube 1A/5V One-Port Charging Block Charger Box for iPhone 14 13 12 11 Pro Max, SE,XR/XS/X,8/7/6, iPad, Samsung Galaxy S22 S21 A13,Note 20, LG, Pixel 6 [VIFIGEN]
- Two-tower top 3: (1) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (2) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]; (3) TETHYS Glass Screen Protector Designed For iPhone 11 / iPhone XR (6.1") [Edge to Edge Coverage] Full Protection Durable Tempered Glass Compatible iPhone XR/11 [Guidance Frame Include] - Pack of 3 [TETHYS]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = unknown; relation = complement; leaf jump = unknown; top-level jump = unknown; second-level jump = unknown; store change = yes; target leaf unseen in history = unknown; price jump = unknown. Note: Universal wall charger -> iPhone 13 case; charger title lists iPhone 13 but is universal. Earlier history names iPhone 11/XR and iPhone 5: model change; target category metadata missing.

### F1 - instance 840982 (user 116293, history length 7)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-09-07 | iFuntecky CO-GPX-ARM-NV Caseology Parallax for Google Pixel Case (2016) - Award Winning Design - Navy Blue, 7 Pack | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Alcoco | - |
| h2 | 2019-02-10 | Motorola G6 – 32 GB – Unlocked (AT&T/Sprint/T-Mobile/Verizon) – Black - (U.S. Warranty) - PAAE0000US | Cell Phones & Accessories > Wireless Trade-In Buy Box Widget > Unlocked Wireless Trade-In | Motorola | 239.99 |
| h3 | 2021-02-01 | Anker USB C Power Bank, PowerCore Essential 20000 PD (18W) Power Bank, High Cell Capacity 20000mAh Portable Charger Battery Pack for iPhone 12/Mini/Pro/Max Pro/11/X, Samsung (PD Charger Not Included) | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Portable Power Banks | Anker | 64.95 |
| h4 | 2021-07-24 | Wepro Screen Protector Case Compatible with Fitbit Versa 2 Smartwatch, 3-Pack Soft TPU Full Cover Cases for Fitbit Versa 2 Watch, Black/Silver/Gold, | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cases | Wepro | 8.99 |
| h5 | 2021-07-24 | Google Wireless Charger Pixel 3, Pixel 3XL - White | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Charging Stations | Google | 24.89 |
| **target** | 2022-05-12 | **[2+2 Pack] for Google Pixel 6 Pro Screen Protector, 9H Tempered Glass, Ultrasonic Fingerprint Compatible, 3D Curved, HD Clear Scratch Resistant for Google Pixel 6 Pro 5G Glass Screen Protector** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Arshek | 11.99 |

- Target `B0C6DG5LZK`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 1,615); recency rank: >200.
- Recency top 3: (1) 3 Pack Screen Protector Compatible Fitbit Versa 2 Case, GHIJKL Ultra-Thin Slim Soft TPU Protective Case All-Around Full Cover Bumper Shell for Fitbit Versa 2 Smartwatch, Black, Silver, Blue [GHIJKL]; (2) Yolovie (3+1 Pack) Screen Protectors Charger (with Case On) Compatible with Fitbit Versa 2 Case Smart Watch TPU Bumper Shell Protector and Replacement Charging (Clear Silver Black) [Yolovie]; (3) Goton Compatible Fitbit Charge 4/3 / 3SE Screen Protector, (3 Packs) Soft TPU Shockproof Full Cover Case Screen Protector for Fitbit Charge 4 / Charge 3 / Charge 3 SE （Black & Clear & Silver） [Goton]
- Two-tower top 3: (1) amFilm Glass Screen Protector for Google Pixel 2 XL, Tempered Glass, 3D Curved [amFilm]; (2) OMOTON [3 Pack Screen Protector for Google Pixel 4XL/ 4 XL Tempered Glass/Easy Installation with Alignment Frame/Only Cover Display Area [OMOTON]; (3) LK 3 Pack Screen Protector Compatible for Google Pixel 3a [Not Fit for Google Pixel 3a XL], HD Transparent, Easy Frame Installation, Tempered Glass, Bubble Free, Anti-Scratch [LK]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = yes; price jump = down. Note: Pixel 3 wireless charger -> Pixel 6 Pro protector: same family, newer model. Two-tower top 3 are Pixel 2 XL/4 XL/3a protectors (right type and family, older models).

### F1 - instance 1411999 (user 195268, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-12-09 | BLU Studio X8 HD Case, Tough Hybrid + Dual Layer Shockproof Drop Protection Metallic Brushed Case Cover + Screen Protector for Studio X8 HD (S530) (VGC Black + SP) | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs | New Frontier Wireless Accessory | 9.99 |
| h2 | 2018-12-09 | BLU Studio X8 HD - 5.0" GSM Unlocked Smartphone -Black | Cell Phones & Accessories > Cell Phones | BLU | - |
| h3 | 2020-03-25 | (2 Pack) Supershieldz Designed for T-Mobile (Revvl 2) Tempered Glass Screen Protector, Anti Scratch, Bubble Free | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Supershieldz | 7.99 |
| h4 | 2021-04-30 | CABINAHOME Car Cell Phone Mount Dashboard Cell Phone Clip Automobile Cradles Car Holder Mount Stand Suitable for 3 to 7 inch Smartphones (Upgrade) | - | CABINAHOME | - |
| **target** | 2021-12-23 | **Red2Fire for iPhone 13 Case, [360 Full Body] [Heavy Duty Protection] with Built-in Screen Protector Tough Rugged 12 FT Dropproof Shockproof Dust-Proof Bumper Case for iPhone 13 6.1 inch Black** | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Red2Fire | 12.79 |

- Target `B0C7QS1G9P`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 22,751); recency rank: >200.
- Recency top 3: (1) Car mount Cell Phone Holder for Car, Clip Gps Display Desk Stand Smartphone for Iphone Samsung and Other Smartphones up to 5.7 Inc Lg Google Nexus Nokia Sony Motorola Htc Blackberry [Kraiovim]; (2) JunDa Car Phone Mount 360 Degree Rotation Dashboard Cell Phone Holder for Car Clip Mount Stand Suitable for 4 to 7 inch Smartphones… [DaJun]; (3) CHDFKKD Car Phone Mount, Cell Phone Holder for Car 360 Degree Rotation Car Dashboard Clip Mount Stand Suitable for 4 to 7 inch Smartphones (Black) [CHDFKKD]
- Two-tower top 3: (1) New bee Bluetooth Earpiece V5.0 Wireless Handsfree Headset 24 Hrs Driving Headset 60 Days Standby Time with Noise Cancelling Mic Headsetcase for iPhone Android Samsung Laptop Truck Driver, Gold [New bee]; (2) Bluetooth Earpiece V5.0 Wireless Handsfree Headset with Microphone Noise Cancelling 16 Hrs Talking 450 Hrs Standby Compatible with iPhone Android Samsung Cell Phone PC Laptop for Driver Trucker, Black [ZEEYU]; (3) Moto G6 with Alexa Hands-Free – 32 GB – Unlocked (AT&T/Sprint/T-Mobile/Verizon) – Oyster Blush - Prime Exclusive Phone [Motorola]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = unknown; top-level jump = unknown; second-level jump = unknown; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: Universal car mount -> iPhone 13 case; earlier history names BLU and T-Mobile Revvl phones, no iPhone. Last-item category missing; apparent move to a new phone brand.

### F1 - instance 1756859 (user 243219, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2016-08-02 | Tektide Case Compatible for Moto G Pure, [Invisible Armor] Xtreme Slim, Clear, Soft, Lightweight, Shock Absorbing TPU Bumper/Back Cover | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Tektide | 6.99 |
| h2 | 2018-06-14 | AmazonBasics 4.8 Amp 24W Dual USB Car Charger for Apple and Android Devices, Black and Red, 4-Pack | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Automobile Chargers | Amazon Basics | 29.28 |
| h3 | 2019-01-06 | Plinrise Animal Desk Phone Stand, Update Dinosaur Stripe Silicone Office Phone Holder, Creative Phone Tablet Stand Mounts, Size:1.3" X 3.1" X 2.8" (Stand 4Pcs) | Cell Phones & Accessories > Accessories > Stands | Z PLINRISE | - |
| h4 | 2019-08-29 | Google Pixel XL Case, Cruzerlite Bugdroid Circuit TPU Case for Google Pixel XL - Retail Packaging - Teal | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | CruzerLite | - |
| **target** | 2022-04-19 | **Caseology Vault Compatible with Google Pixel 5a Case 5G (2021) - Urban Gray** | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Caseology | 13.99 |

- Target `B08XBB6J47`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 1,087); recency rank: >200.
- Recency top 3: (1) Google Pixel XL Case, Cruzerlite Bugdroid Circuit TPU Case for Google Pixel XL - Retail Packaging - Black [CruzerLite]; (2) CruzerLite Google Pixel XL Case, Bugdroid Circuit TPU Case for Google Pixel XL - Retail Packaging - Blue [CruzerLite]; (3) Google Pixel XL Case, Cruzerlite Androidified A2 (TPU and Carbon) case for Google Pixel XL - Blue [CruzerLite]
- Two-tower top 3: (1) amFilm Glass Screen Protector for Google Pixel 2 XL, Tempered Glass, 3D Curved [amFilm]; (2) OMOTON [3 Pack Screen Protector for Google Pixel 4XL/ 4 XL Tempered Glass/Easy Installation with Alignment Frame/Only Cover Display Area [OMOTON]; (3) Spigen Rugged Armor Designed for Google Pixel 2 XL Case (2017) - Black [Spigen]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = yes; relation = substitute; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = unknown. Note: Pixel XL case -> Pixel 5a case (same leaf, 2.6-year gap). Recency top 3 are more Pixel XL cases: retriever echoes the old model; target is the newer model.

### F1 - instance 2347616 (user 324948, history length 5)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-08-05 | L K (3 Pack) Screen Protector for iPhone 7, iPhone 8 (4.7 inch) Tempered Glass (Easy Installation Tray) Case Friendly, 9H Hardness HD Clear, Bubble Free | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | L K | - |
| h2 | 2018-08-05 | Raptic Shield, iPhone SE 2022 case, Military Grade Drop Tested iPhone 7 8 Case, Shockproof Protective Clear Case, Durable Aluminum Frame Case for iPhone SE 2022/8/7, Rose Gold | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | X-Doria | - |
| h3 | 2019-05-27 | Purity Glass Screen Protector for iPhone 8 Plus / 7 Plus (3-Pack) [w/Installation Frame] Tempered Glass Screen Protector Compatible with Apple iPhone 8 Plus, iPhone 7 Plus [Case Friendly] (3 Pack) | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Purity | 6.99 |
| h4 | 2020-10-02 | VICSEED Universal Phone Mount for Car [Solid & Durable] Car Phone Holder Mount Dashboard Windshield Air Vent Long Arm Strong Suction Cell Phone Holder Car for iPhone 14 13 12 Pro Max All Mobile Phones | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | VICSEED | 28.97 |
| h5 | 2020-12-18 | TOZO W3 Mini Wireless Charger 10W Thinnest Fast Charging Pad with Aviation Aluminum Computer Numerical Control Technology Sleep-Friendly Sliver (NO AC Adapter) | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wireless Chargers | TOZO | 13.99 |
| **target** | 2022-03-07 | **Geoumy Magnetic Band Compatible with Apple Watch 38mm 40mm 41mm 42mm 44mm 45mm 49mm, Stainless Steel Mesh Milanese Strap, Metal Wristband for iWatch SE Series 8 7 6 5 4 3 2 1 for Women Men** | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | Geoumy | 10.99 |

- Target `B0BFL6HYPB`: new-in-Val; 0 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 10,874); recency rank: >200.
- Recency top 3: (1) TOZO W8 Wireless Car Charger Auto-attraction Magnets Fast Charging Air Vent Mount Compatible for iPhone 12/12 Mini/12 Pro/ 12 Pro Max NO AC Adapter (Compatible with Mag-Safe Charger Magnetic Charging) [TOZO]; (2) Wireless Car Charger, 15W/10W Fast Charging Car Mount, Air Vent Cell Phone Holder for Car, Universal Car Phone Holder Cradle Compatible with iPhone 13/12/11, Samsung S21/S20/S10, and More, Black [Clin]; (3) andobil Wireless Car Charger Mount, 1 Sec Auto-Clamping Fast Charging Phone Holder for Car Windshield Dashboard Air Vent, Compatible with iPhone Xs/Xs Max/XR/X/8,Samsung Galaxy S10/S10e/S9/S9+/S8 Etc [andobil]
- Two-tower top 3: (1) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]; (2) Yootech Wireless Charger,10W Max Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [yootech]; (3) Wireless Charger, 15W Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [COO]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = yes; price jump = similar. Note: Wireless charging pad (no device named) -> Apple Watch band; earlier history names iPhone 7/8. Smartwatch leaf never seen; new device category.

### F2 - instance 764303 (user 105662, history length 5)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-11-21 | [2 Pack] Galaxy Note 8 Tempered Glass Screen Protector, Loopilops [HD Clear][Anti-Bubble][Anti-Scratch][Anti-Fingerprint] Tempered Glass Screen Protector Samsung Galaxy Note 8 [Case Friendly] | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | ELYAN | - |
| h2 | 2019-01-11 | amFilm Glass Screen Protector for Samsung Galaxy Note 8, Full Screen Coverage, 3D Curved Tempered Glass, Dot Matrix with Easy Installation Tray (Black) | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | amFilm | 8.99 |
| h3 | 2020-02-20 | IQShield Screen Protector Compatible with Samsung Galaxy Watch 3 (45mm)(6-Pack) Anti-Bubble Clear Film | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Screen Protectors | IQShield | 9.95 |
| h4 | 2020-03-01 | Spigen Rugged Armor Designed for Google Pixel 2 Case (2017) - Black | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Spigen | - |
| h5 | 2022-02-01 | Aliruke Google Pixel 6 Pro Case, Google Pixel 6 XL Case with Finger Ring Holder Kickstand, Military Grade Stand Cover Phone Cases Compatible for Google Pixel6 Pro, Black | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Aliruke | - |
| **target** | 2022-04-04 | **Pofesun 2Pack 30W Fast Charger 3.0 Dual USB Wall Charger, QC 3.0 Charging Block Fast Charging Power Adapter Compatible for Galaxy S21 S20 S10 S9 S8 Note 10 9 iPhone iPad LG Pixel Tablets-Black** | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wall Chargers | Pofesun | 12.49 |

- Target `B082DVBVMK`: warm; 72 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 6,148); recency rank: >200.
- Recency top 3: (1) Aliruke Google Pixel 6 Case, Google Pixel6 Case with Finger Ring Holder Kickstand, Military Grade Stand Cover Phone Cases Compatible for Google Pixel 6, Black [Aliruke]; (2) Pixel 6 Pro Case, Google Pixel 6 Pro Case with 3D Curved HD Screen Protector, Androgate Military-Grade Ring Holder Kickstand 15ft Drop Tested Shockproof Cover Case for Pixel 6 Pro Black [Androgate]; (3) Wallet Case for Google Pixel 6 Pro Case with Card Holder Slot with Stand Kickstand Ring Slim Heavy Duty Defender Armor Military Grade Black [Kingshark]
- Two-tower top 3: (1) amFilm Glass Screen Protector for Google Pixel 2 XL, Tempered Glass, 3D Curved [amFilm]; (2) OMOTON [3 Pack Screen Protector for Google Pixel 4XL/ 4 XL Tempered Glass/Easy Installation with Alignment Frame/Only Cover Display Area [OMOTON]; (3) IQShield Screen Protector Compatible with Google Pixel 3a XL 6 inch (2-Pack)(Case Friendly) Anti-Bubble Clear Film [IQShield]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = unknown; relation = complement; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: Pixel 6 Pro case -> universal dual-USB wall charger whose title lists Pixel among many devices; generic accessory with no device anchor.

### F2 - instance 1491415 (user 206272, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-11-20 | Mr.Shield [3-PACK] Designed For Samsung Galaxy J7 [Newest 2017] [Tempered Glass] Screen Protector [Japan Glass With 9H Hardness] with Lifetime Replacement | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Mr.Shield | 6.95 |
| h2 | 2020-11-14 | Lamicall Phone Holder Bed Gooseneck Mount Cell Phone Clamp Clip for Desk, Flexible Lazy Long Arm Headboard Bedside, Overhead Mount Stand, Compatible with Phone 12 Mini 11 Pro Xs Max XR X 8 7 6 Plus | Cell Phones & Accessories > Accessories > Stands | Lamicall | 26.99 |
| h3 | 2020-11-23 | Case-Mate Elastic Strap Phone Grip - Finger & Hand Phone Holder, Mobile & Tablet Phone Strap - For iPhone 14 Pro Max/ 13 Pro Max/ 12 Pro Max/ 11/ S23 Ultra/ Pixel 7, 3M Supreme Adhesive - Pink Glitter | Cell Phones & Accessories > Accessories > Grips | Case-Mate | - |
| h4 | 2021-12-17 | USB C Female to USB Male Adapter (4-Pack),Type C to USB A Charger Cable Adapter for iPhone 11 12 13 14 Pro Max,Samsung Galaxy S23 S22 S21 S20 Ultra,Google Pixel 4 3 2,Apple iWatch Watch Series 7 8 SE | Cell Phones & Accessories > Accessories > Cables & Adapters > OTG Adapters | QianLink | 6.99 |
| **target** | 2021-12-21 | **JXMOX USB Type C Adapter, (4-Pack) Micro USB Female to USB C Male Convert Connector Fast Charging Compatible with Samsung Galaxy S10 S9 S8 Plus,Note 9 8,LG V35 V30 G7,Nexus 6P 5X,Moto Z2 Z3(Red)** | Cell Phones & Accessories > Accessories > Cables & Adapters > OTG Adapters | JXMOX | 7.99 |

- Target `B07MVH52FG`: warm; 200 reviews before T1.
- Two-tower (text) rank: 224 (full-catalogue 224); recency rank: >200.
- Recency top 3: (1) USB C Female to USB Male Adapter 4-Pack,JXMOX Type C to USB A Charger Cable Adapter for iPhone 11 12 13 14 Pro Max,Samsung Galaxy Note 10 S22 S21 S20 Ultra,Google Pixel 5 4 3 2 XL,iPad Air 4 5 Mini 6 [JXMOX]; (2) USB C Female to USB Male Adapter (8 Pack),Type C to USB A Charger Cable Adapter for iPhone 11 12 13 Pro Max,Samsung Galaxy Note 10 S20 Plus 20 S20+ 20+ Ultra,Google Pixel 5 4 4a 3 3A 2 XL [IIABC]; (3) USB C Female to USB Male Adapter (4-Pack),Type C Charging Cord Connect USB A Charger for iPhone 14 13 12 11 Pro Max,iPad Pro Air 4 5 Mini 6,Samsung Galaxy S20 S21 S22 Plus,Google Pixel 5 4 XL(White) [QianLink]
- Two-tower top 3: (1) AMINY Bluetooth Headset,Wireless Bluetooth Earpiece Compatible with iPhone/Android Cell Phones,Auriculares Bluetooth Earpiece 28 Hrs Talking Time V5.3 Wireless Headset,Button Depress Version [AMINY]; (2) Wireless Charger, 15W Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [COO]; (3) P&LLL Bluetooth Mini Single Earbud Wireless Invisible Pro Headset 8Hrs Playtime Noise Smallest in-Ear Noise Cell Phone with Hands-Free Earpiece for iPhone Samsung Android Car Mic Earphone [P&LLL]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = substitute; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = similar. Note: USB-C(F)-to-USB-A adapter -> Micro-USB(F)-to-USB-C adapter, 4 days later, same leaf. Near miss: two-tower rank 224; recency top 1 is a sibling product of the target store (JXMOX).

### F2 - instance 1532879 (user 212050, history length 7)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2019-12-15 | Casetego Compatible with Galaxy S8 Plus Case,Floral Three Layer Heavy Duty Hybrid Sturdy Shockproof Full Body Protective Cover Case for Samsung Galaxy S8 Plus,Blue Flower | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Casetego | 12.99 |
| h2 | 2020-09-23 | EZCO 4-Pack Screen Protector Compatible with Fitbit Versa & Versa Lite (Not for Versa 2), Waterproof Tempered Glass Screen Protector Cover Saver for Versa Smart Watch Scratch Resist Anti-Bubble | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Screen Protectors | EZCO | - |
| h3 | 2020-10-07 | QIBOX Charger Compatible with Fitbit Versa/Versa Lite (Not for Versa 2/3), Replacement USB Charging Cable Dock Stand for Versa/Versa Lite/Versa SE Smartwatch, 2-Pack 3Ft Sturdy Power Charging Cord | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cables & Chargers | QIBOX | - |
| h4 | 2020-10-07 | LK Screen Protector Compatible with Samsung Galaxy S8 Plus Tempered Glass, Case Friendly, Alignment Frame Easy Installation, 3D Curved Full Coverage | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | LK | - |
| h5 | 2021-06-13 | ICEDIO for iPhone 12 Pro Max Case with Screen Protector,Clear with Fashion Leopard Patterns for Girls Women,Shockproof Slim Fit TPU Cover Protective Phone Case for iPhone 12 Pro Max 6.7" | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | ICEDIO | 15.89 |
| **target** | 2022-01-12 | **OHCBOOGIE Stretchy Solo Loop Strap Compatible with Apple Watch Bands 38mm 40mm 42mm 44mm 45mm 49mm 49mm ,Adjustable Stretch Braided Sport Elastics Weave Nylon Women Men Wristband Compatible with iWatch Series 8/7/6/5/4/3/2/1 SE Ultra** | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | OHCBOOGIE | 9.99 |

- Target `B0BZ3LP5P4`: warm; 133 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 31,102); recency rank: >200.
- Recency top 3: (1) ICEDIO iPhone 11 Pro Case with Screen Protector,Clear with Cute Flower Garden Patterns for Girls Women,Shockproof Slim Fit TPU Cover Protective Phone Case for Apple iPhone 11 Pro 5.8 Inch [ICEDIO]; (2) ICEDIO iPhone Xs Max Case with Screen Protector,Clear with Fashion Floral Designs for Girls Women,Shockproof Slim Fit TPU Cover Protective Phone Case for iPhone Xs Max Beautiful Blossoms [ICEDIO]; (3) ICEDIO iPhone 13 Case with Screen Protector,Slim Fit Crystal Clear Cover with Fashionable Designs for Girls Women,Protective Phone Case for iPhone 13 6.1" Leopard Patterns [ICEDIO]
- Two-tower top 3: (1) IQShield Screen Protector Compatible with Samsung Galaxy S8 (2-Pack)(Case Friendly)(Not Glass) Anti-Bubble Clear Film [IQShield]; (2) [2 Pack] ArmorSuit MilitaryShield Screen Protector Designed for Samsung Galaxy S22 Ultra 6.8" Case Friendly (2022 Release) Anti-Bubble HD Clear Film - Made in USA [ArmorSuit]; (3) Ailun Screen Protector Compatible for iPhone 8 plus,7 Plus,6s Plus,6 Plus, 5.5 Inch 3Pack Case Friendly Tempered Glass [Ailun]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = similar. Note: Apple brand only (iPhone 12 Pro Max case -> Apple Watch band); history also has Fitbit Versa accessories. Smartwatch-band leaf absent from history; new device or replacement watch.

### F2 - instance 1732724 (user 239866, history length 2)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2017-05-04 | Gocomcom Honbay Belt Clip Holster for Otterbox Defender Case for Samsung Galaxy Note 9 / N960U Black+ Fishbone Cable Winder | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Gocomcom | - |
| h2 | 2017-05-04 | Bike Mount, GreatShield Clip-Grip [Upgrade Version - Elastic Secure Strap] Handlebar Bike Mount Holder for iPhones, Samsung Galaxy, LG, BlackBerry, HTC Smartphones, GPS Devices and More | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | GreatShield | - |
| **target** | 2021-10-06 | **Cell Phone Clip on Stand Holder - with Grip Flexible Long Arm Gooseneck Bracket Mount Clamp for iPhone X/8/7/6/6s/5 Samsung S8/S7, used for bed, desktop, Black** | Cell Phones & Accessories > Accessories > Stands | MAGIPEA | 12.7 |

- Target `B099J3K1LJ`: warm; 287 reviews before T1.
- Two-tower (text) rank: 323 (full-catalogue 323); recency rank: >200.
- Recency top 3: (1) Handstands Clip Belt Holster for Smartphones [Gadget Grips]; (2) Phone Belt Clip, Lovphone Universal Holder with Quick Mount for iPhone 14/13/12/11,14/13/12/11 Pro,14/13/12/11 Pro Max,XR,XS Max,X,7,8 & Samsung Galaxy S23/S22/S21 Ultra, Plus [LOVPHONE]; (3) Honbay Samsung Galaxy Note 4 N9100 Replacement Belt Clip for Otterbox Defender Case [Gocomcom]
- Two-tower top 3: (1) amFilm Screen Protector for Samsung Galaxy Note 9, Full Screen Coverage Screen Protector, 3D Curved Tempered Glass, Dot Matrix with Easy Installation Tray (Black), 2 Pack [amFilm]; (2) SUPCASE Unicorn Beetle PRO Series Phone Case for Samsung Galaxy Note 9, Full-Body Rugged Holster Case with Built-in Screen Protector for Samsung Galaxy Note 9 2018 (Pink) [SUPCASE]; (3) amFilm (3 Pack) Glass Screen Protector for Samsung Galaxy Note 9, Full Screen Coverage Screen Protector, 3D Curved Tempered Glass, Dot Matrix with Easy Installation Tray [amFilm]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h2 (model-assisted, needs human review):** same device ecosystem = unknown; relation = substitute; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: Universal bike mount -> universal gooseneck clip stand (both phone holders, different leaf), 4.4-year gap; two-tower rank 323.

### F2 - instance 1837606 (user 254445, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-12-08 | YOUMAKER Galaxy S9 Case,Metallic Purple with Built-in Screen Protector Heavy Duty Protection Shockproof Slim Fit Full Body Case Cover for Samsung Galaxy S9 5.8 inch - Purple/Black | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | YOUMAKER | - |
| h2 | 2018-12-08 | SUPCASE Unicorn Beetle Pro Series Case Designed for Galaxy S9, with Built-in Screen Protector Full-Body Rugged Holster Case for Galaxy S9 (2018 Release) (Royal Blue) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | SUPCASE | 22.99 |
| h3 | 2018-12-08 | Macally Windshield Phone Mount for Car Magnetic - Suction Cup Window Mount Phone Holder with 12" Long Gooseneck Arm & Super Strong Magnet Mount for Cell Phone, iPhone, Smartphone | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | Macally | 17.99 |
| h4 | 2019-04-03 | Fotopro UFO2 Flexible Tripod, 12" Phone Camera Tripod with Smartphone Mount and Bluetooth Remote Shutter, 360 Ball Head Tripod for iPhone, Samsung, Huawei(White) | Cell Phones & Accessories > Accessories > Photo & Video Accessories > Tripods | Fotopro | 29.99 |
| **target** | 2021-12-23 | **Magnetic Mount, WizGear Stick On (2 Pack) Dashboard Magnetic Car Mount Holder, for Cell Phones and Mini Tablets with Fast Swift-snap Technology, Magnetic Cell Phone Mount** | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | WixGear | 14.89 |

- Target `B0773F8S74`: warm; 672 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 524); recency rank: >200.
- Recency top 3: (1) Flexible Cell Phone Tripod and Bluetooth Remote Control Camera Shutter - for iPhone, Samsung Galaxy and Many More - Octopus Style Mount - Universal Phone Holder - Solution for Smartphone Photography [CamKix]; (2) Phone Tripod,PEYOU 50" Aluminum Camera Tripod + Wireless Remote Shutter + 360°Rotation Smartphone Holder Mount Compatible for iPhone Xs Max XR X 8 7 6 Plus,for Samsung Galaxy Note 9 8 S10 S9 S8 Plus [PEYOU]; (3) 42inch Remote Controller Shutter + Portable Camera Tripod Stand Holder Adjustable Rotatable Retractable Aluminum Tripods Smartphones Mount for iPhone X XS 8 7 7 11 Pro Max Samsung S9 S8 S10 OnePlus [Dogxiong]
- Two-tower top 3: (1) Wireless Charger, 15W Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [COO]; (2) [2 Pack] ArmorSuit MilitaryShield Screen Protector Designed for Samsung Galaxy S22 Ultra 6.8" Case Friendly (2022 Release) Anti-Bubble HD Clear Film - Made in USA [ArmorSuit]; (3) Anker 2 Pack 313 Wireless Charger (Stand), Qi-Certified for iPhone 14/14 Pro/14 Pro Max/13/13 Pro Max, 10W Fast-Charging Galaxy S20, S10 (No AC Adapter) [Anker]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = down. Note: Tripod -> magnetic dashboard car mount (2.7-year gap); same leaf as h3, a 2018 car mount: repeat of an older need rather than the last product. Two-tower full-catalogue rank 524.

### F2 - instance 1997091 (user 276426, history length 3)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2022-03-18 | Goton 3 in 1 Accessories for Samsung Galaxy Watch 4 Classic 42mm, 1 Rugged TPU Armor Bumper Case Cover +2 Tempered Glass Screen Protector Films + 1 Bezel Ring for Galaxy Watch4 Classic 42mm Black | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cases | Goton | 11.99 |
| h2 | 2022-03-18 | LYWHL 2 Pack for Samsung Galaxy S9 / S8 Screen Protector, [Case Friendly] Tempered Glass HD Clear 9H Hardness 3D Curved Film ONLY for Galaxy S9 / Galaxy S8, Anti-Scratch Bubble Free | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | LYWHL | 8.88 |
| h3 | 2022-03-18 | Leather Bands Compatible with Samsung Galaxy Watch 5 pro Band 45mm Galaxy Watch 5 Band 40mm 44mm Galaxy Watch 4 Band 2021 Galaxy Watch 4 Classic Band 42mm 46mm Men Women | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | HATALKIN | 16.88 |
| **target** | 2022-06-20 | **SURBUID Galaxy S10 Plus Privacy Screen Protector, Tempered Glass [Not for Galaxy S10] [Does Not Support Fingerprint Unlock] [Edge Glue] Case Friendly 3D Touch Screen Protector for Galaxy S10 Plus** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | SURBUID | 9.99 |

- Target `B07Q78V8BC`: warm; 67 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 4,606); recency rank: >200.
- Recency top 3: (1) Goton 3 in 1 Accessories for Samsung Galaxy Watch 3 41mm, 1 Rugged TPU Armor Bumper Case +2 Tempered Glass Screen Protector Films + 1 Bezel Ring for Galaxy Watch 3 41mm(Black,41mm) [Goton]; (2) Goton for Samsung Galaxy Watch 5 & 4 40mm Screen Protector Case, Bling Crystal Diamonds Hard PC Protective Bumper Cover with Tempered Glass Film Accessories for Galaxy Watch 5 & 4 [Goton]; (3) Goton [3 Pack] Bling Case Compatible with Samsung Galaxy Watch 4 Screen Protector 40mm Accessories, Crystal Diamond Protective Cover with Tempered Glass Film for Watch4 40 mm Black+Pink+White [Goton]
- Two-tower top 3: (1) (2 Pack) Supershieldz Designed for Samsung Galaxy Watch 3 (41mm) / Galaxy Watch 4 Classic (42mm) / Galaxy Watch (42mm) Tempered Glass Screen Protector, Anti Scratch, Bubble Free [Supershieldz]; (2) OMOTON Tempered Glass Screen Protector Compatible Samsung Galaxy Watch 46mm (2018) & Gear S3 [3 Pack] [OMOTON]; (3) Diruite 4-Pack for Samsung Galaxy Watch 46mm (2018) / Gear S3 Screen Protector Tempered Glass, [2.5D 9H Hardness] [Anti-Scratch] (DO NOT Fit for Galaxy Watch 4 Classic 46mm) [Diruite]
- Same user's later reviews within 30 days (context only, Val period only): 2022-06-20: S10 Plus Case,Galaxy S10 Plus Case Carbon Fiber,DUEDUE Dual Layer Slim Hybrid Shock Absorbing Cover Hard PC Bumper Rugged Back Case for Samsung Galaxy S10 Plus/S10+ 6.4'' for Men and Boys(2019), Black [DUEDUE]
- **Rubric vs h3 (model-assisted, needs human review):** same device ecosystem = yes; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = similar. Note: Samsung brand only (Galaxy Watch band -> Galaxy S10 Plus privacy protector). A Galaxy S10 Plus case review the same day suggests a newly acquired S10 Plus (context only).

### F2 - instance 2244325 (user 310605, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2013-09-09 | Rose Pink White Zebra Combo Hard Soft High Impact iPhone 4 4S Armor Case Skin Gel with Free Screen Protector | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | UIYTRAESTING | - |
| h2 | 2013-09-09 | ZuGadgets Magenta Leather Flip Wallet Case Holder Cover Pouch with Credit & ID Card Slot for Samsung Galaxy S III S3 i9300 / SC-06D + Free Screen Protector (7484-5) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | ZuGadgets (Samsung Galaxy S3) | - |
| h3 | 2020-07-14 | TOPGO Cup Holder Phone Mount for car, [Upgraded 3.0] Cup Holder Phone Holder [Ultra-Sturdy & Non-Shaking] Cell Phone Automobile Cradles for iPhone 14, Samsung and More Smartphone(Black) | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | TOPGO | 23.99 |
| h4 | 2021-05-11 | Wall Charger, USB Charger Adapter, 3.1A/3Pack Muti Port Fast Charging Station Power Charge Base Block Plug for iPhone 14 13 12 Pro/SE/11Pro Max/X/8/7 Plus, Samsung S21/S10/S9/S8, Kindle Fire USB Plug | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wall Chargers | AILKIN | 13.99 |
| **target** | 2022-06-26 | **Stylus Pens for Touch Screens, StylusHome 10 Pack High Precision Capacitive Stylus for iPad iPhone Tablets Samsung Galaxy All Universal Touch Screen Devices** | Cell Phones & Accessories > Accessories > Stylus Pens | STYLUSHOME | 6.99 |

- Target `B08DCMM8WC`: warm; 134 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 9,086); recency rank: >200.
- Recency top 3: (1) USB Charger Plug,Wall Charger,Charging Block,5-Pack 2.1A/5V Portable Power Cube Charger Adapter Compatible for iPhone 14/13/12/11 Pro Max/Xs Max/XR/X/8/7/6S/6 Plus,Samsung Galaxy S23 S22 S21,LG,Moto [CIQILY]; (2) USB Wall Charger,Charging Block,3Pack 2.1A 2 Port Wall Plug Travel Power Adapter Fast Charger Box Compatible with iPhone 14/13/12 Mini/11 Pro Max/SE/Xs Max/XR/X/8/7/6 Plus,Samsung Galaxy S21 S20 Ultra [CIQILY]; (3) USB Wall Charger,Charger Adapter Charger Block,Double Wall Charger Plug 3Pack 2.1A Dual Port Cube USB Power Adapter Compatible for iPhone 14/13/12/8/7/6 Plus/X,iPad,Samsung Galaxy S5 S6 S7 Edge,Kindle [CIQILY]
- Two-tower top 3: (1) Wireless Charger, 15W Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [COO]; (2) Yootech Wireless Charger,10W Max Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [yootech]; (3) THREEKEY 2 Pack Wireless Charger,10W Wireless Charging Pad Compatible with iPhone iPhone 14/14 Pro/14 Plus/14 Pro Max/13/13 Mini 13 Pro/13 Pro Max/Samsung Note,Galaxy Series [THREEKEY]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = yes; price jump = down. Note: Multi-port wall charger -> universal stylus pack, 13-month gap; generic accessory, stylus leaf never seen.

### F2 - instance 2572126 (user 356052, history length 16)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2020-11-15 | EGV 3 Pack Screen Protector for iPhone 14 Plus/iPhone 13 Pro Max 6.7-inch, 9H Hardness iPhone14 Plus Tempered Glass Film With Align Frame, Protective Glass Cover Screen, Bubble Free | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | EGV | 2.99 |
| h2 | 2020-11-20 | [2 Pack] Penom Compatible for Apple Watch Ultra Case 49mm Bumper Cover, Soft TPU Shockproof Protective Scratch-Resistant Frame for iWatch Series 8 Ultra 49 MM [No Screen Protector]- Clear | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cases | Penom | 8.99 |
| h3 | 2021-11-08 | Stretchy Solo Loop Nylon Bands Compatible with Apple Watch Band 38mm 40mm 41mm 42mm 44mm 45mm 49mm, Braided Elastic Wristbands Women Men Straps for iWatch Series Ultra/8/7/6/5/4/3/2/1/SE | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | CCnutri | 10.99 |
| h4 | 2022-01-12 | elago W2 Charger Stand Compatible with Apple Watch Series Ultra/8/SE2/7/6/SE/5/4/3/2/1 (49mm, 45mm, 44mm, 42mm, 41mm, 40mm, 38mm), Durable Silicone, Compatible with Nightstand Mode (Red) | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cables & Chargers | elago | 10.99 |
| h5 | 2022-04-08 | Aliruke Moto One 5G Ace/Moto G 5G Case with Tempered Glass Screen Protector and Grip Ring Kickstand, Military Grade Cover Magnetic Finger Loop Stand Phone Cases for Motorola Moto One 5G Ace, Red | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Aliruke | - |
| **target** | 2022-05-20 | **[2 Pack] ArmorSuit MilitaryShield Screen Protector Designed for Samsung Galaxy S22 Ultra 6.8" Case Friendly (2022 Release) Anti-Bubble HD Clear Film - Made in USA** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | ArmorSuit | 9.95 |

- Target `B0C3TFVNJ1`: warm; 937 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 889); recency rank: >200.
- Recency top 3: (1) Aliruke Moto One 5G Ace/Moto G 5G Case with Tempered Glass Screen Protector and Grip Ring Kickstand, Military Grade Cover Magnetic Finger Loop Stand Phone Cases for Motorola Moto One 5G Ace, Blue [Aliruke]; (2) Aliruke Case for Moto One 5G/Moto One 5G UW/Moto G 5G Plus Case with Tempered Glass Screen Protector,Military Grade Cover Grip Ring Kickstand Protective Phone Cases for Motorola Moto One 5G UW, Black [Aliruke]; (3) Aliruke Case for Moto One 5G/Moto One 5G UW/Moto G 5G Plus Case with Tempered Glass Screen Protector,Slim Shockproof TPU Bumper Cover Flexible Protective Phone Cases for Motorola Moto One 5G UW, Red [Aliruke]
- Two-tower top 3: (1) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]; (2) Spigen Tempered Glass Screen Protector [GlasTR EZ FIT] designed for iPhone 11 / iPhone XR [6.1 inch] [Case Friendly] - 2 Pack [Spigen]; (3) Wireless Charger, 15W Fast Wireless Charging Pad Compatible with iPhone 14/14 Plus/14 Pro/14 Pro Max/13/13 Mini/SE 2022/12/11/X/8,Samsung Galaxy S22/S21/S20,AirPods Pro 2(No AC Adapter) [COO]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = no; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = no; price jump = unknown. Note: Motorola One 5G Ace case -> Galaxy S22 Ultra film protector; user history spans iPhone, Apple Watch and Motorola. Some history titles name later models than their review dates (crawl-time titles).

### F3 - instance 62142 (user 8618, history length 3)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2015-02-26 | BlastCase iPod Shuffle USB Charger SYNC Cable 3rd, 4th and 5th (Length May Vary) | - | Generic | - |
| h2 | 2021-01-06 | for Applewatch Charger Apple i Watch Charger,Long Apple Watch Charging USB C 2 Pack 6Ft 2-in-1 iPhone and iWatch Cable for Apple Watch Series 8/7/6/SE/5/4, Type-C Travel Block for iPhone 14/13/12/11 | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Cables & Chargers | TUMABER | 24.99 |
| h3 | 2021-07-11 | Mr Shield [Tempered Glass] Screen Protector For iPhone 6 / iPhone 6S / iPhone 7 / iPhone 8 [3-Pack] Screen Protector | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Mr.Shield | 5.95 |
| **target** | 2022-01-24 | **NTG Shockproof Designed for iPhone SE 2022 Case/iPhone SE 2020 Case, Heavy-Duty Tough Rugged Lightweight Slim Protective Case for iPhone SE/8/7-Black** | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | NTG | 7.99 |

- Target `B08W33NH74`: warm; 12 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 4,557); recency rank: >200.
- Recency top 3: (1) Mr.Shield [5-PACK] Designed For iPhone 6 / iPhone 6S [Tempered Glass] Screen Protector with Lifetime Replacement [Mr.Shield]; (2) Mr.Shield [3-PACK] Designed For iPhone 6 Plus/iPhone 6S Plus [Tempered Glass] Screen Protector with Lifetime Replacement [Mr.Shield]; (3) Mr.Shield [5-PACK] Designed For iPhone 8 / iPhone 7 [Tempered Glass] Screen Protector with Lifetime Replacement [Mr.Shield]
- Two-tower top 3: (1) amFilm Screen Protector 4.7“ for Apple iPhone 8, iPhone 7, iPhone 6S and iPhone 6, Tempered Glass 4.7“, 2 Pack [amFilm]; (2) Ailun Screen Protector Compatible for iPhone 8 plus,7 Plus,6s Plus,6 Plus, 5.5 Inch 3Pack Case Friendly Tempered Glass [Ailun]; (3) [2 Pack] Julk 49mm Case Compatible for 2022 New Apple Watch Ultra Screen Protector, [Full Coverage] HD TPU Ultra-Thin Protective Cover for iWatch, Transparent [JULK]
- Same user's later reviews within 30 days (context only, Val period only): 2022-01-24: NTG [1st Generation Designed for iPhone 11 Case, Heavy-Duty Tough Rugged Lightweight Slim Shockproof Protective Case for iPhone 11 6.1 Inch, Grey [NTG]
- **Rubric vs h3 (model-assisted, needs human review):** same device ecosystem = yes; relation = complement; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = similar. Note: iPhone 6/6S/7/8 protector -> iPhone SE/8/7 case; models 7 and 8 named in both titles (not a compatibility check). Same-day NTG iPhone 11 case suggests multiple phones (context only).

### F3 - instance 219280 (user 30338, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2013-01-03 | Krusell Luna 3XL Leather Universal Phone Case for iPhone SE/5s/5, 4s/4, and Smartphones w/ 4.3/4.88-in Screens | Cell Phones & Accessories > Accessories | Krusell | 8.62 |
| h2 | 2013-01-03 | Armor Suit 2 Pack Screen Protector Designed for Samsung Galaxy 5.1 Inch Max Coverage MilitaryShield HD Clear Film - Made in USA | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | ArmorSuit | - |
| h3 | 2013-12-10 | 5-Pack TruGlide Mesh Fiber Stylus with Microfiber Knit Tip for All Capacitive Touch Screen Tablets, iPad, and Smartphone (Black with Gold Clip, Blue, Green, Pink, and Purple with Silver Clip), Lynktec | Cell Phones & Accessories > Accessories > Stylus Pens | Lynktec | - |
| h4 | 2013-12-18 | LG LG500G Tracfone with Double Minutes for Life | Cell Phones & Accessories > TracFone | LG | - |
| **target** | 2021-08-19 | **Gentlestache Leather Cell Phone Holster for Belt,Phone Case Leather, Belt Cell Phone Holder, Leather Phone Pouch, Leather Phone Sheath ,iPhone 12, 12 Pro Max,11 Pro Max, XS Max, 8 Plus, Galaxy S20** | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Gentlestache | 22.99 |

- Target `B08SQJRT55`: warm; 3 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 9,005); recency rank: >200.
- Recency top 3: (1) Tracfone LG LG800G with Double Minutes for Life!! [LG]; (2) Truglide Duo Stylus Pen with Fiber Mesh Tip and Ballpoint Pen for iPad, iPhone, Tablet, Smartphone, and Touchscreen Devices (Blue) [Lynktec]; (3) LG 505C Prepaid Phone With Triple Minutes (Tracfone) [LG]
- Two-tower top 3: (1) LG 840G Prepaid Phone with Triple Minutes (Tracfone) (White) [LG]; (2) Bargains Depot (2 Pcs) [New Upgraded][0.18-inch Small Tip Series] 2-in-1 Stylus/Styli 5.5-inch L with 10 Replacement Rubber Tips -Black/Pink [Bargains Depot]; (3) Bargains Depot Capacitive Stylus/Styli 2-in-1 Universal Stylus Pens for All Touch Screen Tablets/Cell Phones with 20 Extra Replaceable Soft Rubber Tips (4 Pieces, Black/Red/Silver/Blue) [Bargains Depot]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: 2013 LG Tracfone -> universal leather belt holster (7.7-year gap since last review). Target title lists many phones; no device anchor.

### F3 - instance 244763 (user 33864, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2016-08-08 | Anker PowerCore 13000 Portable Charger - Compact 13000mAh 2-Port Ultra Portable Phone Charger Power Bank with PowerIQ and VoltageBoost Technology for iPhone, iPad, Samsung Galaxy (Black) | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Portable Power Banks | Anker | 29.99 |
| h2 | 2018-02-28 | PowerBear Micro USB Cable 6ft [2 Pack] Gold Connectors & Braided Nylon / Android Charger Cable | Cell Phones & Accessories > Accessories > Cables & Adapters | PowerBear | - |
| h3 | 2018-12-16 | Camere Lens Protector (iPhone 11 Mini) | - | UniqueMeINGLE | - |
| h4 | 2021-08-22 | ASDAWN BL-T39 for LG G7 Battery Replacement, for LG G7 ThinQ Battery G710 LMG710VM Verizon LMQ610 Q7 Plus ThinQ G710EM G710N EAC63958401 with Repair Tools + Installation Instruction | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Replacement Parts > Batteries | ASDAWN | 16.96 |
| **target** | 2021-12-26 | **[430mAh] Battery for Samsung Gear S3 Frontier(SM-R760) and Gear S3 Classic, (2023 New Version) Upgrade Replacement Battery for SM-R770, BR760, R765, EB-BR760ABE, GH43-04699A with DIY Repair Tool Kits** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Replacement Parts > Batteries | SHENMZ | 12.95 |

- Target `B083968PLS`: warm; 23 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 18,564); recency rank: >200.
- Recency top 3: (1) 3900mAh BL-T39 for LG G7 Battery Replacement, Battery for LG G7 ThinQ G710 G710EM G710N LMG710VM Verizon LMQ610 Q7 Plus with Full Repair Tools [SHENMZ]; (2) ASDAWN BL-T37 for LG Stylo 4 Battery Replacement, for LG V40 ThinQ LM-V405UA and LG Stylo 4 Q710 Q710AL Q710TS Q710MS Q710CS Q710US EAC63958201 with Repair Tool Parts + Installation Instruction [ASDAWN]; (3) ASDAWN BL-T32 for LG G6 Battery Replacement, for LG G6 LS993 VS998 H870 H871 H872 EAC63438701 with Repair Tool Kit + Installation Instruction [ASDAWN]
- Two-tower top 3: (1) Anker 2 Pack 313 Wireless Charger (Stand), Qi-Certified for iPhone 14/14 Pro/14 Pro Max/13/13 Pro Max, 10W Fast-Charging Galaxy S20, S10 (No AC Adapter) [Anker]; (2) Spigen Rugged Armor Designed for LG G7 ThinQ Case (2018) - Black [Spigen]; (3) LG G6 – 32 GB – Unlocked (AT&T/T-Mobile/Verizon) – Black – Prime Exclusive [LG]
- Same user's later reviews within 30 days (context only, Val period only): 2022-01-03: NEWDERY 2 in 1 Wireless Charger Stand Station for Samsung Galaxy Watch and Buds, Fit for Galaxy Watch 3, Active 2, 1/Gear S3, Galaxy Buds/Buds +/Live/Pro-Black [NEWDERY]
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = no; relation = substitute; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = similar. Note: LG G7 battery -> Samsung Gear S3 watch battery (same leaf, different device). Recency top 1 is the target store SHENMZ LG G7 battery; a later Gear S3 charger review suggests the user owns a Gear S3 (context only).

### F3 - instance 482717 (user 66746, history length 2)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2021-12-12 | CASEKOO Crystal Clear for iPhone 14 Case & iPhone 13 Case, [Not Yellowing] [Military Grade Drop Protection] Shockproof Protective Phone Case 6.1 inch 2022 Purple | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | CASEKOO | 18.99 |
| h2 | 2021-12-12 | omee Stretchy Solo Loop Compatible with Apple Watch Band 38mm 40mm 41mm 42mm 44mm 45mm 49mm, Breathable Trail Loop+Alpine Loop+Ocean Band for iWatch Serie 8 Ultra 7 6 SE 5 4 3 2 1, 8 Pack | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | omee | 15.99 |
| **target** | 2021-12-12 | **EloBeth Decorative Rings Loops Compatible with Apple Watch Bands Charms Series 8/7 41mm 45mm/Series 6/SE2/SE/5/4 44mm 40mm/Series 3/2/1 42mm 38mm Watchband Decorative Charms Accessories (No Band)** | Cell Phones & Accessories > Accessories > Smartwatch Accessories > Smartwatch Bands | EloBeth | 12.99 |

- Target `B07G9MXXTL`: warm; 27 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 28,427); recency rank: >200.
- Recency top 3: (1) CASEKOO Crystal Clear Designed for iPhone 11 Pro Max Case, [Not Yellowing] [Military Grade Drop Tested] Shockproof Protective Phone Cases Slim Thin Cover (6.5 inch) 2019, Clear [CASEKOO]; (2) CASEKOO iPhone 13 Mini Clear case, [Not Yellowing] [Military Drop Protection] Shockproof Protective Phone Case Slim Cover 5.4 inch 2021, Clear [CASEKOO]; (3) FRESHCLOUD Stretchy Solo Loop Compatible with Apple Watch Band 45mm 44mm 42mm 41mm 40mm 38mm, Sports Watch Bands for iWatch Series 8 7 6 5 for Women Men, Silicone No Buckles and Clasps - Wine Red… [FRESHCLOUD]
- Two-tower top 3: (1) [2 Pack] Julk 49mm Case Compatible for 2022 New Apple Watch Ultra Screen Protector, [Full Coverage] HD TPU Ultra-Thin Protective Cover for iWatch, Transparent [JULK]; (2) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (3) Mkeke Compatible with iPhone 11 Screen Protector for iPhone XR Screen Protector, Tempered Glass Film for Apple iPhone 11 and iPhone XR, 3-Pack Clear [Mkeke]
- Same user's later reviews within 30 days (context only, Val period only): 2021-12-12: Cuteey 12 Pack Case for Apple Watch Series 8 7 45mm Tempered Glass Screen Protector, All Round Full Hard PC Leopard Pattern Cover Bumper for iWatch 8 7 Accessories [Cuteey]; 2021-12-31: [8 PACK] Bands Compatible with Apple Watch Bands 45mm 44mm 42mm 41mm 40mm 38mm for Women Men, Replacement Strap with Classic Buckle for iWatch Series SE 8 7 6 5 4 3 2 1 [Vancle]
- **Rubric vs h2 (model-assisted, needs human review):** same device ecosystem = yes; relation = complement; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = similar. Note: Apple Watch solo loop -> Apple Watch band charms (no band), same day, same leaf. Recency top 3 includes an Apple Watch band but the target is outside both top 200s (two-tower full rank 28,427).

### F3 - instance 825873 (user 114172, history length 3)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2014-07-25 | ALLPOWERS153; 10000mAh Solar Battery Charger with iSolar153; Technology for iPhone, iPad Air mini, iPod, Samsung, Android Smart Phones and Tablets, Gopro Camera and other 5V USB devices (Black) | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Solar Chargers | ALLPOWERS | - |
| h2 | 2021-11-07 | SharingMoment Premium Smartphone Holder/Vertical and Horizontal Tripod Mount Adapter Rotatable Bracket with 1/4 inch Screw/Adjustable Clip for iPhone, Android Cell Phone, Selfie Stick, Camera Stand | Cell Phones & Accessories > Accessories > Photo & Video Accessories > Tripods | SharingMoment | 7.99 |
| h3 | 2021-11-07 | Efanr Universal Outdoor Tactical Holster Military Molle Hip Waist Belt Bag Wallet Pouch Purse Phone Case with Zipper Compatible with Samsung Galaxy S7 S6 LG HTC and More Smartphones (Khaki) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Efanr | - |
| **target** | 2021-11-07 | **TITACUTE 2 Pack Anti Dust Plugs for iPhone 13 Pro Max 14 12 Mini Dust Cover 8 Pin Dust Plug + Mini Storage Box iPhone Charging Port Plugs for iPhone 11 XS XR X 8 Plus SE 2020 2022 iPad Mini Air Green** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Anti-Dust Plugs | TITACUTE | 6.69 |

- Target `B0811KZ7TL`: warm; 16 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 39,246); recency rank: >200.
- Recency top 3: (1) Efanr® Universal Outdoor Tactical Holster Military Molle Hip Waist Belt Bag Wallet Pouch Purse Phone Case with Zipper for iPhone 7 6s Plus 5S Samsung Galaxy S7 S6 LG HTC and More (Black) [Efanr]; (2) Efanr Cell Phone Belt Clip Holster, Multipurpose Large Capacity Nylon Tactical Carrying Pouch Belt Loop Waist Bag Money Pocket, Universal Comaptible with Samsung HTC LG and More Smartphones [Efanr]; (3) Cell Phone Holder, Dealgadgets Universal Gooseneck Cell Phone Clip Holder Lazy Bracket Flexible Long Arms for iPhone 6 plus/6/5s/5/4S/4,Samsung Galaxy, HTC, Nokia, LG GPS Devices with Clean Cloth, Great for Bedroom, Office, Bathroom, Kitchen,Car etc.(Black) [DEALGADGETS]
- Two-tower top 3: (1) Anker Car Charger, 36W Metal Dual USB Car Phone Charger Adapter, PowerDrive III 2-Port 36W Alloy for iPhone 14 13 12 11 Pro Max mini X XR XS, iPad Pro mini, Galaxy S20/ S20+/ S10/ S10e/ S10+ and More [Anker]; (2) Lamicall Cell Phone Stand, Desk Phone Holder Cradle, Compatible with Phone 13 12 Mini 11 Pro Xs Max XR X 8 7 6 Plus SE, Smartphones Dock, Office Desktop Accessories - Green [Lamicall]; (3) Aibocn Power Bank 10,000mAh Phone Portable Charger with Flashlight (2 Pack) [Aibocn]
- Same user's later reviews within 30 days (context only, Val period only): 2021-11-07: X-EDITION USB Wall Charger, 4-Pack 2.1A Dual Port USB Wall Plug Power Adapter Charging Block Charger Box Cube Compatible with iPhone 13 12 11 Pro Max Mini XS XR X 8 7 6 Plus, Pad, Samsung, LG, Android [X-EDITION]
- **Rubric vs h3 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = yes; top-level jump = no; second-level jump = yes; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: Universal tactical holster -> iPhone Lightning dust plugs, same day as two other unrelated accessories (basket of miscellaneous accessories).

### F3 - instance 1488653 (user 205879, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2019-01-24 | Nite Ize Original Steelie Magnetic Phone Socket Plus - Additional Magnet for Larger Phones Using Steelie Phone Mounting Systems, Grey | Cell Phones & Accessories > Accessories > Automobile Accessories > Cradles | Nite Ize | 18.89 |
| h2 | 2019-10-13 | SUPCASE Full-Body Rugged Holster Case for Google Pixel 3 XL, with Built-in Screen Protector for Google Pixel 3 XL 2018 Release, Unicorn Beetle PRO Series - Retail Package (Blue) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | SUPCASE | - |
| h3 | 2021-03-04 | GOBUKEE [3GLASSES] for LG Wing 5G Screen Protector [2 Tempered Glass Main Screen Protector + 1 Tempered Glass Small Screen Protector] 3D Full Coverage Scratch Resistant Bubble Free for LG Wing 5G | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | GOBUKEE | 12.99 |
| h4 | 2021-04-18 | [ 2+2 Pack ] UniqueMe 2 Pack TPU Soft Screen Protector + 2 Pack Camera Lens Protector Tempered Glass for LG Wing 5G,[Anhydrous adsorption] Flexible Film HD TPU Clear Anti-Scratch Film | - | UniqueMe | - |
| **target** | 2021-08-15 | **Otium Cigarette Lighter Splitter, Type-C 20W PD Fast USB C Car Charger & Dual QC 3.0 180W 2-Socket Car Adapter for iPhone 12/11/11 Pro/X/8/7, Samsung, Google Pixel, GPS Dash Cam…** | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Automobile Chargers | Otium | - |

- Target `B08SM5W937`: warm; 9 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 7,166); recency rank: >200.
- Recency top 3: (1) [ 3+2 Pack ] UniqueMe TPU Soft Screen Protector + Camera Lens Protector Tempered Glass for LG Wing 5G Screen Protector【Easy to Install 】. [UniqueMe]; (2) [2 Pack] UniqueMe Screen Protector for LG Wing 5G Tempered Glass, 3D Full Coverage Film HD [Easy Installation] Edge to Edge Protection [UniqueMe]; (3) (2 Pack) LG Wing 5G Screen Protector Tempered Glass, Case Friendly Anti Scratch 9H Hardness Bubble-Free Shield High Definition Full Adhesive Clear Screen Protector for LG Wing 5G 6.8 inch [SURBUID]
- Two-tower top 3: (1) LK 3 PACK Screen Protector designed for Google Pixel 3 XL Tempered Glass High Clear, Case Friendly, Come with Installation Tray [LK]; (2) Anker 2 Pack 313 Wireless Charger (Stand), Qi-Certified for iPhone 14/14 Pro/14 Pro Max/13/13 Pro Max, 10W Fast-Charging Galaxy S20, S10 (No AC Adapter) [Anker]; (3) [2 Pack] ArmorSuit MilitaryShield Screen Protector Designed For Samsung Galaxy S20 / S20 5G (6.2")(Case Friendly) Ultrasonic Fingerprint Compatible Anti-Bubble HD Clear Film [ArmorSuit]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = unknown; relation = unrelated; leaf jump = unknown; top-level jump = unknown; second-level jump = unknown; store change = yes; target leaf unseen in history = yes; price jump = unknown. Note: LG Wing 5G protector (category missing) -> universal USB-C car charger/splitter; generic accessory with no device anchor.

### F3 - instance 2323078 (user 321554, history length 5)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2018-02-12 | (6 Pack) Supershieldz Designed for Motorola (Droid Turbo 2) Screen Protector, High Definition Clear Shield (PET) | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Supershieldz | - |
| h2 | 2018-02-12 | LK Case for Droid Turbo 2, [Shock Absorption] Hybrid Dual Layer Armor Defender Protective Case Cover for Verizon Motorola Droid Turbo 2 (Blue) | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | LK | - |
| h3 | 2019-08-21 | Casetego Compatible with iPhone 6S Plus Case,iPhone 6 Plus Case,Floral Three Layer Heavy Duty Hybrid Sturdy Shockproof Protective Cover Case for Apple iPhone 6S Plus/6 Plus,Mandala | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Casetego | - |
| h4 | 2021-07-05 | iPad Pro Charger, USB C Fast Charger, 20W ipad Charger with 10 FT USB C Cable Apple Certified Compatible with iPad Pro 12.9/11 inch 2021/2020/2018, iPad Air 4th Gen/iPad Mini 6th Gen, Pixel, Galaxy | Cell Phones & Accessories > Accessories > Chargers & Power Adapters > Wall Chargers | IOSYS | - |
| h5 | 2022-02-05 | [3 Pack] BAZO Tempered Glass Camera Lens Protector Compatible for iPhone 12 Pro Max 6.7" 5G, [Case Friendly] HD Clear 9H Hardness Back Camera Protector | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | BAZO | - |
| **target** | 2022-06-09 | **Uniwit [2 Pack] Camera Len Protector for iPhone 14 Pro/iPhone 14 Pro Max- Screen Protector Tempered Glass, Case Friendly No-Bubble HD Clear Anti-Scratch Camera Lens Screen Protector** | Cell Phones & Accessories > Accessories > Maintenance, Upkeep & Repairs > Screen Protectors | Uniwit | 5.99 |

- Target `B08YYXPWQ8`: warm; 3 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 8,249); recency rank: >200.
- Recency top 3: (1) [2+2 Pack] BAZO Tempered Glass Screen Protector and Camera Lens Protector Compatible for iPhone 12 Pro (6.1 inch), [Easy Installation Frame] [Anti-Scratch] [Bubble Free] [Case Friendly] 9H Hardness HD Clear Film [BAZO]; (2) [2+2 Pack] BAZO Compatible for iPhone 12 Pro Max 6.7 Inch - Privacy Screen Protector and Camera Lens Protector with Easy Installation Frame - Anti Glare 9H Hardness Tempered Glass Film [Case Friendly] [BAZO]; (3) BAZO [2+2 Pack] Compatible for iPhone 12 (6.1 inch) - Tempered Glass Screen Protector + Camera Lens Protector with Easy Installation Frame - Anti-Scratch - Case Friendly - 9H Hardness - HD Clear Film [BAZO]
- Two-tower top 3: (1) Ailun Glass Screen Protector Compatible for iPhone 11/XR,6.1 Inch 3 Pack Tempered Glass [Ailun]; (2) Ailun Screen Protector for iPhone 11 Pro Max/iPhone Xs Max 3 Pack 6.5 Inch 2019/2018 Release Case Friendly Tempered Glass [Ailun]; (3) Ailun Screen Protector Compatible for iPhone 8 plus,7 Plus,6s Plus,6 Plus, 5.5 Inch 3Pack Case Friendly Tempered Glass [Ailun]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h5 (model-assisted, needs human review):** same device ecosystem = yes; relation = substitute; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = unknown. Note: iPhone 12 Pro Max camera-lens protector -> iPhone 14 Pro lens protector (same leaf). Recency top 3 echoes the BAZO iPhone 12 Pro line. Target title names iPhone 14 on a June 2022 review: crawl-time title, model timing unreliable.

### F3 - instance 2369503 (user 328032, history length 4)

| # | Date | Title | Category path | Store | Price |
|---|---|---|---|---|---|
| h1 | 2016-08-02 | DeBin Case for Galaxy S23, S22, S21, S20, S10, S9, A10e, A01, Note 10, Cell Phone Belt Clip Holster Pouch Carrying Holder Cover (Fits Samsung Phone with Otterbox Commuter Case) Small Black | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | De-Bin | 11.99 |
| h2 | 2017-02-10 | Galaxy S7 Edge (ONLY) Case with Holster, Aduro Shell & Holster COMBO Case Super Slim Shell Case w/Built-In Kickstand + Swivel Belt Clip Holster for Samsung Galaxy S7 Edge | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Aduro | - |
| h3 | 2018-03-30 | Verizon Shell Holster Combo Case for Samsung Galaxy S7 Edge with Kickstand View and Belt Clip - Black - Retail Package | Cell Phones & Accessories > Cases, Holsters & Sleeves > Basic Cases | Verizon | 12.99 |
| h4 | 2020-08-30 | Encased Galaxy S10e Belt Case with Kickstand (2019 Slimline Series) Ultra Thin Cover w/Rotating Holster Clip (for Samsung Galaxy S10 E) Black | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Encased | 17.99 |
| **target** | 2021-10-01 | **Case for Galaxy Z Flip Phone, Nakedcellphone Black Vegan Leather Vertical Pouch [with Belt Loop, Metal Clip, Magnetic Closure] for Samsung Z Flip 4 5G, Z Flip3 (SM-F700, SM-F707, SM-F711, SM-F721)** | Cell Phones & Accessories > Cases, Holsters & Sleeves > Holsters | Nakedcellphone | 14.95 |

- Target `B08HL3ZXH5`: warm; 5 reviews before T1.
- Two-tower (text) rank: >500 (full-catalogue 18,881); recency rank: >200.
- Recency top 3: (1) Encased Galaxy S10 Plus Belt Case with Kickstand (2019 Slimline Series) Ultra Thin Cover w/Rotating Holster Clip (for Samsung Galaxy S10+) Black [Encased]; (2) Encased Belt Case with Kickstand for Galaxy S10 (2019 Slimline Series) Ultra Thin Cover w/Rotating Holster Clip (Compatible with Samsung S10) Black [Encased]; (3) Encased Belt Case for Samsung Galaxy S9 Holster Clip with Metal Kickstand (Slimline Series, Black) [Encased]
- Two-tower top 3: (1) Aduro Cell Phone Holsters for Samsung Galaxy S10 E Case Protector Includes Belt-Clip & Built-in Kickstand [Aduro]; (2) Anker 2 Pack 313 Wireless Charger (Stand), Qi-Certified for iPhone 14/14 Pro/14 Pro Max/13/13 Pro Max, 10W Fast-Charging Galaxy S20, S10 (No AC Adapter) [Anker]; (3) Encased Galaxy S10 Belt Clip Case (2019 DuraClip) Slim Grip Cover w/Holster for Samsung S10 (Black) [Encased]
- Same user's later reviews within 30 days (context only, Val period only): none
- **Rubric vs h4 (model-assisted, needs human review):** same device ecosystem = yes; relation = substitute; leaf jump = no; top-level jump = no; second-level jump = no; store change = yes; target leaf unseen in history = no; price jump = similar. Note: Galaxy S10e belt holster -> Galaxy Z Flip vertical belt pouch (same leaf). Recency top 3 are Encased S10/S9 belt cases: right type, old model; user appears to have moved to a Z Flip.
