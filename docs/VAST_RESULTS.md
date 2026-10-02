# WORLDSTATE unsupervised pipeline on 40 VAST warehouse clips

No ground-truth labels exist for these clips, so nothing below is an accuracy number. Script: `scripts/vast_unsupervised_run.py`. Raw output: `vast_worldstate_report.json`, `vast_worldstate_scores.csv`.

## Setup

- 40 clips, about 5 s each, one fixed ceiling camera (`sdg_warehouse_cam-2`), synthetic SDG warehouse footage. VAST query: "forklift moving through a warehouse aisle".
- Features: optical flow (`series_from_video_flow`). `WorldModel.fit(..., "flow")` was fit on 32 clips; 8 were held out (every 5th clip (index % 5 == 4)). Flow extraction plus fit took 11.6 s.
- Scores for the 32 training clips are **in-sample**, because the threshold is calibrated on those same clips. Only the 8 held-out clips are a fair novelty test.

## Learned graph

- 4 states (kmeans): s0 "Lateral motion", s1 "Lateral motion 2", s2 "Vertical motion", s3 "Lateral motion 4". `interpret_flow` gives only three labels (still, lateral, vertical), so three of the four states are variants of "Lateral motion".
- Dominant path: Lateral motion 4, which is a single state, followed by 11/32 training clips. There are 10 distinct sequences across 32 clips.
- Path agreement (median pairwise LCS): **0.5**. The app's own stability bar is 0.75 (`learn._stable`), so this graph does **not** pass.
- Threshold 0.638. Scores: min 0.012, median 0.118, max 0.660.
- Status counts: all 40 clips {'normal': 39, 'novel': 1}; held-out only {'normal': 7, 'novel': 1}.

## Weak sanity check against VAST Cosmos captions

- 5 of 40 captions contain a non-NONE `EVENTS:` tag (all `PERSON_IN_PATH`): clip_039 (test, score 0.66, novel), clip_001 (train, score 0.27, normal), clip_027 (train, score 0.18, normal), clip_031 (train, score 0.02, normal), clip_013 (train, score 0.02, normal).
- Mean score is 0.231 for clips with caption events and 0.172 for clips without. Only 1 clip was flagged, so the flagged-vs-normal keyword rates in the JSON are based on n=1 and mean nothing statistically.
- The one flagged clip, clip_039, does have a `PERSON_IN_PATH` / RISK medium caption. The second-highest score, clip_009 at 0.635, sits just under the threshold, and its caption says it is a "close-up view of a forklift" with no event. That clip has the largest flow magnitude, which suggests a camera or view change, not a safety event. Words like "near" and "close" show up in boilerplate captions ("no risk of a near-miss"), so keyword matching is unreliable.

## Per-clip table (sorted by score)

| clip | split | score | status | divergence (s) | flow mag | caption EVENTS / RISK | caption snippet | YOLO top classes |
|---|---|---|---|---|---|---|---|---|
| clip_039 | test | 0.660 | novel | 0.867 | 2.32 | PERSON_IN_PATH / medium | The scene shows a warehouse floor with a forklift and a person. The forklift is blue and black with the brand ... | person:2, truck:1, sports ball:1 |
| clip_009 | test | 0.635 | normal | 0.333 | 3.52 | NONE / low | The video shows a close-up view of a forklift with a teal-colored body and black accents, positioned on a ware... | suitcase:2, microwave:1, clock:1 |
| clip_014 | test | 0.520 | normal | 2.309 | 0.57 | NONE / low | The warehouse floor is clear of forklifts, pallet jacks, robots, and pallets. A single person in a white unifo... | person:1 |
| clip_007 | train | 0.485 | normal | 0.735 | 1.09 | NONE / low | The warehouse floor is empty of pedestrians and vehicles, with no forklifts, pallet jacks, robots, or pallets ... | truck:2, person:1, traffic light:1 |
| clip_012 | train | 0.417 | normal |  | 3.04 | NONE / low | The video shows a warehouse environment with a forklift and a person. The forklift, branded 'ATLAS', is initia... | bench:3, person:1, dining table:1 |
| clip_002 | train | 0.395 | normal |  | 1.89 | NONE / low | The clip shows a warehouse interior with a person standing near a forklift. The forklift is an Atlas model, bl... | person:1, teddy bear:1, bench:1 |
| clip_008 | train | 0.298 | normal |  | 1.10 | NONE / low | The video shows a warehouse interior with a forklift and a person. The forklift is black with a blue side pane... | bench:1, suitcase:1, oven:1 |
| clip_037 | train | 0.272 | normal |  | 0.99 | NONE / low | The only person visible is a worker in a white shirt and cap, standing near a blue and black forklift. The for... | person:1, boat:1 |
| clip_010 | train | 0.271 | normal |  | 0.71 | NONE / low | The warehouse floor is dimly lit with a yellowish hue. A person wearing a light-colored shirt and dark pants i... | person:2, suitcase:1 |
| clip_001 | train | 0.270 | normal |  | 0.68 | PERSON_IN_PATH / high | The scene shows a warehouse floor with a person standing near a forklift. The forklift is stationary, and the ... | person:1, boat:1, oven:1 |
| clip_024 | test | 0.259 | normal |  | 0.38 | <NONE> / low | The scene shows a single forklift, a blue and black Atlas model, positioned on a warehouse floor with a grid p... | bench:1 |
| clip_028 | train | 0.239 | normal | 4.094 | 0.91 | NONE / low | The warehouse floor is empty of pallets, boxes, and other objects. A person wearing a light-colored shirt and ... | person:1, oven:1, microwave:1 |
| clip_027 | train | 0.184 | normal |  | 0.48 | PERSON_IN_PATH / low | The only person visible is a worker in a white shirt and dark pants standing near a forklift in the warehouse.... | person:1 |
| clip_003 | train | 0.178 | normal |  | 0.77 | NONE / low | The video shows a close-up view of a blue and black forklift on a warehouse floor. The forklift is stationary,... | suitcase:1 |
| clip_021 | train | 0.165 | normal |  | 0.68 | NONE / low | The warehouse floor is clear of forklifts, pallet jacks, robots, and pallets. The only person visible is a wor... | person:1, dog:1 |
| clip_004 | test | 0.154 | normal |  | 0.34 | NONE / low | The scene shows a blue and black forklift with the brand name 'ATLAS' visible on its side, positioned in a war... | refrigerator:1 |
| clip_026 | train | 0.133 | normal |  | 1.16 | NONE / low | The only person visible is a police officer in a white uniform with the word 'POLICE' on the back, wearing a c... | person:1 |
| clip_035 | train | 0.120 | normal |  | 0.41 | NONE / low | The warehouse floor is clear of vehicles and other hazards. The only person visible is a worker in a light-col... | person:1 |
| clip_033 | train | 0.120 | normal |  | 0.79 | <NONE> / low | The warehouse floor is clear of pallets, boxes, and other obstacles. A person wearing a white uniform and cap ... | person:1, suitcase:1 |
| clip_017 | train | 0.119 | normal |  | 0.72 | NONE / low | The only person visible is a worker in a white shirt and white cap, standing near a blue and black forklift. T... | person:1 |
| clip_020 | train | 0.118 | normal |  | 0.18 | <NONE> / low | The only person visible is a worker in a white shirt and light pants walking away from the camera on the wareh... | person:1 |
| clip_029 | test | 0.117 | normal |  | 0.15 | NONE / low | The warehouse floor is clear of forklifts, pallet jacks, robots, and pallets. A person wearing a white uniform... | person:1, boat:1 |
| clip_015 | train | 0.117 | normal |  | 0.51 | NONE / low | The warehouse floor is clear of pallets, boxes, and other obstacles. A person wearing a light-colored shirt an... | person:1, refrigerator:1 |
| clip_006 | train | 0.116 | normal |  | 0.75 | NONE / low | The warehouse floor is empty of other people, vehicles, or obstacles. The only person visible is standing stil... | person:1, suitcase:1 |
| clip_030 | train | 0.114 | normal |  | 0.70 | NONE / low | The only person visible is a warehouse worker wearing a white shirt with a name tag and a white cap, standing ... | person:1 |
| clip_023 | train | 0.114 | normal |  | 0.13 | NONE / low | The warehouse floor is empty of forklifts, pallet jacks, robots, and pallets. A person wearing a white shirt w... | person:1 |
| clip_025 | train | 0.113 | normal |  | 0.76 | NONE / low | The only person visible is a worker wearing a white shirt with "POLICE" printed on the back and a white cap, s... | person:3, boat:1 |
| clip_018 | train | 0.113 | normal |  | 0.15 | NONE / low | The warehouse floor is clear of forklifts, pallet jacks, robots, and pallets. A single person wearing a white ... | person:1 |
| clip_038 | train | 0.112 | normal |  | 0.90 | NONE / low | The warehouse floor is clear of pallets, boxes, and other obstacles. A person wearing a white shirt with 'POLI... | person:1, tennis racket:1 |
| clip_032 | train | 0.078 | normal |  | 0.55 | NONE / low | The scene shows a warehouse floor with a person standing near a forklift. The person is wearing a light-colore... | person:2, suitcase:1 |
| clip_036 | train | 0.023 | normal |  | 0.71 | NONE / low | The person is wearing a white shirt and gray pants, standing near a blue and black forklift. The forklift is s... | person:2, truck:2, suitcase:2 |
| clip_000 | train | 0.022 | normal |  | 0.68 | <NONE> / low | The warehouse floor is clear of vehicles and pedestrians, with no boats, trucks, or other equipment visible in... | person:1, boat:1, truck:1 |
| clip_031 | train | 0.020 | normal |  | 0.77 | PERSON_IN_PATH / low | The only person visible is a warehouse worker in a white uniform and cap, standing near a blue and black forkl... | person:1, refrigerator:1 |
| clip_013 | train | 0.020 | normal |  | 0.74 | PERSON_IN_PATH / high | The scene shows a warehouse floor with a person in a white uniform and cap walking toward a blue and black for... | suitcase:2, person:1, boat:1 |
| clip_019 | test | 0.018 | normal |  | 0.14 | NONE / low | A person wearing a white shirt and white cap is walking away from the camera toward the back of the warehouse.... | suitcase:2, person:1 |
| clip_022 | train | 0.018 | normal |  | 0.54 | NONE / low | The warehouse floor is dimly lit with an orange hue, and the concrete surface shows wear. A person wearing a l... | person:1, skis:1, suitcase:1 |
| clip_034 | test | 0.018 | normal |  | 0.15 | NONE / low | The only person visible is a worker in a white uniform and cap, standing on the warehouse floor. The only vehi... | person:1, boat:1 |
| clip_016 | train | 0.016 | normal |  | 0.14 | NONE / low | The warehouse floor is empty of forklifts, pallet jacks, robots, and pallets. The only person visible is a wor... | person:1 |
| clip_011 | train | 0.015 | normal |  | 0.55 | NONE / low | The warehouse scene shows a single person in a light-colored uniform and dark pants walking across the floor. ... | person:1 |
| clip_005 | train | 0.012 | normal |  | 0.40 | NONE / low | The person is walking away from the forklift, moving toward the back wall of the warehouse. The forklift is st... | person:1 |

## Honest conclusion

1. No stable process graph exists on this footage. Path agreement is 0.50, against the 0.75 bar, and the 32 training clips follow 10 different state sequences. These are unrelated 5 s slices of open-floor activity, not repetitions of one process.
2. The 4 discovered states are coarse motion clusters (mostly "Lateral motion" variants). They do not correspond to warehouse semantics such as a forklift approaching or a person in an aisle.
3. The detector flagged 1 of 8 held-out clips (clip_039). Its Cosmos caption does report PERSON_IN_PATH, but that is n=1. The other 4 captioned PERSON_IN_PATH clips were training clips, scored in-sample, and all came out low.
4. The flow score mostly tracks overall motion and view changes (clip_009 is a close-up), not safety events. YOLO-COCO classes ("boat", "suitcase", "oven") are also noisy on this synthetic footage.
5. The robot-cell result does not transfer automatically. WORLDSTATE needs many repetitions of the same process from a fixed view. On unrepeated warehouse footage it acts at best as a generic motion-outlier ranker, and making it useful here would take object-level tracks (forklift/person) and more data.
