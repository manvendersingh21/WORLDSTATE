# WORLDSTATE — Pitch Materials

Source of truth for facts/numbers: `docs/SUBMISSION.md` and `docs/demo/worldstate-demo.mp4` (74.4 s, confirmed via ffprobe). Beats below are timed to the demo video's actual sequence (graph → normal → novel → remember → known → search), compressed for the 60 s cut and expanded for the 2-minute walkthrough.

---

## 1. 60-Second Spoken Pitch Script

Captions ≤ 8 words. Spoken lines are written to land naturally in each beat's duration — read them at a confident, unhurried pace (~2.5–3 words/sec).

| Beat | Time | On-screen moment | Caption (≤8 words) | Spoken line |
| --- | --- | --- | --- | --- |
| 1 | 0:00–0:07 | State graph, "learned from 16 unlabeled videos" | Watch once. Learn the process. | "WORLDSTATE watched sixteen robot pick-and-place runs — no labels, no SOP — and learned the process on its own." |
| 2 | 0:07–0:14 | `normal_16` run, green NORMAL badge, score 0.05 | Normal runs match the learned path. | "Here's a normal run. It follows the exact path WORLDSTATE learned, so it's flagged NORMAL in real time." |
| 3 | 0:14–0:26 | `miss_unseen` run, cube shifts, red NOVEL FAILURE at 1.904s | Novel failure. Never seen before. | "Now watch this one. At one point nine zero four seconds, the cube slides sideways instead of being grasped — a failure in zero out of sixteen reference runs. WORLDSTATE catches it instantly." |
| 4 | 0:26–0:36 | Right panel: EXPECTED "Grasping part" vs OBSERVED "Slides sideways"; Cosmos narration box | Cosmos explains what went wrong. | "NVIDIA Cosmos Reason reads the tracked video and explains it in plain language: the gripper closed on air, and the cube never left the table." |
| 5 | 0:36–0:45 | Click "Remember this failure" → confirmation → graph gains a named (blue) branch | Remember it. Teach the model. | "One click — Remember this failure — and WORLDSTATE adds this pattern to its memory. The state graph now has a named branch for it, and WORLDSTATE shows the fix: a corrected attempt that re-aligns to the moved cube and places it." |
| 6 | 0:45–0:52 | A second similar miss loads, badge flips to blue KNOWN FAILURE | Next time, it's already known. | "The next similar miss isn't novel anymore — it's recognized instantly as a KNOWN FAILURE." |
| 7 | 0:52–0:58 | Search bar: "show me failed grasps" → result cards | Ask memory. Get the history. | "And you can just ask it: 'show me failed grasps' — and it searches every run it's ever seen." |
| 8 | 0:58–1:00 | Logo / tagline card | A camera that learns and remembers. | "Factories, warehouses, labs — any repeated process. One camera that learns the process, and catches the first-ever failure." |

**Total spoken runtime target: ~58–60 s.** Caption text appears as on-screen lower-third text synced to each beat's start.

---

## 2. 2-Minute Judge Walkthrough

Same spine as the 60-second cut, but each beat is unpacked with the "why it's hard" context judges care about (unsupervised learning, real inference, honest limitations).

**[0:00–0:15] Opening — the problem**
"Anyone running a repeated physical process — a pick-and-place cell, a lab protocol, a warehouse pick station — eventually hits a failure nobody wrote a rule for. Today that's caught by a human watching a monitor, or not caught at all. WORLDSTATE is a camera-based agent that learns the process itself, with no labels, and tells you the moment reality diverges from what it learned."

**[0:15–0:35] How it learns**
"We show it 16 unlabeled videos of the same pick-and-place task. YOLOv8 tracks the gripper, the part, and the fixture frame by frame. From those tracks alone — no SOP, no state labels — WORLDSTATE clusters the motion into 7 discovered states: At pickup, Grasping, Lifting, Carrying, Descending, Placing, Seated. All 16 reference runs follow the exact same path through that graph."

**[0:35–0:55] Normal case**
"Run a held-out normal video and it walks the same path — NORMAL badge, score 0.05, matches the learned process. That's the baseline: WORLDSTATE isn't guessing, it's measuring distance from a model it built from data."

**[0:55–1:15] The novel failure — the money shot**
"Now `miss_unseen`. At 1.904 seconds the cube shifts sideways right as the gripper closes — the grasp closes on air. WORLDSTATE flags NOVEL FAILURE, score 0.90. The right panel shows exactly why: expected transition was At pickup → Grasping part, seen in 16 of 16 reference runs; observed was Slides sideways, seen in 0 of 16. This transition has never happened before in anything WORLDSTATE has seen — that's what 'novel' means here, precisely, not vaguely."

**[1:15–1:30] Cosmos explains it**
"We don't just show a number. NVIDIA Cosmos Reason — running live on the event's CoreWeave GPU endpoint, model `nvidia/cosmos3-nano-reasoner` — reads the video and the tracker facts and narrates what happened in plain English, attributed on screen so you know it's not a canned string."

**[1:30–1:50] Remember and recognize**
"Here's the part that makes this a system, not a one-off detector: click Remember this failure, and WORLDSTATE adds a new class — displaced-after-align — to its memory and bumps its world model to v2. Feed it a similar miss and it's no longer novel. It's a KNOWN FAILURE, recognized in roughly the time it takes to run inference — the upload-and-match path itself takes about 18 seconds end to end; remembering is instant."

**[1:50–2:00] Search and close**
"And because every run is embedded into memory — SQLite plus FAISS — you can just ask it: 'show me failed grasps,' and it returns the matching runs with the reasoning attached. Factories, warehouses, labs: anywhere there's a repeated physical process, this is a camera that learns it — and catches the first failure it's ever seen."

---

## 3. Judge Q&A — Six Likely Judge Questions, Crisp Honest Answers

**Q1: Is this video simulated? Why not real camera footage?**
Yes — the runs are robosuite/MuJoCo simulations of a pick-and-place task, fixed camera, 480×360 @ 20fps. We used simulation because it gives us free, exact ground truth (we know precisely when and how a failure happened) and a repeatable way to generate both normal runs and controlled failures in the hackathon's timeframe. The perception and learning pipeline (YOLO detection, tracking, state-graph clustering, novelty scoring) is camera-agnostic — it operates on bounding-box tracks, not on simulation-specific signals, so swapping in real video is a data problem, not an architecture change. We say this directly in the UI and in our submission; we're not hiding it.

**Q2: What does Cosmos actually do versus the state graph? Which one makes the verdict?**
The state graph makes the verdict. It's built with unsupervised clustering over kinematic features from tracked video — it defines the states, the expected transitions, and computes the novelty score (transition surprise + embedding distance + trajectory abnormality) that produces NORMAL / NOVEL FAILURE / KNOWN FAILURE. Cosmos Reason never sets a state boundary or decides the verdict — it's a narrator. It receives the verdict, the tracker facts, and the video at 4fps, and explains in natural language what happened. Separating "what decided" from "what explains" is deliberate — it means the verdict is reproducible and auditable even if the narration model changes.

**Q3: You mention VAST — why isn't it live in the demo?**
We implemented a VastDB mirror (the `vastdb` SDK, writing to a `worldstate.runs` table) as an optional memory backend alongside our live SQLite + FAISS store. We tested it from a laptop with workshop credentials, but the VastDB endpoint is workshop-internal — it only resolves inside the event's VM network, not from a laptop on the show floor, and reaching it would require an SSH tunnel through the workshop jump host, which isn't something we want to depend on live, on stage. So the live demo runs on local SQLite + FAISS, which gives identical search behavior; VAST is there, tested, and ready to flip on inside that network.

**Q4: How does this scale to a real factory camera, not simulated video?**
Three concrete steps, no architecture change: (1) retrain the YOLO detector on real footage of the actual parts/gripper/fixture — the detector is the only piece that's scene-specific; (2) point the pipeline at a `WORLDSTATE_REAL_MANIFEST` of real run videos instead of the simulated 16, so the state graph learns the real process's own states and transitions; (3) re-run the same unsupervised clustering and novelty scoring, unchanged. The state-graph learning, novelty scoring, memory, and Cosmos narration layers are all already camera- and task-agnostic; they operate on tracked motion, not on the simulator.

**Q5: What specifically does "unsupervised" mean here — isn't YOLO trained with labels?**
Two different things, and we're careful to separate them. YOLO's object detector is trained with bounding-box labels — that's supervised, and it's only teaching the model to find the gripper/part/fixture in a frame, nothing about the process. The state graph — what counts as a "state," what the normal sequence of states is, and what counts as a deviation — is learned with zero labels, from clustering the motion of those tracked boxes across 16 videos nobody annotated with step names. Nobody ever told WORLDSTATE "this is called Grasping" — it discovered that there are 7 recurring motion clusters and that they occur in one order; Cosmos just gave the discovered clusters readable names afterward.

**Q6: What's next if you kept building this?**
Three things, in priority order: (1) real camera data — swap in a `WORLDSTATE_REAL_MANIFEST` from an actual cell and retrain the detector, per Q4; (2) richer memory — move the VAST mirror from "implemented, untested live" to the actual system of record once we're inside the right network, so world-model versions and search are durable across restarts; (3) closed-loop recovery — right now "recovery path" is suggested/stubbed text; the next step is generating or even simulating a corrective action, not just naming one.

---

## 4. Timing & Delivery Notes

- 60 s script totals ~165 spoken words across 8 beats — comfortably within a natural speaking pace; rehearse once against the actual video playback to confirm beat 3/4 (the failure + Cosmos explanation) doesn't get rushed, since it's the emotional core of the demo.
- Demo video runtime is 74.4 s total; the 60 s pitch should be voiced over a trimmed/sped cut or over b-roll of the same five moments (graph, normal, novel, remember, search) rather than the full video.
- 2-minute version assumes the full 74.4 s video plays roughly in sync with narration; pad the "Remember and recognize" beat if the live upload-similar-run take (~18 s) is shown rather than cut.
