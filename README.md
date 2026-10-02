# WORLDSTATE

Can a model watch repeated videos of one physical process, discover the state graph without a SOP, and then explain a run it has never seen?

WORLDSTATE watches a robot pick-and-place cell from one fixed camera. From 16 unlabeled normal runs it learns the process as a state graph. It then flags a run it has never seen, where the cube shifts after the gripper aligns and the grasp closes on air, as a novel transition. It says when and why. After you confirm it once, the next similar miss comes back as a known failure.

Built for the Real-Time Video Agents Hack (VAST / NVIDIA, SF, Oct 2 2026): video understanding, real-time detection, semantic search and memory.

The video is simulated (robosuite, MuJoCo), not real camera footage. Submission summary: [docs/SUBMISSION.md](docs/SUBMISSION.md). Demo video: `docs/demo/worldstate-demo.mp4`. Live demo: https://calgary-postings-shoot-genealogy.trycloudflare.com (Cloudflare quick tunnel from a laptop running the Docker image; the URL changes if the tunnel restarts).

## Run it

```bash
make deploy          # docker build + run on 0.0.0.0:${PORT:-8000}; reads .env if present; learns the graph from video at boot (~6 min on a laptop CPU)
scripts/tunnel.sh    # public https://*.trycloudflare.com URL (cloudflared + caffeinate)
# or, without Docker:
make demo            # venv, learn, eval, serve on 0.0.0.0:8000
scripts/vm-run.sh    # on the VAST workshop VM: runs WORLDSTATE with the VM's /config team file
```

`scripts/vm-run.sh` reads the team file in the VM's `/config` (Cosmos token and endpoint, VastDB variables), which resolve inside the VM network.

GPU: the event's CoreWeave GPU servers (Cosmos3 Reason, Cosmos Embed1, YOLO11) are reachable with the team token. WORLDSTATE uses Cosmos Reason. The app itself runs on CPU.

Configuration comes only from the environment. Nothing is baked into the image:

| Variable | Effect |
| --- | --- |
| `WORLDSTATE_DATASET` | `process` (default: the robosuite cell in `data/process`), `synthetic` (cartoon fallback), `real` (Exylos-style clips via `WORLDSTATE_REAL_MANIFEST`) |
| `WORLDSTATE_PERCEPTION` | `yolo` (default for the process when weights exist), `auto`, `classical`, `motion` |
| `WORLDSTATE_REAL_MANIFEST` | Your own repeated-process mp4s. Same `episodes` schema as `data/process/manifest.json` |
| `COSMOS_BASE_URL`, `COSMOS_API_KEY` | Event / self-hosted Cosmos endpoint and its bearer token (`GPU_BEARER_TOKEN` also accepted) |
| `NVIDIA_API_KEY` / `NGC_API_KEY` / `NV_API_KEY` | Hosted NVIDIA API (used when no `COSMOS_BASE_URL`) |
| `NVIDIA_COSMOS_MODEL` | Pin the Cosmos model id instead of discovering it |
| `VDB_ENDPOINT`, `VAST_ACCESS_KEY`, `VAST_SECRET_KEY`, `VASTDB_BUCKET` | Mirror run memory into VastDB (`worldstate.runs`) with the `vastdb` SDK. SQLite + FAISS stay the source of truth |
| `VAST_API_URL`, `VAST_API_KEY` | Generic REST mirror for a VAST-fronted service |
| `COSMOS_PREDICT_URL` | Recovery video generation endpoint. Stubbed without it |
| `PORT` | Server port (default 8000) |

## Demo story

1. **The graph** is learned from 16 normal runs with no labels. Robot run: At pickup → Grasping part → Lifting part → Carrying part → Descending → Placing part → Seated in fixture. States are clustered from tracked-object kinematics. Names are attached afterwards.
2. **Unseen failure** (`miss_unseen`): the cube is teleported sideways about 5 cm right after alignment. The run scores as a novel transition, diverging at 1.9 s, with expected vs observed transition and support (seen in 0/16 reference runs). The UI shows NORMAL / NOVEL FAILURE / KNOWN FAILURE badges.
3. **Why** explains it from the graph. **What happened** is the event text. It comes from Cosmos Reason when available and is labeled with its source, otherwise it falls back to the kinematic events. Before the demo, the operator pre-generated the Cosmos narrations for the demo runs and kept ones consistent with the tracker's facts (cached per run and model version).
4. **Ask memory** takes plain questions ("show me failed grasps", "when did the cube move unexpectedly"). Every run is indexed with text derived from its own tracks and verdict. Search is hybrid: FAISS similarity plus a keyword boost over local SQLite. The VastDB mirror is optional and not used in the live demo.
5. **What should have happened** is the nominal path. Recovery video generation is stubbed; the graph recovery path is real.
6. **Remember this failure** adds the class `displaced_after_align`, a failure edge and a recovery edge, and bumps the world model to v2.
7. **Upload similar run** sends another miss through the real upload endpoint. It comes back as a known failure. Subsequent left and right misses become known failures; normals stay normal.

Relearn/reset is an operator action, not a judge action: `curl -X POST http://127.0.0.1:8000/api/learn` locally (takes minutes, too slow for the public tunnel). The Learn button is hidden in the judge UI.

## Judge Mode UI

`web/index.html`, `web/app.js`, `web/styles.css` (the previous UI is kept in `web/legacy/`). Same API and behavior as before, in a presentation layout:

- A dominant robot video with a status badge (NORMAL / NOVEL FAILURE / KNOWN FAILURE).
- An Event Intelligence panel: headline, divergence time (e.g. 1.904s, "REALITY DIVERGED"), Expected / Observed / Prior support (0/16), and a Cosmos Reasoning excerpt with attribution.
- A large "Remember this failure" action.
- A full-width learned state graph. The abnormal branch is red, and blue once remembered.
- "Search world memory" with human-readable result cards.

## 3D digital twin (optional view)

A toggle switches the video panel between CAMERA and 3D DIGITAL TWIN. CAMERA is the fallback. If the twin ships hidden, Judge Mode runs as above without it.

- `web/twin/twin.js` with Three.js vendored under `web/twin/vendor`. A Franka Panda arm, hand and fingers are rendered from robosuite's Panda meshes converted to GLB (`web/twin/assets`; `NOTICE.txt`: robosuite MIT, Franka meshes Apache-2.0).
- It is driven by exact per-frame joint and link poses replayed from the deterministic simulator that generated the videos (`scripts/export_twin.py` writes `data/process/twin/*.json`). `tests/test_twin_assets.py` checks that the replay matches the recorded gripper and cube positions within 2 mm on all 24 runs.
- Playback is synchronized to the camera video time and to WORLDSTATE's detected states. On `miss_unseen` it shows the cube shift and a red "REALITY DIVERGED — 1.904s" marker.
- Orbit, zoom and pan; Perspective / Front / Side / Top presets; Reset View.

Provenance: the digital twin replays the simulator's recorded trajectory, synchronized to WORLDSTATE's states; it is not reconstructed from YOLO.

**The fix loop (after Remember):** a *THE FIX — learned from memory* card shows the recovery plan from the graph (re-align to the displaced cube → grasp → lift → carry → place) and **▶ Show corrected attempt** plays a simulated closed-loop re-grasp in the same cell (same seed and 4.5 cm displacement as `miss_unseen`; generated by `scripts/gen_recovery.py`), ending in *CORRECTED ✓ — next attempt passes* in camera and 3D. It is labeled as a simulated corrective attempt, not a retry of the original run.

## Pipeline

- **Video**: `scripts/gen_process.py` runs robosuite Lift (Panda, MuJoCo on CPU) with a fixed front camera and a green target pad. The same scripted pick-and-place runs every time, with ≤5 mm start jitter and ±5% timing. The misses teleport the cube about 4.5–5.5 cm along ±y right after the gripper aligns. The policy is open loop, so the gripper closes on air and carries nothing. 20 normals (16 reference, 4 held out) and 4 misses, H.264, 9 s at 20 fps. `tests/test_process_dataset.py` checks the physics from simulator state.
- **Detector quality** (`models/yolo_metrics.json`): val mAP50 0.995. On held-out test runs, per-class detection rate ≥ 0.955 and median center error < 0.2% of image width.
- **Perception**: a YOLO detector (`models/worldstate-yolo.pt`; classes gripper, part, fixture) plus Ultralytics tracking, run through `worldstate.perception.yolo_track`. The detector was trained on simulator-projected boxes from the reference runs only (`scripts/train_yolo.py`). Box labels train the detector; the state graph gets no labels. Color blobs and optical flow remain fallbacks.
- **World model** (`worldstate/engine.py`): sliding-window kinematic features from tracks are clustered (HDBSCAN / k-means by silhouette), ordered by time, and turned into a transition graph. Novelty = transition surprise + embedding distance to the nearest state + trajectory abnormality.
- **Cosmos Reason** (`worldstate/adapters.py`): at the event this is the Cosmos3 reasoner served on CoreWeave GPUs (`COSMOS_BASE_URL` + `COSMOS_API_KEY`, the team `GPU_BEARER_TOKEN`). The model id is discovered from `/v1/models` (`nvidia/cosmos3-nano-reasoner`). Without it, the hosted `integrate.api.nvidia.com` is used with `NVIDIA_API_KEY`. Video goes in as a base64 `video_url` at fps 4. Narrations for the demo runs were pre-generated and curated by the operator (see Demo story). The prompt gets the graph's verdict and the tracker's facts (cube shifted at t, lifted or not, ended on the pad), and Cosmos writes the account. Cosmos names clusters and writes event text. It never chooses cluster boundaries or the verdict. If a call fails, the kinematic text stays and the UI shows why.
- **Memory**: SQLite + FAISS (source of truth). Optional VastDB mirror (`vastdb` SDK, table `worldstate.runs`) when `VDB_ENDPOINT` and VAST keys are set, or a generic REST mirror when `VAST_API_URL` is set. Neither is active in the live demo.

## Eval

`python -m worldstate.eval_novelty` scores held-out normals against held-out misses. It then checks that remembering one miss makes the others a known class without flagging normals. Results:

| Dataset | Graph | AUROC | Held-out normal max | Miss min | Memory check |
| --- | --- | --- | --- | --- | --- |
| Robot process, YOLO tracks (deployed) | 7 states, 1 path (16/16 runs) | 1.00 | 0.070 | 0.90 | pass (left and right misses) |
| Robot process, ground-truth boxes (upper bound) | 7 states, 1 path | 1.00 | 0.056 | 0.90 | — |
| Synthetic cell (fallback) | 7 states, 1 path | 1.00 | 0.054 | 1.00 | pass |

The 30 Exylos front-camera clips are domain randomized and do not hold one stable graph (path agreement ~0.62, AUROC ~0.70). They are not the deployed demo.

## Regenerate

```bash
make process   # robosuite dataset (uv, Python 3.12, mujoco 3.3.0)
make yolo      # retrain the detector from simulator boxes
```

## Sponsor integrations

- **NVIDIA**: Cosmos Reason (Cosmos3 reasoner) watches each clip and writes what happened and why a run is unusual. It also names discovered states. It never chooses state boundaries or the verdict. YOLO (Ultralytics) tracks the gripper, cube and target.
- **VAST**: an optional VastDB mirror of run memory (`vastdb` SDK, table `worldstate.runs`) is implemented. Tested from the laptop with team credentials, the VastDB endpoint is workshop-internal and does not resolve outside the VM network (DNS failure), so the live demo uses local SQLite + FAISS. Reaching it needs an SSH tunnel via the workshop VM jump host. VAST is not used live.
- **CoreWeave**: the event's Cosmos endpoint runs on CoreWeave GPUs. The WORLDSTATE app itself runs on CPU (MuJoCo sim, YOLOv8n). W&B is not used.

## How it was built

The work was split across HACP pairs (bilateral contracts, frozen terms, counterparty verification), one git worktree per pair:

| Pair | Peer a | Peer b |
| --- | --- | --- |
| Simulated process | codex (ran out of quota mid-task; Claude took over the generator) | Claude Sonnet: dataset validator and tests |
| Deploy | opencode, GLM-5.3: Dockerfile, .dockerignore | agy, Gemini 3.8 Flash: entrypoint, run script |
| Perception | Cursor, Claude Opus 5: detector training | agy, Gemini 3.1 Pro: tracking tests |
| VAST diagnostic | Cursor, GPT-5.3 Codex: diagnostic script | agy, Gemini Flash: report |

Claude Opus 5.5 integrated everything on main (dataset loader, learn/eval, Cosmos narration, naming, deploy).
