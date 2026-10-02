# WORLDSTATE

Can a model watch repeated videos of one physical process, discover the state graph without a SOP, and then explain a run it has never seen?

WORLDSTATE watches a robot pick-and-place cell from one fixed camera. From 16 unlabeled normal runs it learns the process as a state graph. It then flags a run it has never seen, where the cube shifts after the gripper aligns and the grasp closes on air, as a novel transition. It says when and why. After you confirm it once, the next similar miss comes back as a known failure.

Built for the Real-Time Video Agents Hack (SF, Oct 2 2026): video understanding, real-time detection, semantic search and memory.

## Run it

```bash
make deploy          # docker build + run on 0.0.0.0:${PORT:-8000}; reads .env if present
# or, without Docker:
make demo            # venv, learn, eval, serve on 0.0.0.0:8000
```

Configuration comes only from the environment. Nothing is baked into the image:

| Variable | Effect |
| --- | --- |
| `WORLDSTATE_DATASET` | `process` (default: the robosuite cell in `data/process`), `synthetic` (cartoon fallback), `real` (Exylos-style clips via `WORLDSTATE_REAL_MANIFEST`) |
| `WORLDSTATE_PERCEPTION` | `yolo` (default for the process when weights exist), `auto`, `classical`, `motion` |
| `WORLDSTATE_REAL_MANIFEST` | Your own repeated-process mp4s. Same `episodes` schema as `data/process/manifest.json` |
| `NVIDIA_API_KEY` / `NGC_API_KEY` / `NV_API_KEY` | Turns on Cosmos Reason for event text and state names |
| `NVIDIA_COSMOS_MODEL`, `COSMOS_BASE_URL` | Pin the Cosmos model / point at another NIM |
| `VAST_API_URL`, `VAST_API_KEY` | Mirror episode vectors to VAST. SQLite + FAISS stay the source of truth |
| `COSMOS_PREDICT_URL` | Recovery video generation endpoint. Stubbed without it |
| `PORT` | Server port (default 8000) |

## Demo story

1. **The graph** is learned from 16 normal runs with no labels. Robot run: At pickup → Grasping part → Lifting part → Carrying part → Descending → Placing part → Seated in fixture. States are clustered from tracked-object kinematics. Names are attached afterwards.
2. **Unseen failure** (`miss_unseen`): the cube is displaced 4.5 cm right after alignment. The run scores as a novel transition, with divergence time, expected vs observed transition, and support (seen in 0/16 reference runs).
3. **Why** explains it from the graph. **What happened** is the event text. It comes from Cosmos Reason when available and is labeled with its source, otherwise it falls back to the kinematic events.
4. **Search memory** retrieves reference runs by event text (FAISS, VAST optional).
5. **What should have happened** is the nominal path. Recovery video generation is stubbed; the graph recovery path is real.
6. **Remember this failure** adds the class `displaced_after_align`, a failure edge and a recovery edge, and bumps the world model to v2.
7. **Upload similar run** sends another miss through the real upload endpoint. It comes back as a known failure.

## Pipeline

- **Video**: `scripts/gen_process.py` runs robosuite Lift (Panda, MuJoCo on CPU) with a fixed front camera and a green target pad. The same scripted pick-and-place runs every time, with ≤5 mm start jitter and ±5% timing. The misses teleport the cube 4–6 cm along ±y at the end of the align hold. The policy is open loop, so the gripper closes on air and carries nothing. 20 normals (16 reference, 4 held out) and 4 misses, H.264, 9 s at 20 fps. `tests/test_process_dataset.py` checks the physics from simulator state.
- **Perception**: a YOLO detector (`models/worldstate-yolo.pt`; classes gripper, part, fixture) plus Ultralytics tracking, run through `worldstate.perception.yolo_track`. The detector was trained on simulator-projected boxes from the reference runs only (`scripts/train_yolo.py`). Box labels train the detector; the state graph gets no labels. Color blobs and optical flow remain fallbacks.
- **World model** (`worldstate/engine.py`): sliding-window kinematic features from tracks are clustered (HDBSCAN / k-means by silhouette), ordered by time, and turned into a transition graph. Novelty = transition surprise + embedding distance to the nearest state + trajectory abnormality.
- **Cosmos Reason** (`worldstate/adapters.py`): `nvidia/cosmos3-nano-reasoner` at `https://integrate.api.nvidia.com/v1/chat/completions` (falls back to the catalogued `nvidia/cosmos-reason2-8b` on 404). Video goes in as `video_url` at fps 4. Cosmos names clusters and writes event text. It never chooses cluster boundaries or the verdict. If a call fails, the kinematic text stays and the UI shows why.
- **Memory**: SQLite + FAISS. VAST only if `VAST_API_URL` is set.

## Eval

`python -m worldstate.eval_novelty` scores held-out normals against held-out misses. It then checks that remembering one miss makes the others a known class without flagging normals. The latest numbers are in `data/eval_report.json` after `make demo`; see the table below.

| Dataset | Graph | AUROC | Held-out normal max | Miss min | Memory check |
| --- | --- | --- | --- | --- | --- |
| Robot process, YOLO tracks | see below | see below | | | |
| Robot process, ground-truth boxes (upper bound) | 7 states, 1 path | 1.00 | 0.056 | 0.90 | — |
| Synthetic cell (fallback) | 7 states, 1 path | 1.00 | 0.054 | 1.00 | pass |

The 30 Exylos front-camera clips are domain randomized and do not hold one stable graph (path agreement ~0.62, AUROC ~0.70). They are not the deployed demo.

## Regenerate

```bash
make process   # robosuite dataset (uv, Python 3.12, mujoco 3.3.0)
make yolo      # retrain the detector from simulator boxes
```

## How it was built

The work was split across HACP pairs (bilateral contracts, frozen terms, counterparty verification), one git worktree per pair:

| Pair | Peer a | Peer b |
| --- | --- | --- |
| Simulated process | codex (ran out of quota mid-task; Claude took over the generator) | Claude Sonnet: dataset validator |
| Deploy | opencode, GLM-5.3: Dockerfile, .dockerignore | agy, Gemini 3.8 Flash: entrypoint, run script |
| Perception | Cursor, Claude Opus 5: detector training | agy, Gemini 3.1 Pro: tracking tests |

Claude Opus 5.5 integrated everything on main (dataset loader, learn/eval, Cosmos narration, naming, deploy).
