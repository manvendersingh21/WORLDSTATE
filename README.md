# WORLDSTATE

Can a model watch repeated videos of one physical process, discover the state graph without a SOP, and then explain a run it has never seen?

This is a hackathon prototype for that question. It learns a pick-and-place graph from normal videos, flags a held-out miss as a novel transition, and, after you confirm it, recognizes the next similar miss as a known failure.

## One command

```bash
make demo
```

That creates a virtualenv, generates the synthetic cell if needed, learns the graph, runs the novelty eval, and serves the UI at http://127.0.0.1:8000.

## Demo story

1. The graph is the process discovered from 16 normal runs. States are named only after clustering.
2. Open **Unseen failure**. The alert is a novel transition: score, first divergence time, expected versus observed, and how many reference runs contained that transition (0/16).
3. **Why** explains it from the graph. **Search memory** retrieves reference runs that shared the prefix.
4. **What should have happened** is the nominal path. Recovery video generation is stubbed.
5. **Remember this failure** adds the class `displaced_after_align`, a failure edge, and a recovery edge, and bumps the world-model version.
6. **Upload similar run** sends another miss through the real upload endpoint. It comes back as a known failure.

Held-out normal runs stay in distribution (score about 0.05 versus 1.0 for misses).

## Data

- Synthetic cell: `python -m worldstate.synthetic` writes 20 normal runs and 4 displacement-after-align misses with OpenCV. The generator is the demo, because the repeated process is actually repeated.
- Real clips: the loader reads `internal/datasets/manifest.json` (30 Exylos front-camera MP4s: 20 success, 6 slip_drop, 4 operator_abort). On `make demo` those clips are assessed first. They are domain-randomized, so success runs do not share one stable path and failures overlap normals (path agreement about 0.62, AUROC about 0.70). The UI then learns the synthetic cell and says so.
- Any other folder of mp4s: `WORLDSTATE_REAL_MANIFEST=/path/manifest.json` or drop files and `POST /api/upload`.

`WORLDSTATE_DATASET=synthetic` skips the real-clip pass. `WORLDSTATE_DATASET=real` forces the real graph even when it is unstable.

## Eval

`python -m worldstate.eval_novelty` scores held-out normals against held-out misses, then checks that remembering one miss makes the others a known class without flagging normals. Latest synthetic result: AUROC 1.0, full separation, memory check passed.

## Adapters

| Service | When it is used | If it is missing |
| --- | --- | --- |
| Cosmos Reason | `NVIDIA_API_KEY` (or `NGC_API_KEY` / `NV_API_KEY`). Hosted model `nvidia/cosmos3-nano-reasoner` at `https://integrate.api.nvidia.com/v1/chat/completions`, video as `video_url`, `media_io_kwargs.video.fps=4`. `COSMOS_BASE_URL` or `NVIDIA_BASE_URL` points at another NIM. | Kinematic events and centroid names |
| YOLO / YOLO-World | `WORLDSTATE_PERCEPTION=yolo` and `ultralytics` installed | Color blobs, then optical flow |
| Sentence-transformers | `WORLDSTATE_TEXT_ENCODER=sentence-transformers` | Hashing embedder |
| FAISS | Installed with the core requirements | NumPy inner product |
| VAST | `VAST_API_URL` | SQLite only |
| Cosmos Predict | `COSMOS_PREDICT_URL` | Recovery prompt stub, no pixels |

The API key is read from the environment only. It is not written to disk and not printed.
