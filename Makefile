PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: demo install data learn eval serve

install: .venv/bin/python

.venv/bin/python:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

data: install
	$(PY) -m worldstate.synthetic --force

learn: install
	$(PY) -m worldstate.learn

eval: install
	$(PY) -m worldstate.eval_novelty

serve: install
	$(PY) -m uvicorn worldstate.server:app --host 0.0.0.0 --port 8000

demo: install
	$(PY) -m worldstate.synthetic
	$(PY) -m worldstate.learn
	$(PY) -m worldstate.eval_novelty
	@echo "WORLDSTATE http://127.0.0.1:8000"
	$(PY) -m uvicorn worldstate.server:app --host 0.0.0.0 --port 8000

.PHONY: process yolo deploy

# Regenerate the robosuite Lift repeated process (CPU MuJoCo is enough).
process:
	uv venv -p 3.12 .venv-sim
	uv pip install -p .venv-sim/bin/python -r requirements-sim.txt
	.venv-sim/bin/python scripts/gen_process.py --out data/process --force

# Retrain the detector from simulator boxes (the world model itself never sees labels).
yolo:
	uv venv -p 3.12 .venv-yolo
	uv pip install -p .venv-yolo/bin/python -r requirements.txt -r requirements-yolo.txt
	.venv-yolo/bin/python scripts/train_yolo.py

# Build and run the container on 0.0.0.0:$${PORT:-8000}. Config comes from the environment / .env.
deploy:
	scripts/docker-run.sh
