"""Record a real-time WORLDSTATE walkthrough for a voice-over (no fast-forward).

    python scripts/record_voiceover_demo.py [--url http://127.0.0.1:8000]
Writes docs/demo/worldstate-voiceover.mp4 and docs/demo/voiceover-script.md (lines timed to the video).
Leaves the app at world model v2 (Remember is clicked); reset afterwards with POST /api/learn.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo" / "worldstate-voiceover.mp4"
SCRIPT = ROOT / "docs" / "demo" / "voiceover-script.md"
TMP = ROOT / "docs" / "demo" / "_vo_tmp"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

CAPTION_JS = """(text) => {
  let bar = document.getElementById('__vo_caption');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = '__vo_caption';
    Object.assign(bar.style, {position:'fixed', left:'50%', bottom:'26px', transform:'translateX(-50%)',
      zIndex: 99999, padding:'12px 22px', borderRadius:'10px', background:'rgba(14,17,22,0.88)',
      border:'1px solid rgba(255,255,255,0.12)', color:'#fff', font:'600 21px Inter, system-ui, sans-serif',
      maxWidth:'1100px', textAlign:'center', transition:'opacity .3s'});
    document.body.appendChild(bar);
  }
  bar.style.opacity = text ? '1' : '0';
  bar.textContent = text || '';
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    beats: list[tuple[float, str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(TMP),
                                  record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()
        t0 = time.monotonic()

        def beat(caption: str, voice: str, hold: float) -> None:
            beats.append((time.monotonic() - t0, caption, voice))
            page.evaluate(CAPTION_JS, caption)
            page.wait_for_timeout(int(hold * 1000))

        def select(run: str, status: str) -> None:
            page.select_option("#run-select", run)
            page.wait_for_function("(s) => document.querySelector('#status-badge')?.dataset.status === s", arg=status, timeout=45000)
            page.wait_for_function("() => (document.querySelector('#cosmos-text')?.textContent || '').trim().length > 20", timeout=60000)

        def play_video() -> None:
            page.evaluate("() => { const v = document.querySelector('#video'); if (v) { v.currentTime = 0; v.play(); } }")

        def view(name: str) -> None:
            page.click(f'[data-view="{name}"]')
            page.wait_for_timeout(2500 if name == "twin" else 800)

        page.goto(args.url.rstrip("/") + "/static/architecture.html", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        beat("Architecture: video → YOLO → unsupervised world model → novelty",
             "Here is how it works. A fixed camera watches a repeated process. YOLO tracks the gripper, the cube, and the target in every frame. The system learns how the process should flow, without labels or rules, and scores every run: normal, novel, or known failure.", 18)
        beat("NVIDIA Cosmos Reason explains · World memory remembers · VAST corpus",
             "When a run fails, NVIDIA Cosmos Reason, on CoreWeave GPUs, explains it in plain language. Every run lands in memory built on SQLite and FAISS, searchable in plain words, alongside our team's VAST video corpus with its captions and detections.", 16)
        beat("Remember → fix → the next failure is known",
             "Here is the key loop. When an operator confirms a new failure, WORLDSTATE remembers it and learns the recovery. The next similar failure is recognized instantly, with the fix ready, all shown alongside a 3D digital twin.", 15)

        beat("Why it's different: no labels, no rules, catches the first-ever failure",
             "Why is this different? Most video AI today searches footage after the fact, or detects what someone defined in advance. That breaks on the failure nobody has seen yet. WORLDSTATE learns the normal process on its own, no labels, no rules, catches the first divergence, pinpoints when and why, and turns one confirmation into a remembered fix.", 22)

        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector("#run-select")
        page.wait_for_timeout(2500)

        select("normal_16", "normal")
        play_video()
        beat("WORLDSTATE: physical process intelligence",
             "This is WORLDSTATE. It watches a repeated physical process, a robot pick-and-place cell, and learns how it should behave.", 8)
        page.evaluate("() => document.querySelector('#graph')?.scrollIntoView({behavior:'smooth', block:'center'})")
        beat("Learned from 16 unlabeled robot videos",
             "From sixteen unlabeled videos, YOLO tracks every object, and WORLDSTATE discovers the steps itself: pickup, grasp, lift, carry, descend, place, seated.", 9)
        page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
        beat("A normal run matches the learned process",
             "This is a normal run. It follows the learned path, so it's marked normal, and NVIDIA Cosmos Reason describes it in plain language.", 9)
        view("twin")
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(4.0)")
        beat("The same run as a 3D digital twin",
             "Here is the same run as a 3D digital twin, replaying the simulator's recorded trajectory in sync with the camera.", 8)
        view("camera")

        select("miss_unseen", "novel")
        play_video()
        beat("Now a run it has never seen",
             "Now, a run WORLDSTATE has never seen. Watch the cube after the gripper lines up.", 6)
        beat("NOVEL FAILURE at 1.904 seconds",
             "At one point nine zero four seconds, the cube shifts sideways. WORLDSTATE flags a novel failure: it expected a grasp, saw a slide, a pattern seen in zero of sixteen runs.", 12)
        beat("Cosmos Reason explains the failure",
             "Cosmos Reason explains it: the cube has already moved, so the grasp misses and nothing reaches the target.", 8)
        view("twin")
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(2.3)")
        page.wait_for_timeout(1200)
        beat("In 3D: the cube displaced 4.5 cm, the gripper still targeting the old position",
             "In the digital twin, the cube sits four and a half centimeters away, while the gripper still targets the old position.", 9)
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(3.6)")
        beat("The gripper closes on empty air",
             "Then the gripper closes on empty air. The cube is never lifted.", 6)
        view("camera")

        page.click("#remember-btn")
        page.wait_for_function("() => document.querySelector('#status-badge')?.dataset.status === 'known_failure'", timeout=60000)
        beat("Remember this failure: added to world memory",
             "One click: remember this failure. It's added to world memory, and the graph gets a new branch: displaced after alignment.", 9)
        beat("The fix: a recovery plan from memory",
             "Because it knows this failure, WORLDSTATE knows the fix: re-align to the cube's real position, then grasp and place.", 8)
        if page.locator("#fix-play-btn").count():
            page.click("#fix-play-btn")
            beat("Corrected attempt: re-aligned, grasped, placed",
                 "Here is the corrected attempt, simulated: the robot re-aligns to the cube and grasps it.", 6)
            view("twin")
            page.evaluate("() => { const v = document.querySelector('#video'); if (v) { v.currentTime = 4.5; v.play(); } }")
            beat("The next attempt passes",
                 "It lifts the cube, carries it, and places it on the target. The next attempt passes.", 7)
            view("camera")
            page.evaluate("() => { const v = document.querySelector('#video'); if (v) { v.currentTime = 8.6; v.play(); } }")
            try:
                page.wait_for_selector("#fix-confirm", timeout=12000)
            except Exception:
                pass
            beat("Next attempt passes: placed on target", "Placed on target.", 3)

        select("miss_eval_right", "known_failure")
        play_video()
        beat("Next time: a different miss, recognized instantly",
             "The next time something similar happens, in a run it has never seen, WORLDSTATE recognizes it instantly as a known failure, and the fix is ready.", 10)

        page.fill("#search-input", "show me failed grasps")
        page.press("#search-input", "Enter")
        page.wait_for_timeout(1500)
        page.evaluate("() => document.querySelector('#search-results')?.scrollIntoView({behavior:'smooth', block:'center'})")
        beat("Search world memory in plain language",
             "Everything goes into world memory. Ask it in plain language, show me failed grasps, and it returns the failures.", 8)

        page.goto(args.url.rstrip("/") + "/static/vast/index.html", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        beat("Our team's VAST corpus: Cosmos captions + YOLO11 detections",
             "We also connected our team's VAST video corpus, with its Cosmos captions and YOLO detections, searchable in the same interface.", 9)
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector("#run-select")
        page.wait_for_timeout(1500)
        beat("WORLDSTATE: learns the process, catches the first failure, remembers the fix",
             "WORLDSTATE learns how a process should behave, catches the first divergence, and remembers the fix. Factories, warehouses, labs: one camera that learns.", 9)
        page.evaluate(CAPTION_JS, "")
        page.wait_for_timeout(800)
        ctx.close()
        raw = next(TMP.glob("*.webm"))
        browser.close()

    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    lines = ["# WORLDSTATE voice-over script", "",
             f"Timed to `docs/demo/{OUT.name}`. Paste each line into text-to-speech and place it at the start time.", "",
             "| Start | On screen | Voice-over |", "| --- | --- | --- |"]
    for start, caption, voice in beats:
        lines.append(f"| {int(start // 60)}:{start % 60:04.1f} | {caption} | {voice} |")
    lines += ["", "## Full narration (one block)", "", " ".join(v for _, _, v in beats)]
    SCRIPT.write_text("\n".join(lines) + "\n")
    import json
    entries = []
    for i, (start, caption, voice) in enumerate(beats):
        end = beats[i + 1][0] if i + 1 < len(beats) else start + 9.0
        entries.append({"index": i + 1, "start_s": round(start, 2), "end_s": round(end, 2),
                        "on_screen": caption, "voiceover": voice})
    (SCRIPT.parent / "voiceover-transcript.json").write_text(json.dumps(
        {"video": f"docs/demo/{OUT.name}", "segments": entries,
         "full_text": " ".join(v for _, _, v in beats)}, indent=2))
    print("wrote", OUT, "and", SCRIPT)


if __name__ == "__main__":
    main()
