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

        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector("#run-select")
        page.wait_for_timeout(2500)

        select("normal_16", "normal")
        play_video()
        beat("WORLDSTATE: physical process intelligence",
             "This is WORLDSTATE. It watches a repeated physical process, here a robot pick-and-place cell, and learns how that process should behave, with no labels and no rulebook.", 8)
        page.evaluate("() => document.querySelector('#graph')?.scrollIntoView({behavior:'smooth', block:'center'})")
        beat("Learned from 16 unlabeled robot videos",
             "From sixteen unlabeled videos, YOLO tracks the gripper, the cube and the target, and WORLDSTATE discovers the process on its own: pickup, grasp, lift, carry, descend, place, seated.", 9)
        page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
        beat("A normal run matches the learned process",
             "This is a normal run. It follows the learned path, so it is marked NORMAL. NVIDIA Cosmos Reason, running on the event's GPU endpoint, describes what happened in plain language.", 9)
        view("twin")
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(4.0)")
        beat("The same run as a 3D digital twin",
             "The same run can be viewed as a 3D digital twin. It replays the simulator's recorded trajectory, in sync with the camera and with WORLDSTATE's states.", 8)
        view("camera")

        select("miss_unseen", "novel")
        play_video()
        beat("Now a run it has never seen",
             "Now, a run WORLDSTATE has never seen. Watch the cube right after the gripper lines up.", 6)
        beat("NOVEL FAILURE at 1.904 seconds",
             "At one point nine zero four seconds, reality diverges. The cube shifts sideways after alignment. WORLDSTATE flags a NOVEL FAILURE: it expected grasping, it observed the cube sliding sideways, and that transition appeared in zero of sixteen reference runs.", 12)
        beat("Cosmos Reason explains the failure",
             "Cosmos Reason explains it: the gripper closes, but the cube has already moved, so the grasp misses and nothing reaches the target.", 8)
        view("twin")
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(2.3)")
        page.wait_for_timeout(1200)
        beat("In 3D: the cube displaced 4.5 cm, the gripper still targeting the old position",
             "In the digital twin, the callouts show exactly what went wrong: the cube was displaced four and a half centimeters, while the gripper kept targeting the original position.", 9)
        page.evaluate("() => window.__twin && window.__twin.setTime && window.__twin.setTime(3.6)")
        beat("The gripper closes on empty air",
             "Then the gripper closes on empty air. The cube is never lifted.", 6)
        view("camera")

        page.click("#remember-btn")
        page.wait_for_function("() => document.querySelector('#status-badge')?.dataset.status === 'known_failure'", timeout=60000)
        beat("Remember this failure: added to world memory",
             "One click: remember this failure. The pattern is added to WORLDSTATE's world memory, and the learned graph gets a new, named branch: displaced after alignment.", 9)
        beat("The fix: a recovery plan from memory",
             "Because it now knows this failure, WORLDSTATE also knows the fix: re-align to where the cube actually is, then grasp, lift, carry and place.", 8)
        if page.locator("#fix-play-btn").count():
            page.click("#fix-play-btn")
            beat("Corrected attempt: re-aligned, grasped, placed",
                 "Here is the corrected attempt in the same simulated cell, with the same four and a half centimeter shift. The robot re-aligns to the moved cube and grasps it.", 6)
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
             "And the next time a similar failure happens, here a different run it has never seen, WORLDSTATE recognizes it instantly as a KNOWN FAILURE, and the fix is ready.", 10)

        page.fill("#search-input", "show me failed grasps")
        page.press("#search-input", "Enter")
        page.wait_for_timeout(1500)
        page.evaluate("() => document.querySelector('#search-results')?.scrollIntoView({behavior:'smooth', block:'center'})")
        beat("Search world memory in plain language",
             "Everything goes into world memory. Ask it in plain language, show me failed grasps, and it returns the failures.", 8)

        page.goto(args.url.rstrip("/") + "/static/vast/index.html", wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        beat("Our team's VAST corpus: Cosmos captions + YOLO11 detections",
             "We also pulled our team's footage from the VAST video index, with the VAST pipeline's own Cosmos captions and YOLO detections, searchable in the same interface.", 9)
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector("#run-select")
        page.wait_for_timeout(1500)
        beat("WORLDSTATE: learns the process, catches the first failure, remembers the fix",
             "WORLDSTATE learns how a physical process should behave, catches the first time reality diverges, and remembers the fix. For factories, warehouses and labs: one camera that learns.", 9)
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
    print("wrote", OUT, "and", SCRIPT)


if __name__ == "__main__":
    main()
