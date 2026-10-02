"""Record the WORLDSTATE backup demo video (Playwright + ffmpeg).

Usage: python scripts/capture_demo.py [--base-url http://127.0.0.1:8000]
Needs: playwright (+ chromium). Long waits (> ~8 s) are sped up in post.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
SHOTS = DEMO / "shots"
TMP = DEMO / "_tmp"
MP4 = DEMO / "worldstate-demo.mp4"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

CAPTION_JS = """(t) => {
  let o = document.getElementById('demo-cap');
  if (!o) {
    o = document.createElement('div'); o.id = 'demo-cap';
    Object.assign(o.style, {position:'fixed', top:'10px', left:'50%', transform:'translateX(-50%)',
      zIndex:'2147483647', padding:'12px 26px', borderRadius:'12px', background:'#000000f0',
      color:'#ffe94d', border:'2px solid #ffe94d', font:'700 22px Arial, sans-serif',
      textAlign:'center', maxWidth:'1100px', boxShadow:'0 8px 28px #000a', pointerEvents:'none'});
    document.body.appendChild(o);
  }
  o.textContent = t;
}"""

# Case-insensitive; also accepts the older banner wording served by pre-badge builds.
ALERT_JS = """(w) => {
  const t = (document.getElementById('alert')?.textContent || '').toLowerCase();
  const alts = {NORMAL: ['normal', 'consistent with'], 'NOVEL FAILURE': ['novel'], 'KNOWN FAILURE': ['known failure']}[w];
  return alts.some((a) => t.includes(a));
}"""
NARR_JS = """() => {
  const b = (document.getElementById('narration')?.textContent || '').trim();
  const s = (document.getElementById('narration-source')?.textContent || '').trim();
  return b.length > 0 && !b.includes('Asking the event model') && s.length > 0;
}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    base = ap.parse_args().base_url.rstrip("/")
    SHOTS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    waits: list[tuple[float, float]] = []  # long idle segments (video seconds)
    timings: dict[str, float] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                                  record_video_dir=str(TMP), record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()
        t0 = time.time()
        now = lambda: time.time() - t0  # noqa: E731
        cap = lambda t: page.evaluate(CAPTION_JS, t)  # noqa: E731
        shot = lambda n: page.screenshot(path=str(SHOTS / n))  # noqa: E731
        sleep = lambda s: page.wait_for_timeout(int(s * 1000))  # noqa: E731

        def top() -> None:  # scroll page and the right-hand insight rail back to the top
            page.evaluate("() => { window.scrollTo(0,0); document.querySelectorAll('.insight, main, .shell').forEach((e) => { e.scrollTop = 0; }); }")

        def play() -> None:
            page.evaluate("() => { const v = document.getElementById('player'); v.muted = true; v.play().catch(()=>{}); }")

        def run(rid: str) -> None:
            page.click(f"#episode-list button[data-episode='{rid}']")

        def wait_known(not_id: str, label: str) -> None:
            a = now()
            page.wait_for_function(
                """(nid) => {
                  const act = document.querySelector('#episode-list .ep.active');
                  return act && act.dataset.episode !== nid
                    && (document.getElementById('alert')?.textContent || '').toLowerCase().includes('known failure');
                }""" if not_id else ALERT_JS, arg=not_id or "KNOWN FAILURE", timeout=90_000)
            timings[label] = now() - a
            if now() - a > 8:
                waits.append((a, now()))

        page.goto(base, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector("#episode-list button[data-episode='normal_16']", timeout=30_000)
        page.wait_for_selector("#graph *", state="attached", timeout=30_000)
        page.evaluate("document.getElementById('learn-btn') && (document.getElementById('learn-btn').style.display='none')")

        # 1
        cap("WORLDSTATE learned this pick-and-place process from 16 unlabeled robot videos")
        page.locator("#graph").scroll_into_view_if_needed()
        sleep(1.5); shot("01-graph.png"); sleep(2.5)

        # 2
        page.evaluate("window.scrollTo(0,0)")
        run("normal_16")
        cap("Normal run: YOLO tracks gripper, cube, target")
        page.wait_for_function(ALERT_JS, arg="NORMAL", timeout=45_000)
        play(); sleep(5)
        cap("Cosmos Reason explains what happened")
        page.wait_for_function(NARR_JS, timeout=90_000)
        page.locator("#narration").scroll_into_view_if_needed()
        sleep(3); shot("02-normal.png"); sleep(3)

        # 3
        page.evaluate("window.scrollTo(0,0)")
        run("miss_unseen")
        cap("Unseen failure: the cube shifts after alignment, the grasp misses")
        page.wait_for_function(ALERT_JS, arg="NOVEL FAILURE", timeout=45_000)
        play(); sleep(4)
        page.locator("#why").scroll_into_view_if_needed()
        cap("Novel transition at 1.9 s, seen in 0/16 reference runs; Cosmos explains the failure")
        page.wait_for_function(NARR_JS, timeout=90_000)
        page.hover("#narration"); sleep(3)
        shot("03-novel.png"); sleep(3.5)

        # 4
        cap("Memory search (SQLite + FAISS)")
        page.fill("#search-q", "")
        page.type("#search-q", "show me failed grasps", delay=60)
        page.press("#search-q", "Enter")
        page.wait_for_function("() => document.getElementById('search-results').children.length > 0", timeout=45_000)
        page.locator("#search-results").scroll_into_view_if_needed()
        sleep(2); shot("04-search.png"); sleep(3)

        # 5
        page.locator("#remember-btn").scroll_into_view_if_needed()
        cap("Remember this failure → world model v2")
        sleep(1.5)
        page.click("#remember-btn")
        wait_known("", "remember_s")
        top()
        sleep(1); page.wait_for_function(NARR_JS, timeout=90_000)
        sleep(3); shot("05-known.png"); sleep(2)

        # 6
        page.locator("#upload-similar-btn").scroll_into_view_if_needed()
        cap("A similar miss arrives → recognized as a KNOWN FAILURE")
        sleep(1.5)
        page.click("#upload-similar-btn")
        wait_known("miss_unseen", "upload_s")
        play()
        top()
        try:
            page.wait_for_function(NARR_JS, timeout=90_000)
        except Exception:
            pass
        sleep(3); shot("06-upload-known.png"); sleep(3)

        # 7
        cap("WORLDSTATE · NVIDIA Cosmos Reason · YOLO · SQLite+FAISS (VAST optional)")
        sleep(3)
        total = now()
        video_path = page.video.path()
        ctx.close(); browser.close()

    print("timings", json.dumps(timings), "waits", waits, "total", round(total, 1))
    # Build filter: speed up the tail of long waits, keep the first 3 s real-time.
    segs: list[tuple[float, float, float]] = []  # (start, end, speed)
    cur = 0.0
    for a, b in waits:
        segs += [(cur, a + 3, 1.0)]
        if b - (a + 3) > 0.5:
            segs += [(a + 3, b, max(1.0, (b - a - 3) / 3.0))]
        cur = b
    segs.append((cur, total + 2, 1.0))
    parts, labels = [], []
    for i, (s, e, sp) in enumerate(segs):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=(PTS-STARTPTS)/{sp:.3f}[v{i}]")
        labels.append(f"[v{i}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(segs)}:v=1:a=0,format=yuv420p[out]"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video_path), "-filter_complex", fc,
                    "-map", "[out]", "-c:v", "libx264", "-crf", "26", "-preset", "medium", "-r", "25",
                    "-movflags", "+faststart", str(MP4)], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print("wrote", MP4, MP4.stat().st_size)


if __name__ == "__main__":
    main()
