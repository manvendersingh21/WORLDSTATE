"""Record the WORLDSTATE judge-mode demo with Playwright and ffmpeg.

This script captures a narrated run-through and post-processes the recording by
speeding up detected idle waits longer than 4 seconds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
  from playwright.sync_api import Page
else:
  Page = Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "demo" / "worldstate-demo.mp4"
DEFAULT_SHOTS = ROOT / "docs" / "demo" / "shots"
DEFAULT_URL = "http://127.0.0.1:8000"
TMP_VIDEO_DIR = ROOT / "docs" / "demo" / "_capture_tmp"

FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")
MAX_BYTES = 25 * 1024 * 1024

CAPTION_JS = """(text) => {
  let bar = document.getElementById('__demo_caption_bar');
  if (!bar) {
    bar = document.createElement('div');
    bar.id = '__demo_caption_bar';
    Object.assign(bar.style, {
      position: 'fixed',
      left: '50%',
      bottom: '24px',
      transform: 'translateX(-50%)',
      zIndex: '2147483647',
      maxWidth: '1120px',
      padding: '10px 22px',
      borderRadius: '12px',
      background: 'rgba(30, 33, 39, 0.74)',
      border: '1px solid rgba(255, 255, 255, 0.16)',
      color: '#FFFFFF',
      textAlign: 'center',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
      fontWeight: '600',
      fontSize: '21px',
      letterSpacing: '0.01em',
      lineHeight: '1.32',
      boxShadow: '0 8px 26px rgba(0, 0, 0, 0.34)',
      pointerEvents: 'none'
    });
    document.body.appendChild(bar);
  }
  bar.textContent = text;
}"""

HAS_TWIN_JS = """() => {
  const root = document.getElementById('view-toggle');
  if (!root) return false;
  const twin = root.querySelector('[data-view="twin"]');
  const cam = root.querySelector('[data-view="camera"]');
  return Boolean(twin && cam);
}"""

WAIT_STATUS_JS = """(status) => {
  const badge = document.getElementById('status-badge');
  const word = document.getElementById('status-word');
  if (!badge || !word) return false;
  return badge.dataset.status === status && word.dataset.status === status;
}"""

WAIT_COSMOS_JS = """() => {
  const text = (document.getElementById('cosmos-text')?.textContent || '').trim();
  return text.length > 0 && !text.toLowerCase().startsWith('analyzing with the event model');
}"""

WAIT_REMEMBER_JS = """() => {
  const confirm = document.getElementById('remember-confirm');
  const word = document.getElementById('status-word');
  if (!confirm || !word) return false;
  const shown = !confirm.classList.contains('hidden') && confirm.textContent.includes('Failure pattern added to world memory');
  return shown && word.dataset.status === 'known_failure';
}"""

WAIT_UPLOAD_KNOWN_JS = """(before) => {
  const word = document.getElementById('status-word');
  const select = document.getElementById('run-select');
  if (!word || !select) return false;
  return word.dataset.status === 'known_failure' && !!select.value && select.value !== before;
}"""

WAIT_SEARCH_RESULTS_JS = """() => {
  return document.querySelectorAll('#search-results .result-card').length > 0;
}"""


@dataclass
class IdleWait:
  label: str
  start: float
  end: float

  @property
  def duration(self) -> float:
    return max(0.0, self.end - self.start)


class Clock:
  def __init__(self) -> None:
    self._t0 = time.monotonic()
    self.marks: dict[str, float] = {}
    self.waits: dict[str, float] = {}
    self.idle_waits: list[IdleWait] = []

  def now(self) -> float:
    return time.monotonic() - self._t0

  def mark(self, name: str) -> None:
    self.marks[name] = round(self.now(), 3)

  def timed_wait(self, label: str, fn: Callable[[], None]) -> None:
    start = self.now()
    fn()
    end = self.now()
    elapsed = end - start
    self.waits[label] = round(elapsed, 3)
    if elapsed > 4.0:
      self.idle_waits.append(IdleWait(label=label, start=start, end=end))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Capture WORLDSTATE judge demo video with Playwright.")
  parser.add_argument("--url", default=DEFAULT_URL, help=f"Demo URL (default: {DEFAULT_URL})")
  parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output mp4 path (default: {DEFAULT_OUT})")
  parser.add_argument("--shots-dir", default=str(DEFAULT_SHOTS), help=f"Screenshots directory (default: {DEFAULT_SHOTS})")
  return parser.parse_args()


def run(cmd: list[str]) -> None:
  subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float | None:
  if not FFPROBE.exists():
    return None
  try:
    out = subprocess.check_output(
      [
        str(FFPROBE),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
      ],
      text=True,
    ).strip()
    return float(out)
  except Exception:
    return None


def resolve_run_ids(page: Page) -> tuple[str, str]:
  options = page.eval_on_selector_all(
    "#run-select option",
    "opts => opts.map((o) => ({ value: o.value, text: (o.textContent || '').toLowerCase() }))",
  )
  novel = None
  normal = None
  for option in options:
    value = option["value"]
    text = option["text"]
    if value == "miss_unseen":
      novel = value
    if normal is None and "normal" in text:
      normal = value
  if novel is None and options:
    for option in options:
      if "failure" in option["text"] or "unseen" in option["text"]:
        novel = option["value"]
        break
  if novel is None:
    raise RuntimeError("Could not resolve novel run id from #run-select options.")
  if normal is None:
    for option in options:
      if option["value"] != novel:
        normal = option["value"]
        break
  if normal is None:
    raise RuntimeError("Could not resolve a normal run id from #run-select options.")
  return normal, novel


def wait_status(page: Page, status: str, timeout_ms: int = 45_000) -> None:
  page.wait_for_function(WAIT_STATUS_JS, arg=status, timeout=timeout_ms)


def wait_cosmos(page: Page, timeout_ms: int = 45_000) -> None:
  page.wait_for_function(WAIT_COSMOS_JS, timeout=timeout_ms)


def select_run(page: Page, run_id: str) -> None:
  page.select_option("#run-select", run_id)
  page.wait_for_function(
    "(expected) => document.getElementById('run-select')?.value === expected",
    arg=run_id,
    timeout=10_000,
  )


def maybe_play(page: Page) -> None:
  page.evaluate(
    """() => {
      const v = document.getElementById('video');
      if (!v) return;
      v.muted = true;
      v.play().catch(() => {});
    }"""
  )


def set_caption(page: Page, text: str) -> None:
  page.evaluate(CAPTION_JS, text)


def sleep(page: Page, seconds: float) -> None:
  page.wait_for_timeout(int(seconds * 1000))


def maybe_show_twin(page: Page, shots_dir: Path) -> bool:
  if not page.evaluate(HAS_TWIN_JS):
    return False
  page.click('#view-toggle [data-view="twin"]')
  divergence = page.evaluate(
    """() => {
      const raw = (document.getElementById('divergence-time')?.textContent || '').trim();
      const n = Number.parseFloat(raw.replace(/[^\d.]/g, ''));
      return Number.isFinite(n) ? n : 1.9;
    }"""
  )
  start_t = max(0.0, float(divergence) - 1.2)
  for i in range(6):
    page.evaluate("(t) => window.__twin?.setTime?.(t)", start_t + i * 0.85)
    sleep(page, 1.0)
  page.screenshot(path=str(shots_dir / "08-twin.png"))
  page.click('#view-toggle [data-view="camera"]')
  sleep(page, 0.6)
  return True


def build_segments(total: float, waits: list[IdleWait]) -> list[tuple[float, float, float]]:
  if total <= 0:
    return []
  waits = sorted(waits, key=lambda w: w.start)
  segments: list[tuple[float, float, float]] = []
  cur = 0.0
  for wait in waits:
    if wait.start >= total:
      break
    start = max(cur, wait.start)
    end = min(total, wait.end)
    if end <= start:
      continue
    if cur < start:
      segments.append((cur, start, 1.0))
    keep = 2.0
    keep_end = min(end, start + keep)
    if keep_end > start:
      segments.append((start, keep_end, 1.0))
    rem = end - keep_end
    if rem > 0:
      # Compress the tail of each idle wait so long backend waits do not dominate.
      speed = max(1.05, rem / 2.0)
      segments.append((keep_end, end, speed))
    cur = end
  if cur < total:
    segments.append((cur, total, 1.0))
  return [(a, b, s) for (a, b, s) in segments if b - a > 0.02]


def encode_segments(raw_video: Path, target: Path, segments: list[tuple[float, float, float]]) -> None:
  if not FFMPEG.exists():
    raise RuntimeError(f"ffmpeg not found at {FFMPEG}")
  if not segments:
    shutil.copy2(raw_video, target)
    return
  parts: list[str] = []
  labels: list[str] = []
  for idx, (start, end, speed) in enumerate(segments):
    parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=(PTS-STARTPTS)/{speed:.6f}[v{idx}]")
    labels.append(f"[v{idx}]")
  graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(segments)}:v=1:a=0,format=yuv420p[out]"
  run(
    [
      str(FFMPEG),
      "-y",
      "-loglevel",
      "error",
      "-i",
      str(raw_video),
      "-filter_complex",
      graph,
      "-map",
      "[out]",
      "-r",
      "25",
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "27",
      "-movflags",
      "+faststart",
      str(target),
    ]
  )


def normalize_duration(src: Path, dst: Path) -> float | None:
  duration = probe_duration(src)
  if duration is None:
    shutil.copy2(src, dst)
    return None
  if 55.0 <= duration <= 70.0:
    shutil.copy2(src, dst)
    return duration
  if duration > 70.0:
    speed = duration / 68.0
    vf = f"setpts=PTS/{speed:.6f},format=yuv420p"
  else:
    hold = max(0.0, 55.5 - duration)
    vf = f"tpad=stop_mode=clone:stop_duration={hold:.3f},format=yuv420p"
  run(
    [
      str(FFMPEG),
      "-y",
      "-loglevel",
      "error",
      "-i",
      str(src),
      "-vf",
      vf,
      "-r",
      "25",
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "27",
      "-movflags",
      "+faststart",
      str(dst),
    ]
  )
  return probe_duration(dst)


def enforce_size(path: Path) -> None:
  if not path.exists() or path.stat().st_size <= MAX_BYTES:
    return
  shrink = path.with_name(path.stem + ".smaller.mp4")
  run(
    [
      str(FFMPEG),
      "-y",
      "-loglevel",
      "error",
      "-i",
      str(path),
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "30",
      "-r",
      "25",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      str(shrink),
    ]
  )
  if shrink.stat().st_size < path.stat().st_size:
    shrink.replace(path)
  else:
    shrink.unlink(missing_ok=True)


def main() -> None:
  args = parse_args()
  out = Path(args.out).expanduser().resolve()
  shots_dir = Path(args.shots_dir).expanduser().resolve()
  out.parent.mkdir(parents=True, exist_ok=True)
  shots_dir.mkdir(parents=True, exist_ok=True)
  shutil.rmtree(TMP_VIDEO_DIR, ignore_errors=True)
  TMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

  clock = Clock()

  captions = {
    "normal": "Watch once. Learn the process.",
    "graph": "Normal runs match the learned path.",
    "novel": "Novel failure. Never seen before.",
    "cosmos": "Cosmos explains what went wrong.",
    "remember": "Remember this failure. One click.",
    "fix": "The fix: re-align to the moved cube. Next attempt passes.",
    "next": "Next time: a new miss, recognized instantly.",
    "upload": "A brand-new upload: known failure too.",
    "search": "Ask memory. Get the history.",
    "outro": "A camera that learns and remembers.",
    "twin": "Cosmos explains what went wrong.",
  }

  from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

  with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="chrome", args=["--autoplay-policy=no-user-gesture-required"])
    context = browser.new_context(
      viewport={"width": 1440, "height": 900},
      record_video_dir=str(TMP_VIDEO_DIR),
      record_video_size={"width": 1440, "height": 900},
    )
    page = context.new_page()
    page.goto(args.url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("#video", timeout=30_000)
    page.wait_for_selector("#run-select", timeout=30_000)
    page.wait_for_selector("#graph", timeout=30_000)
    clock.mark("page_ready")

    normal_id, novel_id = resolve_run_ids(page)

    select_run(page, normal_id)
    clock.timed_wait("wait_normal_status", lambda: wait_status(page, "normal", timeout_ms=45_000))
    clock.timed_wait("wait_normal_cosmos", lambda: wait_cosmos(page, timeout_ms=60_000))
    maybe_play(page)
    set_caption(page, captions["normal"])
    sleep(page, 3.6)
    set_caption(page, captions["graph"])
    page.screenshot(path=str(shots_dir / "01-normal.png"))
    sleep(page, 2.5)
    clock.mark("normal_complete")

    select_run(page, novel_id)
    clock.timed_wait("wait_novel_status", lambda: wait_status(page, "novel", timeout_ms=45_000))
    maybe_play(page)
    set_caption(page, captions["novel"])
    sleep(page, 3.8)
    page.screenshot(path=str(shots_dir / "02-novel.png"))
    clock.timed_wait("wait_novel_cosmos", lambda: wait_cosmos(page, timeout_ms=60_000))
    set_caption(page, captions["cosmos"])
    sleep(page, 3.0)
    page.screenshot(path=str(shots_dir / "03-cosmos.png"))
    sleep(page, 1.4)
    clock.mark("novel_complete")

    twin_used = False
    try:
      set_caption(page, captions["twin"])
      twin_used = maybe_show_twin(page, shots_dir)
      if not twin_used:
        set_caption(page, captions["cosmos"])
    except PlaywrightTimeoutError:
      twin_used = False

    set_caption(page, captions["remember"])
    page.wait_for_selector("#remember-btn:not(.hidden)", timeout=20_000)
    page.click("#remember-btn")
    clock.timed_wait("wait_remember_known", lambda: page.wait_for_function(WAIT_REMEMBER_JS, timeout=60_000))
    sleep(page, 2.0)
    page.screenshot(path=str(shots_dir / "04-remember-known.png"))
    clock.mark("remember_complete")

    # The fix: the corrected attempt (simulated closed-loop re-grasp) passes.
    if page.locator("#fix-play-btn").count():
      set_caption(page, captions["fix"])
      page.click("#fix-play-btn")
      sleep(page, 2.0)
      view = page.locator('[data-view="twin"]')
      if view.count():
        view.first.click()
        sleep(page, 6.0)
        page.screenshot(path=str(shots_dir / "04b-fix-3d.png"))
        page.locator('[data-view="camera"]').first.click()
      clock.timed_wait("wait_fix_confirm", lambda: page.wait_for_selector("#fix-confirm", timeout=20_000))
      sleep(page, 2.0)
      page.screenshot(path=str(shots_dir / "04c-fix-passed.png"))
      clock.mark("fix_complete")

    # The payoff: a different miss the model has never seen is now a KNOWN FAILURE.
    select_run(page, "miss_eval_right")
    clock.timed_wait("wait_next_known", lambda: wait_status(page, "known_failure", timeout_ms=45_000))
    maybe_play(page)
    set_caption(page, captions["next"])
    sleep(page, 4.5)
    page.screenshot(path=str(shots_dir / "05-next-known.png"))
    sleep(page, 1.0)
    clock.mark("next_known_complete")

    set_caption(page, captions["upload"])
    before_upload = page.input_value("#run-select")
    page.click("#upload-similar-btn")
    clock.timed_wait(
      "wait_upload_known",
      lambda: page.wait_for_function(WAIT_UPLOAD_KNOWN_JS, arg=before_upload, timeout=90_000),
    )
    maybe_play(page)
    clock.timed_wait("wait_upload_cosmos", lambda: wait_cosmos(page, timeout_ms=60_000))
    sleep(page, 2.2)
    page.screenshot(path=str(shots_dir / "06-upload-known.png"))
    clock.mark("upload_complete")

    set_caption(page, captions["search"])
    search = page.locator("#search-input")
    search.click()
    search.fill("")
    search.type("show me failed grasps", delay=52)
    page.click("#search-btn")
    clock.timed_wait("wait_search_results", lambda: page.wait_for_function(WAIT_SEARCH_RESULTS_JS, timeout=45_000))
    if page.locator("#search-results .result-card").count() > 0:
      page.locator("#search-results .result-card").first.click()
      sleep(page, 0.8)
    sleep(page, 2.4)
    page.screenshot(path=str(shots_dir / "07-search.png"))
    clock.mark("search_complete")

    set_caption(page, captions["outro"])
    sleep(page, 2.4)
    clock.mark("capture_end")

    raw_path = Path(page.video.path())
    context.close()
    browser.close()

  raw_total = max(clock.now(), 1.0)
  tmp_sped = out.with_name(out.stem + ".sped.mp4")
  tmp_final = out.with_name(out.stem + ".final.mp4")
  segments = build_segments(raw_total, clock.idle_waits)
  encode_segments(raw_path, tmp_sped, segments)
  final_duration = normalize_duration(tmp_sped, tmp_final)
  tmp_final.replace(out)
  enforce_size(out)

  meta = {
    "url": args.url,
    "output": str(out),
    "shots_dir": str(shots_dir),
    "raw_duration_s": round(raw_total, 3),
    "final_duration_s": round(final_duration, 3) if isinstance(final_duration, float) else None,
    "idle_waits": [
      {
        "label": wait.label,
        "start_s": round(wait.start, 3),
        "end_s": round(wait.end, 3),
        "duration_s": round(wait.duration, 3),
      }
      for wait in clock.idle_waits
    ],
    "step_timestamps_s": clock.marks,
    "wait_durations_s": clock.waits,
    "segments": [
      {"start_s": round(start, 3), "end_s": round(end, 3), "speed": round(speed, 3)}
      for start, end, speed in segments
    ],
    "shots": [
      "01-normal.png",
      "02-novel.png",
      "03-cosmos.png",
      "04-remember-known.png",
      "05-next-known.png",
      "06-upload-known.png",
      "07-search.png",
    ] + (["08-twin.png"] if twin_used else []),
  }
  meta_path = out.with_suffix(".timestamps.json")
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

  tmp_sped.unlink(missing_ok=True)
  shutil.rmtree(TMP_VIDEO_DIR, ignore_errors=True)

  print(json.dumps({"wrote": str(out), "meta": str(meta_path), "bytes": out.stat().st_size}, indent=2))


if __name__ == "__main__":
  main()
