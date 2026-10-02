"""Drive the demo UI and save screenshots plus a short recording."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/cursor/stores/bc-01a0fcc2-e2d9-74ce-8e08-fea622ea1f27/media")
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video_dir = OUT / "_video_tmp"
    video_dir.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1600, "height": 1000},
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Novel transition", timeout=20000)
        page.wait_for_function("() => document.getElementById('player').currentTime > 4", timeout=10000)
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "worldstate-01-novel-transition.png"))
        page.click("#search-btn")
        page.wait_for_selector("#search-results article", timeout=10000)
        page.screenshot(path=str(OUT / "worldstate-02-memory-search.png"))
        page.click("#remember-btn")
        page.wait_for_selector("text=Known failure", timeout=20000)
        page.wait_for_selector("text=world model v2", timeout=10000)
        page.screenshot(path=str(OUT / "worldstate-03-remembered-failure.png"))
        page.click("#upload-similar-btn")
        page.wait_for_selector("text=upload_", timeout=20000)
        page.wait_for_selector("#alert.known", timeout=20000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "worldstate-04-known-similar-run.png"))
        video = page.video
        context.close()
        browser.close()
        if video is not None:
            webm = Path(video.path())
            target = OUT / "worldstate-demo.webm"
            webm.replace(target)
            print("video", target)
    print("screenshots", list(OUT.glob("worldstate-*.png")))


if __name__ == "__main__":
    main()
