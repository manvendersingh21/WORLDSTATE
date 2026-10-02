import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:8101"
EXPECTED_GLB_NAMES = {
    "link0.glb",
    "link1.glb",
    "link2.glb",
    "link3.glb",
    "link4.glb",
    "link5.glb",
    "link6.glb",
    "link7.glb",
    "hand.glb",
    "leftfinger.glb",
    "rightfinger.glb",
}
IGNORED_CONSOLE_ERRORS = {
    "Failed to load resource: the server responded with a status of 404 (Not Found)",
    "Failed to load resource: the server responded with a status of 404 (File not found)",
}


def _wait_http_ready(timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{BASE_URL}/api/health", timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - startup race
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"Judge UI server not ready at {BASE_URL}: {last_error}")


def _assert_not_raw_title(title: str) -> None:
    normalized = (title or "").strip().lower()
    assert normalized, "result title should not be empty"
    assert not re.fullmatch(r"(normal|miss|upload|reference|heldout|eval)[a-z0-9_]*", normalized), (
        f"result title should be human-readable, got raw-looking id: {title!r}"
    )


def _video_metrics(page) -> dict:
    return page.evaluate(
        """
        () => {
          const v = document.querySelector("#video");
          if (!v) return { ok: false, reason: "missing video" };
          const r = v.getBoundingClientRect();
          const cs = getComputedStyle(v);
          return {
            ok: true,
            width: r.width,
            height: r.height,
            display: cs.display,
            visibility: cs.visibility,
            opacity: cs.opacity,
            paused: v.paused,
            readyState: v.readyState,
            currentTime: v.currentTime,
          };
        }
        """
    )


def _canvas_probe(page) -> dict:
    return page.evaluate(
        """() => {
            const canvas = document.getElementById('twin-canvas');
            if (!canvas) return { ok: false, reason: "missing canvas" };
            const probe = document.createElement('canvas');
            probe.width = canvas.width;
            probe.height = canvas.height;
            const ctx = probe.getContext('2d');
            ctx.drawImage(canvas, 0, 0);
            const bg = [8, 10, 13];
            const pts = [[0.5,0.5],[0.25,0.25],[0.75,0.25],[0.25,0.75],[0.75,0.75],[0.5,0.2],[0.2,0.5]];
            let nonBg = 0;
            for (const [ux, uy] of pts) {
                const x = Math.max(0, Math.min(probe.width - 1, Math.floor(probe.width * ux)));
                const y = Math.max(0, Math.min(probe.height - 1, Math.floor(probe.height * uy)));
                const d = ctx.getImageData(x, y, 1, 1).data;
                if (Math.abs(d[0]-bg[0]) + Math.abs(d[1]-bg[1]) + Math.abs(d[2]-bg[2]) > 12) nonBg += 1;
            }
            return { ok: true, nonBg, width: probe.width, height: probe.height };
        }"""
    )


def test_judge_twin_toggle_flow():
    _wait_http_ready()
    console_errors: list[str] = []
    page_errors: list[str] = []
    glb_requests: set[str] = set()
    glb_status: dict[str, int] = {}
    glb_failed: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--use-angle=metal"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def on_console(msg):
            if msg.type == "error" and msg.text not in IGNORED_CONSOLE_ERRORS:
                console_errors.append(msg.text)

        def on_page_error(exc):
            page_errors.append(str(exc))

        def on_request(req):
            name = Path(urlparse(req.url).path).name
            if name.endswith(".glb"):
                glb_requests.add(name)

        def on_response(resp):
            name = Path(urlparse(resp.url).path).name
            if name.endswith(".glb"):
                glb_status[name] = resp.status

        def on_request_failed(req):
            name = Path(urlparse(req.url).path).name
            if name.endswith(".glb"):
                glb_failed.append(f"{name}: {req.failure}")

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector("#run-select", timeout=120_000)
        page.wait_for_selector("#view-toggle button[data-view='camera']", timeout=120_000)
        page.wait_for_selector("#view-toggle button[data-view='twin']", timeout=120_000)

        page.wait_for_function(
            """
            () => {
              const v = document.querySelector("#video");
              if (!v || v.readyState < 2) return false;
              v.muted = true;
              const p = v.play();
              if (p && typeof p.then === "function") {
                p.catch(() => {});
              }
              return true;
            }
            """,
            timeout=60_000,
        )
        video = _video_metrics(page)
        assert video["ok"], f"video should exist: {video}"
        assert video["display"] != "none" and video["visibility"] != "hidden"
        assert float(video["opacity"]) > 0.0, f"video should be visible in CAMERA mode: {video}"
        assert video["height"] >= 450.0, f"video rendered height should be >= 450px, got {video['height']:.1f}px"

        assert page.locator("#view-toggle button[data-view='camera']").evaluate(
            "el => el.classList.contains('active') || el.getAttribute('aria-pressed') === 'true'"
        ), "CAMERA should be active by default"

        # No twin instance should exist before first explicit twin toggle click.
        assert page.evaluate("() => !window.__twin"), "window.__twin should be undefined before twin view mount"

        page.click("#view-toggle button[data-view='twin']")
        page.wait_for_function(
            """
            () => {
              const mount = document.querySelector("#twin-mount");
              const canvas = document.querySelector("#twin-canvas");
              return !!window.__twin && !!mount && !!canvas &&
                getComputedStyle(mount).display !== "none";
            }
            """,
            timeout=120_000,
        )

        deadline = time.time() + 25.0
        while len(glb_status) < len(EXPECTED_GLB_NAMES) and time.time() < deadline:
            page.wait_for_timeout(150)

        missing_requested = EXPECTED_GLB_NAMES - glb_requests
        missing_loaded = EXPECTED_GLB_NAMES - set(glb_status.keys())
        bad_status = {name: status for name, status in glb_status.items() if status >= 400}
        assert not missing_requested, f"Missing GLB requests: {sorted(missing_requested)}"
        assert not missing_loaded, f"Missing successful GLB responses: {sorted(missing_loaded)}"
        assert not bad_status, f"GLB HTTP status errors: {bad_status}"
        assert not glb_failed, f"GLB request failures: {glb_failed}"

        canvas = _canvas_probe(page)
        assert canvas["ok"] and canvas["width"] > 0 and canvas["height"] > 0, f"invalid twin canvas: {canvas}"
        assert canvas["nonBg"] >= 2, f"twin canvas appears blank/background-only: {canvas}"

        page.select_option("#run-select", "miss_unseen")
        page.wait_for_function(
            """
            () => {
              const sw = document.querySelector("#status-word");
              const dataStatus = (sw?.getAttribute("data-status") || "").toLowerCase();
              return dataStatus === "novel" && document.querySelector("#graph")?.getAttribute("data-branch") === "novel";
            }
            """,
            timeout=90_000,
        )
        assert page.locator("#graph .branch-novel").count() > 0, "missing red abnormal branch for novel run"

        page.evaluate("() => window.__twin.setTime(2.5)")
        page.wait_for_function(
            "() => getComputedStyle(document.getElementById('twin-divergence')).display !== 'none'",
            timeout=15_000,
        )
        twin_divergence = (page.text_content("#twin-divergence") or "").strip()
        assert "REALITY DIVERGED" in twin_divergence
        assert "1.904" in twin_divergence

        perf = page.evaluate(
            """async () => {
                window.__twin.setTime(0);
                window.__twin.play();
                const dts = [];
                let last = performance.now();
                const start = last;
                await new Promise((resolve) => {
                    const step = (now) => {
                        dts.push(now - last);
                        last = now;
                        if (now - start >= 2000) {
                            resolve();
                            return;
                        }
                        requestAnimationFrame(step);
                    };
                    requestAnimationFrame(step);
                });
                window.__twin.pause();
                const avg = dts.reduce((a, b) => a + b, 0) / Math.max(dts.length, 1);
                return { avg, count: dts.length };
            }"""
        )
        assert perf["count"] >= 60, f"Too few frame samples collected: {perf}"
        assert perf["avg"] < 33.0, f"Average frame time too high: {perf['avg']:.2f} ms"

        page.select_option("#run-select", "normal_16")
        page.wait_for_function(
            """
            () => {
              const sw = document.querySelector("#status-word");
              const status = (sw?.getAttribute("data-status") || "").toLowerCase();
              return status === "normal" && document.querySelector('#run-select')?.value === 'normal_16';
            }
            """,
            timeout=90_000,
        )
        page.evaluate("() => window.__twin.setTime(2.5)")
        page.wait_for_timeout(200)
        assert page.locator("#twin-divergence").evaluate(
            "el => getComputedStyle(el).display === 'none'"
        ), "normal_16 should not show twin divergence marker"

        page.fill("#search-input", "show me failed grasps")
        page.click("#search-btn")
        page.wait_for_selector(".result-card", timeout=60_000)
        cards = page.locator(".result-card")
        assert cards.count() >= 3, "expected at least 3 search result cards"
        titles = [
            (cards.nth(i).locator(".result-title").inner_text() or "").strip()
            for i in range(min(3, cards.count()))
        ]
        for title in titles:
            _assert_not_raw_title(title)

        page.click("#view-toggle button[data-view='camera']")
        page.wait_for_function(
            """
            () => {
              const v = document.querySelector("#video");
              if (!v) return false;
              const cs = getComputedStyle(v);
              return cs.display !== "none" && cs.visibility !== "hidden" && !v.paused && v.readyState >= 2;
            }
            """,
            timeout=60_000,
        )
        video_back = _video_metrics(page)
        assert video_back["height"] >= 450.0, f"video height regressed after toggle: {video_back}"
        assert not video_back["paused"], f"video should be playing after returning to CAMERA: {video_back}"

        context.close()
        browser.close()

    assert not console_errors, f"Console errors seen: {console_errors}"
    assert not page_errors, f"Page errors seen: {page_errors}"
