from __future__ import annotations

import hashlib
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
RESERVED_PORTS = {8000, 8100, 8210}
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
    # Chrome emits this if /favicon.ico is absent; unrelated to twin module behavior.
    "Failed to load resource: the server responded with a status of 404 (File not found)",
}


def _pick_free_port() -> int:
    for _ in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in RESERVED_PORTS:
            return port
    raise RuntimeError("Could not find a free non-reserved port for twin UI test server")


def _wait_http_ready(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/twin/index.html", timeout=1.5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - best effort while polling
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local twin demo server at {base_url}: {last_error}")


def _canvas_hash(page) -> str:
    png = page.locator("#twin-canvas").screenshot()
    return hashlib.sha256(png).hexdigest()


def _stable_canvas_hash(page, max_samples: int = 8, settle_ms: int = 120) -> str:
    last = ""
    for _ in range(max_samples):
        page.wait_for_timeout(settle_ms)
        current = _canvas_hash(page)
        if current == last:
            return current
        last = current
    return last


@pytest.fixture(scope="module")
def twin_demo_server():
    port = _pick_free_port()
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(WEB_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_http_ready(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - best effort cleanup
            proc.kill()


def test_twin_index_demo_behaviors(twin_demo_server):
    console_errors: list[str] = []
    page_errors: list[str] = []
    glb_requests: set[str] = set()
    glb_ok_status: dict[str, int] = {}
    glb_failed: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--use-angle=metal"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def on_console(msg):
            if msg.type == "error" and msg.text not in IGNORED_CONSOLE_ERRORS:
                console_errors.append(msg.text)

        def on_page_error(exc):
            page_errors.append(str(exc))

        def on_request(request):
            path = urlparse(request.url).path
            if path.endswith(".glb"):
                glb_requests.add(Path(path).name)

        def on_response(response):
            path = urlparse(response.url).path
            if path.endswith(".glb"):
                glb_ok_status[Path(path).name] = response.status

        def on_request_failed(request):
            path = urlparse(request.url).path
            if path.endswith(".glb"):
                glb_failed.append(f"{Path(path).name}: {request.failure}")

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)

        page.goto(f"{twin_demo_server}/twin/index.html", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector("#twin-canvas", timeout=60_000)
        page.wait_for_function(
            "() => window.__twin && document.querySelector('.twin-root')?.dataset.status !== 'loading'",
            timeout=60_000,
        )

        deadline = time.time() + 20
        while len(glb_ok_status) < len(EXPECTED_GLB_NAMES) and time.time() < deadline:
            page.wait_for_timeout(150)

        missing_requested = EXPECTED_GLB_NAMES - glb_requests
        missing_loaded = EXPECTED_GLB_NAMES - set(glb_ok_status.keys())
        bad_status = {name: status for name, status in glb_ok_status.items() if status >= 400}
        assert not missing_requested, f"Missing GLB requests: {sorted(missing_requested)}"
        assert not missing_loaded, f"Missing successful GLB responses: {sorted(missing_loaded)}"
        assert not bad_status, f"GLB responses with HTTP errors: {bad_status}"
        assert not glb_failed, f"Failed GLB requests: {glb_failed}"

        # Prove a real render happened: sampled pixels should not all equal the dark background.
        pixel_probe = page.evaluate(
            """() => {
                const canvas = document.getElementById('twin-canvas');
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
                return { nonBg, width: probe.width, height: probe.height };
            }"""
        )
        assert pixel_probe["width"] > 0 and pixel_probe["height"] > 0
        assert pixel_probe["nonBg"] >= 2, f"Canvas appears blank/background-only: {pixel_probe}"

        page.select_option("#twin-run-select", "miss_unseen")
        page.wait_for_function(
            "() => document.getElementById('twin-run-select')?.value === 'miss_unseen' "
            "&& document.querySelector('.twin-root')?.dataset.status !== 'loading'",
            timeout=30_000,
        )

        page.evaluate("() => window.__twin.setTime(1.0)")
        assert page.locator("#twin-divergence").evaluate("el => getComputedStyle(el).display") == "none"

        page.evaluate("() => window.__twin.setTime(2.5)")
        page.wait_for_function(
            "() => getComputedStyle(document.getElementById('twin-divergence')).display !== 'none'",
            timeout=10_000,
        )
        divergence_text = page.locator("#twin-divergence").inner_text()
        assert "REALITY DIVERGED" in divergence_text
        assert "1.904" in divergence_text

        page.select_option("#twin-run-select", "normal_16")
        page.wait_for_function(
            "() => document.getElementById('twin-run-select')?.value === 'normal_16' "
            "&& document.querySelector('.twin-root')?.dataset.status === 'normal'",
            timeout=30_000,
        )
        state_05 = page.evaluate(
            "() => { window.__twin.setTime(0.5); return document.getElementById('twin-state-label').textContent.trim(); }"
        )
        state_50 = page.evaluate(
            "() => { window.__twin.setTime(5.0); return document.getElementById('twin-state-label').textContent.trim(); }"
        )
        assert state_05, "State label at t=0.5 is empty"
        assert state_50, "State label at t=5.0 is empty"
        assert state_05 != state_50, f"State label did not change: {state_05!r} vs {state_50!r}"

        page.evaluate("() => { window.__twin.pause(); window.__twin.setTime(0.0); }")
        page.click("button[data-preset='perspective']")
        perspective_hash = _stable_canvas_hash(page)
        preset_hashes = {"perspective": perspective_hash}
        for preset in ("front", "side", "top"):
            page.click(f"button[data-preset='{preset}']")
            preset_hashes[preset] = _stable_canvas_hash(page)
            assert preset_hashes[preset] != perspective_hash, f"Preset {preset} did not change camera view"
        assert len(set(preset_hashes.values())) == 4, f"Preset views are not distinct: {preset_hashes}"

        page.click("button[data-preset='side']")
        side_hash = _stable_canvas_hash(page)
        page.click("#twin-reset")
        reset_hash = _stable_canvas_hash(page)
        assert reset_hash != side_hash, "Reset View did not move camera away from side preset"
        assert page.locator("button[data-preset='perspective']").evaluate(
            "el => el.classList.contains('twin-btn-active')"
        ), "Reset View did not return to perspective preset state"

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

        context.close()
        browser.close()

    assert not console_errors, f"Console errors seen: {console_errors}"
    assert not page_errors, f"Page errors seen: {page_errors}"
