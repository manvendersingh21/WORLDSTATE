from __future__ import annotations

import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
RESERVED_PORTS = {8000, 8100, 8101}
IGNORED_CONSOLE_ERRORS = {
    # Chrome reports this when /favicon.ico is absent; unrelated to callouts behavior.
    "Failed to load resource: the server responded with a status of 404 (File not found)",
}


def _pick_free_port() -> int:
    for _ in range(48):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in RESERVED_PORTS:
            return port
    raise RuntimeError("Could not find a free non-reserved port for callouts test server")


def _wait_http_ready(base_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/twin/index.html", timeout=1.5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - best effort polling while server starts
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local twin demo server at {base_url}: {last_error}")


def _set_time(page, seconds: float) -> None:
    page.evaluate(
        """(t) => {
            if (window.__seek) return window.__seek(t);
            window.__twin.pause();
            window.__twin.setTime(t);
            return Promise.resolve(window.__twin.getTime());
        }""",
        seconds,
    )
    page.wait_for_timeout(40)


def _set_run(page, run_id: str) -> None:
    page.evaluate(
        """(rid) => {
            if (window.__setRun) return window.__setRun(rid);
            const select = document.getElementById('twin-run-select');
            if (select) select.value = rid;
            return Promise.resolve();
        }""",
        run_id,
    )
    page.wait_for_function(
        """(rid) => {
            const select = document.getElementById('twin-run-select');
            const root = document.querySelector('.twin-root');
            return select && root && select.value === rid && root.dataset.status !== 'loading';
        }""",
        arg=run_id,
        timeout=40_000,
    )
    page.wait_for_timeout(40)


def _visible_callouts(page) -> dict:
    return page.evaluate(
        """() => {
            const cards = [...document.querySelectorAll('.co-card[data-visible="true"]')];
            const visible = cards.filter((card) => {
                const style = getComputedStyle(card);
                return style.visibility !== 'hidden' && style.display !== 'none' && Number(style.opacity || '1') > 0.01;
            });
            const rects = visible.map((card) => {
                const r = card.getBoundingClientRect();
                return {
                    id: card.getAttribute('data-id') || '',
                    left: r.left,
                    right: r.right,
                    top: r.top,
                    bottom: r.bottom,
                    width: r.width,
                    height: r.height,
                };
            });
            const overlaps = [];
            for (let i = 0; i < rects.length; i += 1) {
                for (let j = i + 1; j < rects.length; j += 1) {
                    const a = rects[i];
                    const b = rects[j];
                    const overlapW = Math.min(a.right, b.right) - Math.max(a.left, b.left);
                    const overlapH = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
                    if (overlapW > 1 && overlapH > 1) overlaps.push([a.id, b.id, overlapW, overlapH]);
                }
            }
            return {
                count: visible.length,
                texts: visible.map((card) => card.innerText.trim()),
                overlaps,
            };
        }"""
    )


def _assert_layout_sane(page) -> dict:
    state = _visible_callouts(page)
    assert state["count"] <= 3, f"More than 3 callouts visible: {state['count']} :: {state['texts']}"
    assert not state["overlaps"], f"Visible callouts overlap: {state['overlaps']} :: {state['texts']}"
    return state


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


def test_twin_in_scene_callouts(twin_demo_server):
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--use-angle=metal"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def on_console(msg):
            if msg.type == "error" and msg.text not in IGNORED_CONSOLE_ERRORS:
                console_errors.append(msg.text)

        def on_page_error(exc):
            page_errors.append(str(exc))

        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        page.goto(f"{twin_demo_server}/twin/index.html", wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector("#twin-canvas", timeout=60_000)
        page.wait_for_function(
            "() => window.__twin && document.querySelector('.twin-root')?.dataset.status !== 'loading'",
            timeout=60_000,
        )
        page.wait_for_function(
            "() => !!document.querySelector('[data-callouts]') && !!window.__callouts",
            timeout=60_000,
        )

        _set_run(page, "miss_unseen")
        _set_time(page, 1.0)
        s10 = _assert_layout_sane(page)
        blob10 = " | ".join(s10["texts"]).lower()
        assert "displaced" not in blob10, f"Failure callout appeared before divergence: {s10['texts']}"
        assert "empty air" not in blob10, f"Failure callout appeared before divergence: {s10['texts']}"

        _set_time(page, 2.2)
        s22 = _assert_layout_sane(page)
        blob22 = " | ".join(s22["texts"])
        displaced_match = re.search(r"displaced\s+([0-9]+(?:\.[0-9]+)?)\s*cm", blob22, flags=re.IGNORECASE)
        assert displaced_match, f"Missing cube displacement callout at t=2.2: {s22['texts']}"
        displaced_cm = float(displaced_match.group(1))
        assert 4.0 <= displaced_cm <= 6.0, f"Displacement out of expected range (4.0-6.0 cm): {displaced_cm}"

        # Summary card can begin a few frames later than divergence due to its own scheduled t.
        _set_time(page, 2.35)
        s235 = _assert_layout_sane(page)
        blob235 = " | ".join(s235["texts"])
        assert re.search(r"0\s*/\s*16", blob235), f"Missing 0/16 support summary in callouts: {s235['texts']}"
        assert "Slides sideways" in blob235, f"Missing observed class summary text after divergence: {s235['texts']}"

        _set_time(page, 3.5)
        s35_fail = _assert_layout_sane(page)
        blob35_fail = " | ".join(s35_fail["texts"]).lower()
        assert "empty air" in blob35_fail, f"Missing 'empty air' failure callout at t=3.5: {s35_fail['texts']}"

        _set_run(page, "normal_16")
        _set_time(page, 5.0)
        s50_norm = _assert_layout_sane(page)
        blob50_norm = " | ".join(s50_norm["texts"])
        assert "Lift" in blob50_norm, f"Missing Lift callout at normal_16 t=5.0: {s50_norm['texts']}"
        assert re.search(r"\bcm\b", blob50_norm), f"Lift callout missing centimeter detail: {s50_norm['texts']}"

        _set_time(page, 8.5)
        s85 = _assert_layout_sane(page)
        blob85 = " | ".join(s85["texts"])
        assert "Cycle complete" in blob85, f"Missing cycle completion summary at t=8.5: {s85['texts']}"

        # Sweep a few times in both runs to enforce max-visible + no-overlap constraints globally.
        for rid in ("miss_unseen", "normal_16"):
            _set_run(page, rid)
            for t in (0.0, 0.7, 1.4, 2.2, 3.5, 5.0, 6.8, 8.5):
                _set_time(page, t)
                _assert_layout_sane(page)

        perf = page.evaluate(
            """async () => {
                window.__twin.pause();
                window.__twin.setTime(0.0);
                window.__twin.play();
                const dts = [];
                let last = performance.now();
                const start = last;
                await new Promise((resolve) => {
                    const step = (now) => {
                        dts.push(now - last);
                        last = now;
                        if (now - start >= 2200) {
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
        assert perf["count"] >= 60, f"Too few frame samples collected for perf check: {perf}"
        assert perf["avg"] < 30.0, f"Average frame time too high with callouts: {perf['avg']:.2f} ms"

        context.close()
        browser.close()

    assert not console_errors, f"Console errors seen: {console_errors}"
    assert not page_errors, f"Page errors seen: {page_errors}"
