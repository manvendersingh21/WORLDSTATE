import json
import re
import time
from pathlib import Path
from urllib import request

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:8100"
SHOTS_DIR = Path("/tmp/judge-shots")


def _api_json(method: str, path: str, payload: dict | None = None, timeout: float = 30.0):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _wait_for_health_version(version: int, timeout_s: float = 360.0) -> dict:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            health = _api_json("GET", "/api/health", timeout=10.0)
            last = health
            if int(health.get("version", -1)) == version:
                return health
        except Exception as exc:  # pragma: no cover - transient network/readiness errors
            last = {"error": str(exc)}
        time.sleep(2.0)
    raise AssertionError(f"/api/health did not reach version {version} within {timeout_s:.0f}s; last={last}")


def _wait_for_api_ready(timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            _api_json("GET", "/api/health", timeout=10.0)
            return
        except Exception as exc:  # pragma: no cover - transient startup/down periods
            last = str(exc)
            time.sleep(2.0)
    raise AssertionError(f"API at {BASE_URL} did not become reachable within {timeout_s:.0f}s; last={last}")


def _reset_to_v1() -> dict:
    _wait_for_api_ready(timeout_s=120.0)
    # /api/learn can take minutes; allow enough time for the request itself to finish.
    deadline = time.time() + 120.0
    last = None
    while True:
        try:
            _api_json("POST", "/api/learn", timeout=420.0)
            break
        except Exception as exc:
            last = str(exc)
            if time.time() >= deadline:
                raise AssertionError(f"POST /api/learn unreachable for 120s; last={last}") from exc
            time.sleep(2.0)
    return _wait_for_health_version(1, timeout_s=360.0)


def _shot(page, name: str) -> None:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS_DIR / name), full_page=False)


def _wait_for_status(page, status: str, timeout_ms: float = 60_000) -> None:
    page.wait_for_function(
        """
        ({ expected }) => {
          const ids = ["#status-word", "#status-badge"];
          const expectedText = expected.replaceAll("_", " ").toUpperCase();
          for (const sel of ids) {
            const el = document.querySelector(sel);
            if (!el) continue;
            const dataStatus = (el.getAttribute("data-status") || "").toLowerCase();
            const text = (el.textContent || "").toUpperCase();
            if (dataStatus === expected || text.includes(expectedText)) return true;
          }
          return false;
        }
        """,
        arg={"expected": status},
        timeout=timeout_ms,
    )


def _set_run(page, run_id: str, expected_status: str, timeout_ms: float = 60_000) -> None:
    page.select_option("#run-select", value=run_id)
    page.wait_for_function(
        "(id) => document.querySelector('#run-select') && document.querySelector('#run-select').value === id",
        arg=run_id,
        timeout=15_000,
    )
    _wait_for_status(page, expected_status, timeout_ms=timeout_ms)


def _assert_video_ready_and_playable(page) -> None:
    page.wait_for_function(
        "() => document.querySelector('#video') && document.querySelector('#video').readyState >= 2",
        timeout=60_000,
    )
    # Trigger playback to ensure media is decodable in chrome channel.
    playable = page.evaluate(
        """
        () => {
          const v = document.querySelector("#video");
          if (!v) return false;
          v.muted = true;
          const p = v.play();
          if (p && typeof p.then === "function") {
            return p.then(() => v.readyState >= 2).catch(() => v.readyState >= 2);
          }
          return v.readyState >= 2;
        }
        """
    )
    assert playable, "video failed to reach a playable state"


def _assert_human_text(title: str) -> None:
    assert title and title.strip(), "result title should not be empty"
    canonical = title.strip().lower()
    # Raw run IDs look like miss_eval_right / normal_16 / upload_123...
    assert not re.fullmatch(r"(normal|miss|upload|reference|heldout|eval)[a-z0-9_]*", canonical), (
        f"expected human-readable result text, got raw-looking id: {title}"
    )


def test_judge_mode_e2e():
    _reset_to_v1()
    console_errors: list[str] = []
    page_errors: list[str] = []
    final_reset_error = None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(BASE_URL, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_selector("#run-select", timeout=120_000)
            page.wait_for_function(
                """
                () => {
                  const sel = document.querySelector("#run-select");
                  const sw = document.querySelector("#status-word");
                  const sb = document.querySelector("#status-badge");
                  const value = sel && sel.value;
                  const dataStatus = ((sw && sw.getAttribute("data-status")) ||
                                      (sb && sb.getAttribute("data-status")) ||
                                      "").toLowerCase();
                  return !!value && ["normal", "novel", "known_failure"].includes(dataStatus);
                }
                """,
                timeout=120_000,
            )
            _shot(page, "01-load.png")

            _set_run(page, "normal_16", "normal", timeout_ms=60_000)
            _assert_video_ready_and_playable(page)
            page.wait_for_function(
                """
                () => {
                  const t = document.querySelector("#cosmos-text");
                  const a = document.querySelector("#cosmos-attribution");
                  return !!(t && t.textContent && t.textContent.trim().length > 20 &&
                            a && a.textContent && a.textContent.trim().length > 0);
                }
                """,
                timeout=90_000,
            )
            attribution = (page.text_content("#cosmos-attribution") or "").strip()
            assert attribution in {"NVIDIA Cosmos Reason", "Kinematic summary"}
            _shot(page, "02-normal-16.png")

            _set_run(page, "miss_unseen", "novel", timeout_ms=60_000)
            divergence = (page.text_content("#divergence-time") or "").strip()
            assert "1.9" in divergence, f"expected divergence around 1.9s, got {divergence!r}"
            expected_field = (page.text_content("#field-expected") or "").strip()
            observed_field = (page.text_content("#field-observed") or "").strip()
            support = (page.text_content("#field-support") or "").strip()
            assert expected_field, "expected field should be non-empty"
            assert observed_field, "observed field should be non-empty"
            assert re.search(r"0\s*/\s*16", support), f"support should be 0 / 16, got {support!r}"
            page.wait_for_function(
                "() => document.querySelector('#graph')?.getAttribute('data-branch') === 'novel'",
                timeout=30_000,
            )
            assert page.locator("#graph .branch-novel").count() > 0, "missing red abnormal branch"
            _shot(page, "03-miss-unseen-novel.png")

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
                _assert_human_text(title)
            failure_like = sum(
                any(token in title.lower() for token in ("fail", "failure", "diverg", "displaced", "unexpected"))
                for title in titles
            )
            assert failure_like >= 2, f"top search results should mostly be failures; got {titles}"
            _shot(page, "04-search-failed-grasps.png")

            remember_btn = page.locator("#remember-btn")
            assert remember_btn.is_visible(), "remember button should be visible for novel failure"
            assert remember_btn.is_enabled(), "remember button should be enabled for novel failure"
            remember_btn.click()
            _wait_for_status(page, "known_failure", timeout_ms=45_000)
            confirm = (page.text_content("#remember-confirm") or "").strip().lower()
            assert "failure pattern added to world memory" in confirm
            page.wait_for_function(
                "() => document.querySelector('#graph')?.getAttribute('data-branch') === 'known'",
                timeout=30_000,
            )
            assert page.locator("#graph .branch-known").count() > 0, "missing blue remembered branch"
            _shot(page, "05-known-after-remember.png")

            with page.expect_response(
                lambda res: "/api/upload" in res.url and res.request.method == "POST" and res.ok,
                timeout=90_000,
            ) as upload_response:
                page.click("#upload-similar-btn")
            upload_payload = upload_response.value.json()
            uploaded_id = upload_payload["episode"]["id"]
            _wait_for_status(page, "known_failure", timeout_ms=90_000)
            page.wait_for_function(
                "(runId) => document.querySelector('#run-select')?.value === runId",
                arg=uploaded_id,
                timeout=30_000,
            )
            _shot(page, "06-upload-similar-known.png")

            _set_run(page, "normal_17", "normal", timeout_ms=90_000)
            _shot(page, "07-normal-17.png")

            context.close()
            browser.close()
    finally:
        try:
            _reset_to_v1()
        except Exception as exc:  # pragma: no cover - only hit on backend outage/reset failure
            final_reset_error = exc

    assert not console_errors, f"console errors detected: {console_errors}"
    assert not page_errors, f"page errors detected: {page_errors}"
    if final_reset_error:
        raise AssertionError(f"final /api/learn reset to v1 failed: {final_reset_error}") from final_reset_error
