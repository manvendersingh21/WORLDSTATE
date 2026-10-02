from __future__ import annotations

import re
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:8101"
DEMO_URL = f"{BASE_URL}/static/twin/evidence-demo.html"


def test_live_twin_evidence_module() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    api_requests: list[tuple[str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def record_request(request) -> None:
            path = urlparse(request.url).path
            if path.startswith("/api/"):
                api_requests.append((request.method, path))

        page.on("request", record_request)

        response = page.goto(DEMO_URL, wait_until="domcontentloaded", timeout=60_000)
        assert response is not None and response.ok, f"demo failed to load: {response}"

        result = page.evaluate(
            """async () => {
                const module = await import(`/static/twin/evidence.js?test=${Date.now()}`);
                const evidence = await module.loadEvidence("miss_unseen");
                const atOneSecond = evidence.atTime(1.0);
                return {
                    atOneSecond,
                    cosmos: evidence.cosmos,
                    tags: [
                        module.formatTag(atOneSecond.part, "cube"),
                        module.formatTag(atOneSecond.gripper, "gripper"),
                        module.formatTag(atOneSecond.fixture, "target"),
                    ],
                };
            }"""
        )

        context.close()
        browser.close()

    detections = result["atOneSecond"]
    for label in ("gripper", "part", "fixture"):
        detection = detections[label]
        assert detection is not None, f"missing {label} detection at t=1.0"
        assert 0 < detection["conf"] <= 1, (
            f"{label} confidence must be in (0, 1], got {detection['conf']}"
        )

    cosmos = result["cosmos"]
    assert cosmos["source"] == "cosmos-reason"
    assert cosmos["failureSentence"].strip()
    assert re.search(
        r"fail|shift|miss|empty|not",
        cosmos["failureSentence"],
        flags=re.IGNORECASE,
    )

    tag_pattern = re.compile(r"YOLO · (cube|gripper|target) 0\.\d\d")
    assert all(tag_pattern.fullmatch(tag) for tag in result["tags"]), result["tags"]

    requested_paths = {path for _, path in api_requests}
    assert "/api/episodes/miss_unseen" in requested_paths
    assert "/api/episodes/miss_unseen/narration" in requested_paths
    assert all(method == "GET" for method, _ in api_requests), api_requests

    assert not console_errors, f"console errors detected: {console_errors}"
    assert not page_errors, f"page errors detected: {page_errors}"
