"""The event bus and the local panel.

The web tests are security tests as much as functional ones. Every control the
panel relies on -- the token, the Host check, the Origin check, the custom-header
requirement on POSTs, the absence of CORS -- is asserted here, because each of them
is one line away from being accidentally removed and none of them would fail
visibly if it were.
"""

from __future__ import annotations

import http.client
import json
import threading
import time

import httpx
import pytest

from meetscribe.bus.pubsub import InMemoryEventBus
from meetscribe.core.events import EV_LEVEL, EV_NOTICE, EV_SEGMENT
from meetscribe.web.server import MAX_BODY_BYTES, TOKEN_HEADER, PanelServer

# --------------------------------------------------------------------- the bus


def test_sequence_numbers_are_monotonic_and_gap_free(bus):
    events = [bus.publish(EV_SEGMENT, {"i": i}) for i in range(5)]
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert bus.last_seq == 5


def test_volatile_events_are_delivered_live_but_never_replayed(bus):
    """Replaying two hours of meter ticks after a refresh would bury the
    transcript and could evict it from the buffer entirely."""
    subscription = bus.subscribe()
    bus.publish(EV_SEGMENT, {"text": "hola"})
    bus.publish(EV_LEVEL, {"tracks": {}})
    assert subscription.get(timeout=1).type == EV_SEGMENT
    assert subscription.get(timeout=1).type == EV_LEVEL

    assert [e.type for e in bus.history()] == [EV_SEGMENT]
    subscription.close()


def test_a_reconnecting_subscriber_receives_exactly_what_it_missed(bus):
    for i in range(6):
        bus.publish(EV_SEGMENT, {"i": i})
    resumed = bus.subscribe(after_seq=3)
    received = [resumed.get(timeout=1).data["i"] for _ in range(3)]
    assert received == [3, 4, 5]
    resumed.close()


def test_replay_and_live_delivery_do_not_race(bus):
    """The seam between the history snapshot and going live is where a naive
    implementation drops exactly one line per reconnect."""
    bus.publish(EV_SEGMENT, {"i": 0})
    subscription = bus.subscribe(after_seq=0)
    bus.publish(EV_SEGMENT, {"i": 1})
    assert [subscription.get(timeout=1).data["i"] for _ in range(2)] == [0, 1]
    subscription.close()


def test_a_stuck_subscriber_is_disconnected_not_waited_for(bus):
    """A paused browser tab must never apply backpressure to the transcriber."""
    slow = InMemoryEventBus(history_limit=100, subscriber_queue=3)
    subscription = slow.subscribe()
    for i in range(20):
        slow.publish(EV_SEGMENT, {"i": i})
    assert subscription.closed
    assert subscription.dropped
    assert slow.disconnected == 1
    assert slow.subscriber_count == 0
    # ...and the events themselves survive for the reconnect.
    assert len(slow.history()) == 20


def test_iterating_a_subscription_ends_when_it_closes(bus):
    subscription = bus.subscribe()
    bus.publish(EV_NOTICE, {"level": "info", "code": "x", "message": "y"})

    seen = []

    def consume():
        seen.extend(event.type for event in subscription)

    thread = threading.Thread(target=consume)
    thread.start()
    time.sleep(0.05)
    subscription.close()
    thread.join(timeout=2)
    assert seen == [EV_NOTICE]


def test_history_is_bounded(bus):
    small = InMemoryEventBus(history_limit=5)
    for i in range(20):
        small.publish(EV_SEGMENT, {"i": i})
    history = small.history()
    assert len(history) == 5
    assert history[0].data["i"] == 15


def test_time_source_stamps_events(bus):
    bus.set_time_source(lambda: 42.5)
    assert bus.publish(EV_SEGMENT, {}).at == 42.5


def test_a_broken_time_source_cannot_break_publishing(bus):
    def explode():
        raise RuntimeError("clock died")

    bus.set_time_source(explode)
    assert bus.publish(EV_SEGMENT, {}).at == 0.0


def test_closing_the_bus_releases_every_subscriber(bus):
    a, b = bus.subscribe(), bus.subscribe()
    bus.close()
    assert a.closed and b.closed
    assert bus.subscriber_count == 0


# ------------------------------------------------------------------- the panel


@pytest.fixture
def panel(bus):
    renames: list[tuple[str, str]] = []
    stops: list[bool] = []
    server = PanelServer(
        bus,
        port=0,
        heartbeat_s=0.3,
        state_provider=lambda: {"session": {"name": "prueba"}, "cursor": bus.last_seq},
        on_rename=lambda sid, name: (renames.append((sid, name)), 3)[1],
        on_stop=lambda: stops.append(True),
    )
    server.start()
    server.renames = renames  # type: ignore[attr-defined]
    server.stops = stops  # type: ignore[attr-defined]
    yield server
    server.stop()


def get(panel: PanelServer, path: str, **kw) -> httpx.Response:
    return httpx.get(f"{panel.origin}{path}", timeout=5, **kw)


def test_the_page_and_its_assets_are_served_without_a_token(panel):
    """They contain no meeting content, so gating them would add friction and no
    secrecy."""
    page = get(panel, "/")
    assert page.status_code == 200
    assert "MeetScribe" in page.text
    assert get(panel, "/static/app.js").status_code == 200
    assert get(panel, "/static/styles.css").status_code == 200


def test_data_requires_the_token(panel):
    assert get(panel, "/api/state").status_code == 401
    assert get(panel, "/api/state?t=incorrecto").status_code == 401
    good = get(panel, f"/api/state?t={panel.token}")
    assert good.status_code == 200
    assert good.json()["session"]["name"] == "prueba"


def test_a_forged_host_header_is_refused(panel):
    """Defeats DNS rebinding: a page that resolves its own name to 127.0.0.1."""
    response = get(
        panel, f"/api/state?t={panel.token}", headers={"Host": "evil.example.com"}
    )
    assert response.status_code == 400


def test_a_foreign_origin_is_refused(panel):
    response = get(
        panel,
        f"/api/state?t={panel.token}",
        headers={"Origin": "http://evil.example.com"},
    )
    assert response.status_code == 403


def test_the_panels_own_origin_is_accepted(panel):
    response = get(
        panel, f"/api/state?t={panel.token}", headers={"Origin": panel.origin}
    )
    assert response.status_code == 200


def test_a_post_needs_the_token_in_a_custom_header(panel):
    """A cross-origin page cannot set one without a CORS preflight this server
    never answers, so form-style CSRF is structurally impossible."""
    without = httpx.post(
        f"{panel.origin}/api/speaker?t={panel.token}",
        json={"speaker_id": "p1", "name": "Ana"},
        timeout=5,
    )
    assert without.status_code == 401

    with_header = httpx.post(
        f"{panel.origin}/api/speaker?t={panel.token}",
        json={"speaker_id": "p1", "name": "Ana"},
        headers={TOKEN_HEADER: panel.token},
        timeout=5,
    )
    assert with_header.status_code == 200
    assert with_header.json() == {"ok": True, "segments": 3}
    assert panel.renames == [("p1", "Ana")]


def test_a_rejected_post_does_not_corrupt_the_next_request(panel):
    """Regression: an unread body used to be parsed as the next request line.

    Every check that can reject a POST runs after the body has been read, because
    on a keep-alive connection the bytes of an undrained body stay in the socket
    and the server reads them as the start of whatever comes next. The symptom was
    a legitimate request answered with ``501 Unsupported method
    ('{"speaker_id":...}POST')`` -- so one denied request broke the following one.

    ``http.client`` is used rather than ``httpx`` because this only reproduces when
    both requests travel over the *same* connection.
    """
    connection = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=5)
    try:
        body = json.dumps({"speaker_id": "p1", "name": "Ana"})
        # Rejected at the token check, with a body the server never asked for.
        connection.request(
            "POST", f"/api/speaker?t={panel.token}", body=body,
            headers={"Content-Type": "application/json"},
        )
        denied = connection.getresponse()
        assert denied.status == 401
        denied.read()

        # The very next request on this same connection must be understood.
        connection.request("GET", f"/api/state?t={panel.token}")
        following = connection.getresponse()
        payload = following.read()
        assert following.status == 200, payload
        assert json.loads(payload)["session"]["name"] == "prueba"
    finally:
        connection.close()


def test_an_oversized_body_is_refused_without_being_read(panel):
    connection = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=5)
    try:
        connection.putrequest("POST", f"/api/speaker?t={panel.token}")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413
    finally:
        connection.close()


def test_a_malformed_content_length_is_refused(panel):
    connection = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=5)
    try:
        connection.putrequest("POST", f"/api/speaker?t={panel.token}")
        connection.putheader("Content-Length", "no-soy-un-numero")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
    finally:
        connection.close()


def test_rename_validates_its_input(panel):
    for payload in ({}, {"speaker_id": "p1"}, {"speaker_id": "p1", "name": "x" * 200}):
        response = httpx.post(
            f"{panel.origin}/api/speaker",
            json=payload,
            headers={TOKEN_HEADER: panel.token},
            timeout=5,
        )
        assert response.status_code == 400


def test_stop_is_accepted_and_runs_off_the_request_thread(panel):
    response = httpx.post(
        f"{panel.origin}/api/stop",
        json={},
        headers={TOKEN_HEADER: panel.token},
        timeout=5,
    )
    assert response.status_code == 202
    time.sleep(0.2)
    assert panel.stops == [True]


def test_no_cors_headers_are_ever_emitted(panel):
    response = get(panel, f"/api/state?t={panel.token}")
    assert not any(h.lower().startswith("access-control") for h in response.headers)


def test_security_headers_are_present(panel):
    response = get(panel, "/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp


def test_path_traversal_is_a_plain_404(panel):
    response = get(panel, "/static/../../../../Windows/win.ini")
    assert response.status_code in (400, 404)
    assert "win.ini" not in response.text


def test_unknown_routes_404(panel):
    assert get(panel, f"/api/nope?t={panel.token}").status_code == 404


def test_the_url_carries_the_token(panel):
    assert panel.url.startswith(f"http://127.0.0.1:{panel.port}/?t=")
    assert panel.token in panel.url


def test_sse_streams_events_and_resumes_from_a_cursor(panel, bus):
    bus.publish(EV_SEGMENT, {"text": "primero"})
    with httpx.stream(
        "GET", f"{panel.origin}/events?t={panel.token}&cursor=0", timeout=5
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frame = _read_frame(response)
    assert frame["event"] == EV_SEGMENT
    assert json.loads(frame["data"])["data"]["text"] == "primero"
    first_id = int(frame["id"])

    bus.publish(EV_SEGMENT, {"text": "segundo"})
    with httpx.stream(
        "GET",
        f"{panel.origin}/events?t={panel.token}",
        headers={"Last-Event-ID": str(first_id)},
        timeout=5,
    ) as response:
        frame = _read_frame(response)
    assert json.loads(frame["data"])["data"]["text"] == "segundo"


def test_sse_requires_the_token(panel):
    with httpx.stream("GET", f"{panel.origin}/events", timeout=5) as response:
        assert response.status_code == 401


def test_sse_sends_a_keepalive_while_idle(panel):
    with httpx.stream("GET", f"{panel.origin}/events?t={panel.token}", timeout=5) as r:
        for line in r.iter_lines():
            if line.startswith(":"):
                assert "keepalive" in line
                return
    pytest.fail("no keepalive frame arrived")


def _read_frame(response: httpx.Response) -> dict[str, str]:
    frame: dict[str, str] = {}
    for line in response.iter_lines():
        if line == "":
            if frame:
                return frame
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(": ")
        frame[key] = value
    raise AssertionError("stream ended before a complete frame")


def test_starting_the_panel_twice_is_an_error(panel):
    from meetscribe.errors import PanelError

    with pytest.raises(PanelError):
        panel.start()


def test_valid_origin_accepts_localhost_alias(panel):
    assert panel.valid_origin(f"http://localhost:{panel.port}")
    assert not panel.valid_origin(f"https://127.0.0.1:{panel.port}")
    assert not panel.valid_origin("http://127.0.0.1:1")
    assert panel.valid_origin(None)
    assert not panel.valid_origin("null")
