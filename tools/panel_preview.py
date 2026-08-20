"""Serve the live panel against a scripted meeting, so the UI can be worked on.

    python tools/panel_preview.py

Prints a URL and holds it open until Ctrl-C. Without this, the only way to look at
the panel is to record a real meeting, which makes the two states that matter most
-- a dead microphone, and a finished session with its minutes -- the two hardest
ones to reach.

Everything below the scripted events is production code: the real
:class:`PanelServer`, the real :class:`InMemoryEventBus`, the real
``core.events`` makers. Only the pipeline is replaced. So the page under
inspection is genuinely the page a meeting produces, and a rendering bug found here
is a real one rather than an artifact of a mock.

The microphone is deliberately fed digital silence, because "your mic is producing
no signal" is the single most important thing this panel ever says and it should
never be the least-exercised path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from meetscribe.bus.pubsub import InMemoryEventBus
from meetscribe.core import events as ev
from meetscribe.core.models import (
    ActionItem,
    ConsentMode,
    LevelReading,
    SegmentState,
    Session,
    SessionState,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
)
from meetscribe.web.server import PanelServer

TOKEN = "preview-token-not-a-real-session-token-0001"  # noqa: S105 - fixed for reloads

TURNS = [
    ("p1", "Helena", TrackKind.LOOPBACK, 0.1, 4.0,
     "Buenos días, vamos a revisar el avance del proyecto de facturación."),
    ("p2", "Zira", TrackKind.LOOPBACK, 5.6, 9.4,
     "Por mi parte el módulo de validación ya quedó terminado ayer."),
    ("local", "Yo", TrackKind.MIC, 10.9, 13.6,
     "¿Y cómo vamos con la integración al sistema del banco?"),
    ("p2", "Zira", TrackKind.LOOPBACK, 15.1, 18.4,
     "Ahí tenemos un bloqueo con las credenciales de pruebas."),
    ("local", "Yo", TrackKind.MIC, 20.2, 24.6,
     "Entendido, yo lo escalo con el contacto comercial esta semana."),
    ("p2", "Zira", TrackKind.LOOPBACK, 25.7, 30.1,
     "Perfecto, mientras avanzo con el simulador y las pruebas unitarias."),
    ("p1", "Helena", TrackKind.LOOPBACK, 31.4, 35.0,
     "El presupuesto aprobado es de cuarenta y cinco mil pesos."),
    ("p2", "Zira", TrackKind.LOOPBACK, 36.2, 40.4,
     "De acuerdo, lo entrego para el próximo martes sin falta."),
]

SPEAKERS = [
    Speaker("local", "Yo", TrackKind.MIC, is_local=True, utterance_count=2),
    Speaker("p1", "Helena", TrackKind.LOOPBACK, utterance_count=2),
    Speaker("p2", "Zira", TrackKind.LOOPBACK, utterance_count=4),
]

ACTIONS = [
    ActionItem(
        text="Escalar el bloqueo de credenciales con el contacto comercial",
        owner="Yo", due="esta semana", at_s=24.6,
    ),
    ActionItem(
        text="Entregar el simulador y las pruebas unitarias",
        owner="Zira", due="el próximo martes", at_s=40.4,
    ),
]


def build_session() -> Session:
    return Session(
        session_id="2652d12b01eb",
        name="Revisión de facturación",
        started_at=dt.datetime.now(dt.UTC),
        state=SessionState.RUNNING,
        consent=ConsentMode.ALL_PARTIES,
        stt_model="large-v3-turbo",
        language="es",
    )


def build_summary(session_id: str) -> Summary:
    return Summary(
        session_id=session_id,
        executive=(
            "El equipo revisó el avance del proyecto de facturación. El módulo de "
            "validación quedó terminado, mientras que la integración con el sistema "
            "del banco sigue bloqueada por las credenciales del entorno de pruebas. "
            "Se confirmó el presupuesto aprobado y se acordaron los siguientes "
            "pasos con fechas concretas."
        ),
        topics=(
            "Avance del módulo de validación",
            "Integración con el sistema del banco",
            "Presupuesto aprobado",
        ),
        decisions=(
            "El presupuesto aprobado es de 45.000 pesos",
            "La entrega del simulador queda fijada para el martes",
        ),
        action_items=tuple(ACTIONS),
        open_questions=("¿Quién provee las credenciales del entorno de pruebas?",),
        follow_ups=("Confirmar la fecha de la siguiente revisión",),
        provider="anthropic",
        model="claude-opus-5",
        generated_at=dt.datetime.now(dt.UTC),
    )


def publish_meeting(bus: InMemoryEventBus, session: Session) -> None:
    """One meeting's worth of events, then level meters for as long as it runs."""
    time.sleep(0.4)
    bus.publish(ev.EV_SESSION, ev.make_session(session, 0.0))
    for speaker in SPEAKERS:
        bus.publish(ev.EV_SPEAKER, ev.make_speaker(speaker))

    for index, (sid, name, track, start, end, text) in enumerate(TURNS, start=1):
        bus.publish(ev.EV_SEGMENT, ev.make_segment(TranscriptSegment(
            segment_id=f"seg{index:04d}",
            session_id=session.session_id,
            track=track,
            seq=index,
            start_s=start,
            end_s=end,
            speaker_id=sid,
            speaker_name=name,
            text=text,
            state=SegmentState.FINAL,
            language="es",
        )))

    for item in ACTIONS:
        bus.publish(ev.EV_ACTION_ITEM, ev.make_action_item(item))

    bus.publish(ev.EV_NOTICE, ev.make_notice(
        "warn", "capture.silent.mic",
        "La pista del micrófono lleva 12s en silencio digital (sin señal alguna). "
        "Revisa Configuración > Privacidad > Micrófono.",
    ))
    bus.publish(ev.EV_SUMMARY, ev.make_summary(build_summary(session.session_id)))

    for tick in itertools.count():
        bus.publish(ev.EV_LEVEL, ev.make_level([
            # Digital silence on the mic: the alarm state, always on show.
            LevelReading(track=TrackKind.MIC, dbfs=-95.0, peak=0.0, at_s=float(tick)),
            LevelReading(
                track=TrackKind.LOOPBACK,
                dbfs=-18.0 - (tick % 7) * 1.5,
                peak=0.4,
                at_s=float(tick),
            ),
        ]))
        bus.publish(ev.EV_VAD, ev.make_vad(TrackKind.LOOPBACK, tick % 4 < 2, float(tick)))
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8799, help="0 picks a free port")
    args = parser.parse_args()

    bus = InMemoryEventBus()
    session = build_session()
    bus.set_time_source(lambda: 0.0)

    panel = PanelServer(
        bus,
        port=args.port,
        token=TOKEN,
        state_provider=lambda: {"session": ev.make_session(session, 41.0)},
        on_rename=lambda _speaker_id, _name: 2,
        on_stop=lambda: None,
    )
    print(panel.start(), flush=True)  # noqa: T201 - this is the tool's whole output

    threading.Thread(
        target=publish_meeting, args=(bus, session), name="preview", daemon=True
    ).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        panel.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
