"""The command line interface. All user-facing text is Spanish.

``start`` is the only long-running command; everything else operates on the archive
and returns immediately. The shape of ``start`` is deliberate:

* Every failure that can be detected *before* recording begins is detected before
  recording begins -- devices, models, consent, the panel port. Discovering a
  missing model twenty minutes into a meeting is not acceptable.
* Ctrl-C is the normal way to end a session, not an abort. It runs the full drain
  and summarise sequence. A second Ctrl-C during that sequence exits immediately,
  because someone pressing it twice means it now.
* The panel URL contains the session token, so it is printed once and never logged.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import signal
import sys
import threading
import webbrowser
from pathlib import Path

import click

from meetscribe import __version__
from meetscribe import export as export_module
from meetscribe.assets import REQUIRED_MODELS, find_model, install_model
from meetscribe.bus.pubsub import InMemoryEventBus
from meetscribe.cli import doctor as doctor_module
from meetscribe.config import DEFAULT_CONFIG_YAML, Config, load_config, save_config
from meetscribe.core import events as ev
from meetscribe.core.clock import SystemClock
from meetscribe.core.models import ConsentMode, Session, SessionState, TrackKind, new_id
from meetscribe.errors import MeetScribeError
from meetscribe.logging_setup import setup_logging
from meetscribe.pipeline.factory import (
    build_diarizer,
    build_replay_factory,
    build_source_factory,
    build_transcriber,
    build_vad_factory,
)
from meetscribe.pipeline.session import SessionRunner, recover_interrupted_sessions
from meetscribe.security import consent as consent_module
from meetscribe.security.paths import (
    audio_dir,
    cloud_sync_root,
    config_path,
    data_dir,
    db_path,
    ensure_data_dirs,
    exports_dir,
    harden_directory,
)
from meetscribe.security.retention import delete_session_data, purge_expired_sessions
from meetscribe.security.secrets import delete_api_key, store_api_key
from meetscribe.store.sqlite_store import SqliteStore
from meetscribe.summarize.factory import EGRESS_WARNING_ES, build_summarizer
from meetscribe.summarize.prompts import format_timestamp
from meetscribe.web.server import PanelServer

__all__ = ["cli", "main"]


# ---------------------------------------------------------------------- output


def ok(message: str) -> None:
    click.echo(click.style("  ok  ", fg="green", bold=True) + message)


def warn(message: str) -> None:
    click.echo(click.style(" aviso", fg="yellow", bold=True) + " " + message)


def fail(message: str) -> None:
    click.echo(click.style(" error", fg="red", bold=True) + " " + message, err=True)


def head(message: str) -> None:
    click.echo()
    click.echo(click.style(message, bold=True))


def _open_store() -> SqliteStore:
    ensure_data_dirs()
    return SqliteStore(db_path())


def _load(config_file: Path | None = None) -> Config:
    return load_config(config_file)


# ------------------------------------------------------------------- the group


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", message="MeetScribe %(version)s")
@click.option("--config", "config_file", type=click.Path(path_type=Path),
              help="Ruta a un config.yaml alternativo.")
@click.option("--verbose", "-v", is_flag=True, help="Muestra el detalle técnico en la consola.")
@click.pass_context
def cli(ctx: click.Context, config_file: Path | None, verbose: bool) -> None:
    """MeetScribe: transcribe y resume tus reuniones sin que salgan de tu equipo."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    setup_logging(console_level="DEBUG" if verbose else "WARNING")


# ------------------------------------------------------------------------ start


@cli.command()
@click.option("--nombre", "-n", "name", default=None, help="Nombre de la reunión.")
@click.option("--consentimiento", "-c", "consent", default=None,
              help="anunciado | todos | solo-yo | ninguno")
@click.option("--nota", "consent_note", default="", help="Nota sobre el consentimiento.")
@click.option("--modelo", "model", default=None, help="Modelo de transcripción.")
@click.option("--idioma", "language", default=None, help="Idioma principal ('es', 'auto').")
@click.option("--sin-diarizacion", "no_diarization", is_flag=True,
              help="No separa hablantes remotos; todo se etiqueta 'Remoto'.")
@click.option("--offline", is_flag=True, help="Prohibe que cualquier texto salga del equipo.")
@click.option("--proveedor", "provider", default=None, help="anthropic | ollama | none")
@click.option("--dispositivo-mic", "mic_device", default=None,
              help="Índice o nombre del micrófono.")
@click.option("--dispositivo-loopback", "loopback_device", default=None,
              help="Índice o nombre del endpoint de loopback.")
@click.option("--solo-remoto", "remote_only", is_flag=True, help="No graba tu micrófono.")
@click.option("--solo-mic", "mic_only", is_flag=True, help="No graba el audio de la reunión.")
@click.option("--sin-panel", "no_panel", is_flag=True, help="No abre el panel web.")
@click.option("--panel-port", type=int, default=None, help="Puerto del panel (0 = automático).")
@click.option("--no-abrir", "no_open", is_flag=True, help="No abre el navegador.")
@click.option("--guardar-audio", "retain_audio", type=click.Choice(["none", "session", "keep"]),
              default=None, help="Política de retención del audio bruto.")
@click.option("--replay", type=click.Path(exists=True, path_type=Path), default=None,
              help="Reproduce un WAV en lugar de capturar (para pruebas).")
@click.pass_context
def start(ctx: click.Context, **opts: object) -> None:
    """Inicia una sesión: captura, transcribe y resume la reunión."""
    config = _load(ctx.obj.get("config_file"))
    _apply_start_overrides(config, opts)

    ensure_data_dirs()
    _first_run_notice(config, ctx.obj.get("config_file"))

    consent_mode = _resolve_consent(config, opts.get("consent"))
    store = _open_store()

    try:
        purged = purge_expired_sessions(
            store, audio_dir(), config.privacy.retention_days
        )
        if purged:
            warn(f"Se eliminaron {len(purged)} sesiones por la política de retención.")
        for recovered in recover_interrupted_sessions(store):
            warn(
                f"La sesión '{recovered.name}' ({recovered.session_id}) quedó "
                "interrumpida; su transcripción está guardada. Resumela con "
                f"'meetscribe summarize {recovered.session_id}'."
            )

        runner, panel = _build_session(config, store, consent_mode, opts)
    except MeetScribeError as exc:
        store.close()
        fail(exc.user_message)
        if exc.detail:
            click.echo(f"        ({exc.detail})", err=True)
        raise SystemExit(2) from exc

    _run_until_stopped(runner, panel, store)


def _apply_start_overrides(config: Config, opts: dict) -> None:
    if opts.get("model"):
        config.stt.model = str(opts["model"])
    if opts.get("language"):
        config.stt.language = str(opts["language"])
    if opts.get("no_diarization"):
        config.diarization.enabled = False
    if opts.get("provider"):
        config.summarizer.provider = str(opts["provider"])
    if opts.get("offline"):
        config.privacy.offline_only = True
        if config.summarizer.provider == "anthropic":
            # --offline is an explicit instruction about this meeting, so it wins
            # over a config file that predates the decision.
            config.summarizer.provider = "ollama"
    if opts.get("no_panel"):
        config.panel.enabled = False
    if opts.get("panel_port") is not None:
        config.panel.port = int(opts["panel_port"])  # type: ignore[arg-type]
    if opts.get("retain_audio"):
        config.capture.retain_audio = str(opts["retain_audio"])
    if opts.get("mic_device"):
        config.capture.mic_device = str(opts["mic_device"])
    if opts.get("loopback_device"):
        config.capture.loopback_device = str(opts["loopback_device"])
    config.validate()


def _first_run_notice(config: Config, config_file: Path | None) -> None:
    if config.privacy.first_run_notice_shown:
        return
    click.echo()
    click.echo(click.style(consent_module.FIRST_RUN_NOTICE_ES, fg="cyan"))
    click.echo()
    if not click.confirm("¿Continuar?", default=True):
        raise SystemExit(1)
    config.privacy.first_run_notice_shown = True
    # Showing the notice again is harmless; failing to start is not.
    with contextlib.suppress(OSError):
        save_config(config, config_file)


def _resolve_consent(config: Config, value: object) -> ConsentMode:
    if value:
        try:
            return consent_module.parse_consent(str(value))
        except ValueError as exc:
            fail(str(exc))
            raise SystemExit(2) from exc

    if not config.privacy.require_consent:
        return ConsentMode.NOT_RECORDED

    if config.privacy.default_consent:
        # Answered once during setup. A wrong value here is still correctable per
        # meeting with --consentimiento, which is checked above.
        try:
            return consent_module.parse_consent(config.privacy.default_consent)
        except ValueError:
            warn(
                f"privacy.default_consent='{config.privacy.default_consent}' no es "
                "válido; preguntando."
            )

    click.echo()
    click.echo("¿Cómo se informó a los participantes de que se va a transcribir?")
    click.echo("  1) Lo anuncié en voz alta")
    click.echo("  2) Todos dieron su consentimiento explícito")
    click.echo("  3) Solo voy a conservar mi propia pista")
    click.echo("  4) No lo he informado")
    choice = click.prompt("Opción", type=click.Choice(["1", "2", "3", "4"]), default="1")
    mode = {
        "1": ConsentMode.ANNOUNCED,
        "2": ConsentMode.ALL_PARTIES,
        "3": ConsentMode.SELF_ONLY,
        "4": ConsentMode.NOT_RECORDED,
    }[choice]
    if mode is ConsentMode.NOT_RECORDED:
        click.echo()
        warn(
            "En muchos países grabar una llamada sin avisar es ilegal. "
            "Frase lista para leer:"
        )
        click.echo(click.style(f'  "{consent_module.ANNOUNCEMENT_ES}"', fg="cyan"))
        if not click.confirm("¿Grabar de todos modos?", default=False):
            raise SystemExit(1)
    return mode


def _build_session(
    config: Config, store: SqliteStore, consent_mode: ConsentMode, opts: dict
) -> tuple[SessionRunner, PanelServer | None]:
    clock = SystemClock()
    bus = InMemoryEventBus(history_limit=config.panel.history_limit)

    def notice(level: str, code: str, message: str) -> None:
        (warn if level != "info" else ok)(message)

    head("Preparando")
    replay = opts.get("replay")
    want_mic = not opts.get("remote_only")
    want_loopback = not opts.get("mic_only")
    if replay is not None:
        name = Path(str(replay)).name
        mic_factory = build_replay_factory(replay, TrackKind.MIC) if want_mic else None
        loopback_factory = (
            build_replay_factory(replay, TrackKind.LOOPBACK) if want_loopback else None
        )
        tracks = ", ".join(
            label for label, wanted in (("micrófono", want_mic), ("remoto", want_loopback))
            if wanted
        )
        ok(f"Reproduciendo '{name}' en la(s) pista(s): {tracks}")
    else:
        mic_factory = (
            build_source_factory(config, TrackKind.MIC) if want_mic else None
        )
        loopback_factory = (
            build_source_factory(config, TrackKind.LOOPBACK) if want_loopback else None
        )

    vad_factory = build_vad_factory(config)
    ok("Detección de voz lista (silero-vad)")

    diarizer = build_diarizer(config, notice)
    ok(
        "Separación de hablantes activa"
        if diarizer.enabled
        else "Separación de hablantes desactivada: todo lo remoto será 'Remoto'"
    )

    transcriber = build_transcriber(config)
    with click.progressbar(length=1, label=f"  cargando {config.stt.model}") as bar:
        transcriber.load()
        bar.update(1)

    summarizer = None
    if config.summarizer.provider != "none":
        try:
            summarizer = build_summarizer(config, offline_only=config.privacy.offline_only)
            if not summarizer.is_local:
                click.echo()
                click.echo(click.style(EGRESS_WARNING_ES, fg="yellow"))
            ok(f"Resumen: {summarizer.name} ({summarizer.model})")
        except MeetScribeError as exc:
            warn(f"{exc.user_message} La reunión se grabará sin resumen.")

    session = Session(
        session_id=new_id(),
        name=str(opts.get("name") or _default_name(clock.now())),
        started_at=clock.now(),
        language=config.stt.language,
        state=SessionState.IDLE,
        consent=consent_mode,
        consent_note=str(opts.get("consent_note") or ""),
        offline_only=config.privacy.offline_only,
        diarization_enabled=diarizer.enabled,
        stt_model=transcriber.name,
        audio_retained=config.capture.retain_audio == "keep",
    )

    runner = SessionRunner(
        session,
        config,
        store,
        bus,
        clock,
        transcriber=transcriber,
        diarizer=diarizer,
        mic_source_factory=mic_factory,
        loopback_source_factory=loopback_factory,
        vad_factory=vad_factory,
        summarizer=summarizer,
        realtime=replay is None,
        audio_dir=audio_dir(),
    )

    _mirror_notices(bus)

    panel: PanelServer | None = None
    if config.panel.enabled:
        panel = PanelServer(
            bus,
            host=config.panel.host,
            port=config.panel.port,
            heartbeat_s=config.panel.heartbeat_s,
            state_provider=lambda: _panel_state(runner, store),
            on_rename=runner.rename_speaker,
            on_stop=runner.stop,
        )
        panel.start()

    runner.start()

    head("Grabando")
    ok(f"Sesión '{session.name}' · id {session.session_id}")
    if panel is not None:
        click.echo("  panel " + click.style(panel.url, fg="cyan", underline=True))
        click.echo("        (el enlace incluye el token de acceso; no lo compartas)")
        if config.panel.open_browser and not opts.get("no_open"):
            webbrowser.open(panel.url)
    click.echo()
    click.echo("  Pulsa " + click.style("Ctrl-C", bold=True) + " para terminar y generar el acta.")
    return runner, panel


def _mirror_notices(bus) -> object:
    """Print the panel's notices to the terminal as well.

    Without this the operational messages -- a silent microphone, audio dropped
    under load, a summary that failed -- would reach only the web panel, and a
    user running with ``--sin-panel`` would be told nothing at all. That is the
    "no silent failures" rule applied to the CLI itself.
    """
    subscription = bus.subscribe(after_seq=0)

    def pump() -> None:
        printers = {"error": fail, "warn": warn, "info": ok}
        for event in subscription:
            if event.type != ev.EV_NOTICE:
                continue
            data = event.data
            printers.get(str(data.get("level")), ok)(str(data.get("message", "")))

    threading.Thread(target=pump, name="cli-notices", daemon=True).start()
    return subscription


def _default_name(now: dt.datetime) -> str:
    return f"Reunión {now.astimezone().strftime('%Y-%m-%d %H:%M')}"


def _panel_state(runner: SessionRunner, store: SqliteStore) -> dict:
    session = runner.session
    summary = store.get_summary(session.session_id)
    from meetscribe.core import events as ev

    return {
        "session": ev.make_session(session, runner.elapsed_s),
        "speakers": [ev.make_speaker(s) for s in store.get_speakers(session.session_id)],
        "segments": [ev.make_segment(s) for s in store.get_segments(session.session_id)],
        "action_items": [
            ev.make_action_item(a) for a in store.get_action_items(session.session_id)
        ],
        "summary": ev.make_summary(summary) if summary else None,
        "cursor": runner.bus.last_seq,
        "stats": runner.stats(),
    }


def _run_until_stopped(
    runner: SessionRunner, panel: PanelServer | None, store: SqliteStore
) -> None:
    interrupts = {"count": 0}

    def on_signal(signum, frame):
        interrupts["count"] += 1
        if interrupts["count"] >= 2:
            fail("Segunda interrupción: saliendo inmediatamente.")
            raise SystemExit(130)
        raise KeyboardInterrupt

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_signal)
    try:
        try:
            while not runner.wait(0.5):
                if runner.capture_finished:
                    # A finite source (--replay) reached the end of the file; there
                    # is nothing left to wait for.
                    click.echo("Fin del audio: cerrando la sesión...")
                    break
        except KeyboardInterrupt:
            click.echo()
            click.echo("Terminando: vaciando el audio pendiente y generando el acta...")
        session = runner.stop(summarize=True)
    finally:
        signal.signal(signal.SIGINT, previous)
        if panel is not None:
            panel.stop()

    _report_session(store, session.session_id)
    store.close()


def _report_session(store: SqliteStore, session_id: str) -> None:
    session = store.get_session(session_id)
    if session is None:
        return
    segments = store.get_segments(session_id)
    summary = store.get_summary(session_id)

    head("Resultado")
    duration = format_timestamp(segments[-1].end_s if segments else 0)
    ok(f"{len(segments)} intervenciones · {duration}")
    if summary is not None:
        ok(f"Acta generada con {summary.provider}/{summary.model}")
        click.echo()
        click.echo(click.style("  " + summary.executive, fg="white"))
        if summary.action_items:
            click.echo()
            click.echo(click.style("  Compromisos:", bold=True))
            for item in summary.action_items:
                owner = f"{item.owner}: " if item.owner else ""
                due = f" ({item.due})" if item.due else ""
                click.echo(f"    - {owner}{item.text}{due}")
    else:
        warn("Sin acta. Puedes generarla después:")
        click.echo(f"        meetscribe summarize {session_id}")
    click.echo()
    click.echo(f"  Exportar:  meetscribe export {session_id} --formato md")


# ----------------------------------------------------------------------- doctor


@cli.command()
@click.option("--sin-audio", "no_audio", is_flag=True, help="Omite las pruebas de dispositivos.")
@click.option("--segundos", "seconds", default=2.0, help="Duración de cada prueba de audio.")
@click.pass_context
def doctor(ctx: click.Context, no_audio: bool, seconds: float) -> None:
    """Comprueba que este equipo puede grabar una reunión ahora mismo."""
    try:
        config = _load(ctx.obj.get("config_file"))
    except MeetScribeError as exc:
        fail(exc.user_message)
        raise SystemExit(2) from exc

    head("Diagnóstico de MeetScribe")
    if not no_audio:
        click.echo("  (las pruebas de audio abren los dispositivos reales durante "
                   f"{seconds:.0f}s cada una)")
    click.echo()

    checks = doctor_module.run_all(config, with_audio=not no_audio, seconds=seconds)
    colours = {
        doctor_module.OK: ("green", "  ok  "),
        doctor_module.WARN: ("yellow", " aviso"),
        doctor_module.ERROR: ("red", " error"),
    }
    for check in checks:
        colour, marker = colours[check.status]
        click.echo(
            click.style(marker, fg=colour, bold=True)
            + f" {check.name}: {check.detail}"
        )
        if check.fix and check.status != doctor_module.OK:
            click.echo(f"        -> {check.fix}")

    errors = [c for c in checks if c.status == doctor_module.ERROR]
    warnings = [c for c in checks if c.status == doctor_module.WARN]
    click.echo()
    if errors:
        fail(f"{len(errors)} problema(s) impiden grabar correctamente.")
        raise SystemExit(1)
    if warnings:
        warn(f"Todo listo, con {len(warnings)} aviso(s).")
    else:
        ok("Todo listo.")


# ---------------------------------------------------------------------- devices


@cli.command()
def devices() -> None:
    """Lista los dispositivos de audio disponibles."""
    from meetscribe.capture.devices import list_devices

    try:
        found = list_devices()
    except MeetScribeError as exc:
        fail(exc.user_message)
        raise SystemExit(2) from exc

    head("Entradas (micrófonos)")
    for device in found:
        if not device.is_loopback:
            click.echo("  " + device.describe())
    head("Loopback (audio de la reunión)")
    for device in found:
        if device.is_loopback:
            click.echo("  " + device.describe())
    click.echo()
    click.echo("  * = predeterminado.  Usa el índice con --dispositivo-mic / "
               "--dispositivo-loopback.")


# --------------------------------------------------------------------- sessions


@cli.command(name="sessions")
@click.option("--limite", "limit", default=25, help="Cuántas sesiones mostrar.")
def list_sessions(limit: int) -> None:
    """Lista las sesiones grabadas."""
    with _open_store() as store:
        rows = store.list_sessions(limit)
        if not rows:
            click.echo("  No hay sesiones todavía. Empieza con 'meetscribe start'.")
            return
        head(f"{len(rows)} sesión(es)")
        for session in rows:
            segments = store.get_segments(session.session_id)
            duration = format_timestamp(segments[-1].end_s if segments else 0)
            has_summary = "acta" if store.get_summary(session.session_id) else "    "
            state = "" if session.state is SessionState.STOPPED else f" [{session.state.value}]"
            click.echo(
                f"  {session.session_id}  "
                f"{session.started_at.astimezone().strftime('%Y-%m-%d %H:%M')}  "
                f"{duration:>8}  {len(segments):>5} int.  {has_summary}  "
                f"{session.name}{state}"
            )


@cli.command()
@click.argument("session_id")
@click.option("--completo", "full", is_flag=True, help="Muestra la transcripción entera.")
def show(session_id: str, full: bool) -> None:
    """Muestra una sesión: acta, compromisos y transcripción."""
    with _open_store() as store:
        try:
            session = store.resolve_session(session_id)
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc
        segments = store.get_segments(session.session_id)
        summary = store.get_summary(session.session_id)
        speakers = store.get_speakers(session.session_id)

    head(session.name)
    click.echo(
        f"  {session.started_at.astimezone().strftime('%Y-%m-%d %H:%M')} · "
        f"{format_timestamp(segments[-1].end_s if segments else 0)} · "
        f"{len(segments)} intervenciones · "
        f"{consent_module.describe_consent(session.consent)}"
    )
    if speakers:
        click.echo("  Participantes: " + ", ".join(s.display_name for s in speakers))

    if summary is not None:
        head("Acta")
        click.echo("  " + summary.executive)
        for title, items in (
            ("Decisiones", summary.decisions),
            ("Preguntas abiertas", summary.open_questions),
        ):
            if items:
                head(title)
                for item in items:
                    click.echo(f"  - {item}")
        if summary.action_items:
            head("Compromisos")
            for item in summary.action_items:
                owner = f"{item.owner}: " if item.owner else ""
                due = f" ({item.due})" if item.due else ""
                click.echo(f"  - {owner}{item.text}{due}")

    head("Transcripción")
    shown = segments if full else segments[:40]
    for segment in shown:
        click.echo(
            f"  [{format_timestamp(segment.start_s)}] "
            + click.style(f"{segment.speaker_name}:", bold=True)
            + f" {segment.text}"
        )
    if not full and len(segments) > len(shown):
        click.echo(f"  ... {len(segments) - len(shown)} más. Usa --completo.")


@cli.command(name="export")
@click.argument("session_id")
@click.option("--formato", "fmt", type=click.Choice(export_module.FORMATS), default="md")
@click.option("--salida", "-o", "output", type=click.Path(path_type=Path), default=None)
def export_session(session_id: str, fmt: str, output: Path | None) -> None:
    """Exporta una sesión a md, txt, json o srt."""
    with _open_store() as store:
        try:
            session = store.resolve_session(session_id)
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc
        text = export_module.render(
            fmt,
            session,
            store.get_segments(session.session_id),
            store.get_speakers(session.session_id),
            store.get_summary(session.session_id),
            store.get_action_items(session.session_id),
        )

    if output is None:
        exports_dir().mkdir(parents=True, exist_ok=True)
        output = exports_dir() / f"{Session.slug(session.name)}-{session.session_id}.{fmt}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    ok(f"Exportado a {output}")


@cli.command()
@click.argument("session_id")
@click.option("--proveedor", "provider", default=None, help="anthropic | ollama")
@click.option("--modelo", "model", default=None, help="Modelo de resumen.")
@click.pass_context
def summarize(ctx: click.Context, session_id: str, provider: str | None, model: str | None) -> None:
    """Genera (o regenera) el acta de una sesión ya grabada."""
    config = _load(ctx.obj.get("config_file"))
    if provider:
        config.summarizer.provider = provider
    if model:
        config.summarizer.model = model

    with _open_store() as store:
        try:
            session = store.resolve_session(session_id)
            segments = store.get_segments(session.session_id, final_only=True)
            if not segments:
                fail("Esa sesión no tiene transcripción.")
                raise SystemExit(1)
            summarizer = build_summarizer(config, offline_only=session.offline_only)
            if not summarizer.is_local:
                click.echo(click.style(EGRESS_WARNING_ES, fg="yellow"))
                if not click.confirm("¿Enviar la transcripción?", default=True):
                    raise SystemExit(1)
            from meetscribe.core.ports import SummaryRequest

            click.echo(f"  Generando el acta con {summarizer.name}/{summarizer.model}...")
            summary = summarizer.summarize(
                SummaryRequest(
                    session=session,
                    segments=segments,
                    speakers=store.get_speakers(session.session_id),
                    language=session.language,
                )
            )
            store.save_summary(summary)
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc

    ok("Acta generada.")
    click.echo()
    click.echo("  " + summary.executive)


@cli.command()
@click.argument("session_id")
@click.argument("speaker_id")
@click.argument("name")
def rename(session_id: str, speaker_id: str, name: str) -> None:
    """Renombra un hablante en toda la transcripción. Ej: rename ab12 p2 Maria"""
    with _open_store() as store:
        try:
            session = store.resolve_session(session_id)
            updated = store.rename_speaker(session.session_id, speaker_id, name)
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc
    ok(f"'{speaker_id}' ahora es '{name}' en {updated} intervenciones.")


@cli.command()
@click.argument("session_id")
@click.option("--si", "assume_yes", is_flag=True, help="No pregunta.")
def delete(session_id: str, assume_yes: bool) -> None:
    """Borra una sesión y su audio de forma permanente."""
    with _open_store() as store:
        try:
            session = store.resolve_session(session_id)
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc
        if not assume_yes and not click.confirm(
            f"Borrar '{session.name}' ({session.session_id}) y su audio. "
            "Esto no se puede deshacer. ¿Continuar?",
            default=False,
        ):
            raise SystemExit(1)
        delete_session_data(store, audio_dir(), session.session_id)
    ok("Sesión eliminada.")


# ----------------------------------------------------------------------- models


@cli.command(name="models")
@click.option("--download", "-d", "download", is_flag=True, help="Descarga lo que falte.")
@click.option("--stt", "stt", is_flag=True, help="Descarga también el modelo de Whisper.")
@click.pass_context
def models_cmd(ctx: click.Context, download: bool, stt: bool) -> None:
    """Muestra o descarga los modelos locales."""
    config = _load(ctx.obj.get("config_file"))
    ensure_data_dirs()

    if download:
        _download_models(config, include_stt=stt)
        return

    head("Modelos ONNX")
    for asset in REQUIRED_MODELS:
        path = find_model(asset)
        if path is None:
            warn(f"{asset.filename} ({asset.size_mb:.0f} MB) — falta · {asset.purpose}")
        else:
            ok(f"{asset.filename} — {path.parent} · {asset.purpose}")

    if stt:
        _download_stt_model(config)
    else:
        click.echo()
        click.echo("  Descarga lo que falte con 'meetscribe models --download'.")


def _download_models(config: Config, *, include_stt: bool) -> None:
    """Fetch every model this machine needs. Shared by ``models`` and ``setup``."""
    head("Modelos ONNX")
    for asset in REQUIRED_MODELS:
        try:
            with click.progressbar(
                length=asset.size_bytes, label=f"  {asset.filename}"
            ) as bar:
                seen = {"n": 0}

                def progress(done: int, total: int, bar=bar, seen=seen) -> None:
                    bar.update(done - seen["n"])
                    seen["n"] = done

                path, how = install_model(asset, progress)
            ok(f"{asset.filename}: {how} ({asset.size_mb:.0f} MB) -> {path.parent}")
        except MeetScribeError as exc:
            fail(exc.user_message)
            raise SystemExit(2) from exc
    if include_stt:
        _download_stt_model(config)


def _download_stt_model(config: Config) -> None:
    head(f"Modelo de transcripción ({config.stt.model})")
    click.echo("  descargando; puede tardar varios minutos la primera vez...")
    try:
        transcriber = build_transcriber(config)
        transcriber.load()
    except MeetScribeError as exc:
        fail(exc.user_message)
        raise SystemExit(2) from exc
    ok(f"{config.stt.model} listo.")


# ----------------------------------------------------------------------- config


SETUP_MODELS: tuple[tuple[str, str], ...] = (
    ("large-v3-turbo", "más preciso · 3.3x tiempo real · ~1.6 GB  (recomendado)"),
    ("small", "más rápido · 8.1x tiempo real · ~0.5 GB"),
    ("base", "rápido y ligero · 22x tiempo real"),
    ("tiny", "el más ligero · 42x tiempo real · menos preciso"),
)

SETUP_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("anthropic", "Claude · mejor acta · el TEXTO sale del equipo"),
    ("ollama", "local · nada sale del equipo · requiere Ollama instalado"),
    ("none", "sin acta · solo transcripción"),
)


def _choose(prompt: str, options: tuple[tuple[str, str], ...], default: int = 1) -> str:
    """A numbered menu. Numbers rather than free text: a typo in a model name
    would otherwise only surface at the start of a meeting."""
    click.echo()
    click.echo(f"  {prompt}")
    for index, (value, description) in enumerate(options, start=1):
        click.echo(f"    {index}) {click.style(value, bold=True)} — {description}")
    choice = click.prompt(
        "  Elige",
        type=click.IntRange(1, len(options)),
        default=default,
        show_default=True,
    )
    return options[choice - 1][0]


def apply_setup_answers(
    config: Config,
    *,
    model: str,
    language: str,
    provider: str,
    consent: ConsentMode,
    retain_audio: str,
    diarization: bool,
) -> Config:
    """Fold the wizard's answers into a config. Pure, so it can be tested without
    driving a single prompt."""
    config.stt.model = model
    config.stt.language = language
    config.summarizer.provider = provider
    config.privacy.offline_only = provider != "anthropic"
    config.privacy.first_run_notice_shown = True
    config.capture.retain_audio = retain_audio
    config.diarization.enabled = diarization
    config.privacy.default_consent = consent.value
    return config


@cli.command(name="setup")
@click.option("--defaults", "use_defaults", is_flag=True,
              help="No pregunta nada: escribe la configuración recomendada.")
@click.option("--con-modelos", "with_models", is_flag=True,
              help="Descarga también los modelos al terminar.")
@click.pass_context
def setup(ctx: click.Context, use_defaults: bool, with_models: bool) -> None:
    """Configura MeetScribe en este equipo, paso a paso.

    Pensado para la primera vez en una máquina nueva: escribe el config.yaml,
    restringe los permisos del directorio de datos, guarda la clave de API en el
    almacén del sistema si hace falta, y termina con un diagnóstico. Con
    --defaults no pregunta nada, que es como lo usa el instalador.
    """
    config_file = ctx.obj.get("config_file")
    root = ensure_data_dirs()
    head("Configurando MeetScribe")
    ok(f"Datos en {root}")
    if harden_directory(root):
        ok("Permisos restringidos al usuario actual.")
    else:
        warn("No se pudieron restringir los permisos del directorio de datos.")

    synced = cloud_sync_root(root)
    if synced is not None:
        warn(
            f"El directorio de datos está dentro de '{synced.name}' y se "
            "sincronizaría a la nube. Cambia MEETSCRIBE_DATA_DIR antes de grabar."
        )

    config = Config()
    if use_defaults:
        model, language, provider = "large-v3-turbo", "es", "none"
        consent, retain_audio, diarization = ConsentMode.ANNOUNCED, "none", True
    else:
        model = _choose("¿Qué modelo de transcripción?", SETUP_MODELS)
        language = _choose(
            "¿Idioma principal de las reuniones?",
            (("es", "español"), ("auto", "detectar (mezcla de idiomas)"), ("en", "inglés")),
        )
        provider = _choose("¿Cómo se genera el acta?", SETUP_PROVIDERS)
        consent_value = _choose(
            "¿Cómo informas a los participantes, normalmente?",
            (
                ("anunciado", "lo anuncias en voz alta al empezar"),
                ("todos", "todos dan su consentimiento explícito"),
                ("solo-yo", "solo se graba tu propia voz"),
            ),
        )
        consent = consent_module.parse_consent(consent_value)
        retain_audio = _choose(
            "¿Guardar el audio original?",
            (
                ("none", "no guardarlo nunca  (recomendado)"),
                ("session", "solo durante la sesión, luego se destruye"),
                ("keep", "conservarlo"),
            ),
        )
        diarization = click.confirm(
            "\n  ¿Separar a los participantes remotos por voz?", default=True
        )

    apply_setup_answers(
        config,
        model=model,
        language=language,
        provider=provider,
        consent=consent,
        retain_audio=retain_audio,
        diarization=diarization,
    )

    if provider == "anthropic":
        click.echo()
        click.echo(click.style(EGRESS_WARNING_ES, fg="yellow"))
        if not use_defaults and click.confirm("  ¿Guardar la clave de API ahora?", default=True):
            try:
                where = store_api_key("anthropic", click.prompt("  Clave", hide_input=True))
                ok(f"Clave guardada en {where}.")
            except MeetScribeError as exc:
                warn(f"{exc.user_message} Puedes hacerlo después con 'meetscribe config set-key'.")

    target = save_config(config, config_file)
    head("Listo")
    ok(f"Configuración escrita en {target}")

    if with_models or (not use_defaults and click.confirm(
        "  ¿Descargar los modelos ahora? (recomendado)", default=True
    )):
        _download_models(config, include_stt=True)

    click.echo()
    click.echo("  Comprueba el equipo:  meetscribe doctor")
    click.echo("  Empieza una reunión:  meetscribe start")


@cli.group(name="config")
def config_group() -> None:
    """Configuración, claves y permisos."""


@config_group.command(name="path")
def config_path_cmd() -> None:
    """Muestra dónde viven la configuración y los datos."""
    click.echo(f"  configuración : {config_path()}")
    click.echo(f"  datos         : {data_dir()}")
    click.echo(f"  base de datos : {db_path()}")
    click.echo(f"  audio         : {audio_dir()}")
    click.echo(f"  exportaciones : {exports_dir()}")


@config_group.command(name="init")
@click.option("--forzar", "force", is_flag=True, help="Sobrescribe la configuración existente.")
def config_init(force: bool) -> None:
    """Crea un config.yaml comentado con los valores por defecto."""
    ensure_data_dirs()
    target = config_path()
    if target.exists() and not force:
        warn(f"Ya existe {target}. Usa --forzar para sobrescribirlo.")
        raise SystemExit(1)
    target.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    ok(f"Configuración creada en {target}")


@config_group.command(name="show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Muestra la configuración efectiva (sin la clave de API)."""
    import yaml

    config = _load(ctx.obj.get("config_file"))
    click.echo(yaml.safe_dump(config.to_public_dict(), allow_unicode=True, sort_keys=False))


@config_group.command(name="set-key")
@click.option("--proveedor", "provider", default="anthropic")
def config_set_key(provider: str) -> None:
    """Guarda una clave de API en el almacén de credenciales del sistema."""
    key = click.prompt("Clave de API", hide_input=True)
    try:
        where = store_api_key(provider, key)
    except MeetScribeError as exc:
        fail(exc.user_message)
        raise SystemExit(2) from exc
    ok(f"Clave de '{provider}' guardada en {where}. No se escribió en ningún archivo.")


@config_group.command(name="delete-key")
@click.option("--proveedor", "provider", default="anthropic")
def config_delete_key(provider: str) -> None:
    """Elimina una clave guardada."""
    if delete_api_key(provider):
        ok(f"Clave de '{provider}' eliminada.")
    else:
        warn("No había ninguna clave guardada para ese proveedor.")


@config_group.command(name="harden")
def config_harden() -> None:
    """Restringe los permisos del directorio de datos al usuario actual."""
    root = ensure_data_dirs(harden=False)
    if harden_directory(root):
        ok(f"Permisos restringidos en {root}")
    else:
        warn(f"No se pudieron ajustar los permisos de {root}. Revisa el log.")


@cli.command(name="anuncio")
def announcement() -> None:
    """Imprime una frase lista para leer al inicio de la reunión."""
    click.echo()
    click.echo(click.style(f'  "{consent_module.ANNOUNCEMENT_ES}"', fg="cyan"))
    click.echo()


# -------------------------------------------------------------------- entry


def main() -> int:
    try:
        cli(obj={}, standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("\nCancelado.", err=True)
        return 130
    except SystemExit as exc:
        return int(exc.code or 0)
    except MeetScribeError as exc:
        fail(exc.user_message)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
