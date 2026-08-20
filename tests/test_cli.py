"""The CLI surface and the export formats.

Every command except ``start`` is covered end to end through Click's runner
against a real database in the isolated data directory. ``start`` itself is
covered by ``test_pipeline.py``, which exercises the same machinery without the
argument parsing.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from click.testing import CliRunner

from meetscribe import export as export_module
from meetscribe.cli import doctor as doctor_module
from meetscribe.cli.main import _resolve_consent, apply_setup_answers, cli
from meetscribe.config import Config, load_config
from meetscribe.core.models import (
    ActionItem,
    ConsentMode,
    SegmentState,
    Session,
    SessionState,
    Speaker,
    Summary,
    TrackKind,
    TranscriptSegment,
)
from meetscribe.security.paths import config_path, db_path, exports_dir
from meetscribe.store.sqlite_store import SqliteStore


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, *args: str, **kw):
    return runner.invoke(cli, list(args), obj={}, catch_exceptions=False, **kw)


# ------------------------------------------------------------------- fixtures


def build_archive() -> Session:
    """A finished session with two speakers, a transcript and minutes."""
    session = Session(
        session_id="abc123def456",
        name="Revisión de facturación",
        started_at=dt.datetime(2026, 3, 1, 10, 0, tzinfo=dt.UTC),
        ended_at=dt.datetime(2026, 3, 1, 10, 30, tzinfo=dt.UTC),
        state=SessionState.STOPPED,
        consent=ConsentMode.ANNOUNCED,
        stt_model="faster-whisper/large-v3-turbo/int8",
    )
    with SqliteStore(db_path()) as store:
        store.create_session(session)
        store.upsert_speaker(session.session_id, Speaker("yo", "Yo", TrackKind.MIC, True))
        store.upsert_speaker(
            session.session_id, Speaker("p1", "Participante 1", TrackKind.LOOPBACK)
        )
        rows = [
            (TrackKind.MIC, "yo", "Yo", 0.0, "Buenos días, vamos a revisar el avance."),
            (TrackKind.LOOPBACK, "p1", "Participante 1", 5.0, "El módulo ya está listo."),
            (TrackKind.MIC, "yo", "Yo", 9.5, "Perfecto, yo hablo con el banco."),
        ]
        for i, (track, sid, name, start, text) in enumerate(rows, start=1):
            store.add_segment(
                TranscriptSegment(
                    segment_id=f"seg{i}",
                    session_id=session.session_id,
                    track=track,
                    seq=i,
                    state=SegmentState.FINAL,
                    start_s=start,
                    end_s=start + 4.0,
                    text=text,
                    speaker_id=sid,
                    speaker_name=name,
                    language="es",
                )
            )
        store.save_summary(
            Summary(
                session_id=session.session_id,
                provider="anthropic",
                model="claude-opus-5",
                generated_at=dt.datetime(2026, 3, 1, 10, 31, tzinfo=dt.UTC),
                executive="Se revisó el avance del proyecto de facturación.",
                topics=("avance", "integración bancaria"),
                decisions=("escalar el bloqueo de credenciales",),
                action_items=(
                    ActionItem("hablar con el contacto comercial", owner="Yo", due="esta semana"),
                ),
                open_questions=("¿quién firma el contrato?",),
                follow_ups=("revisar el lunes",),
            )
        )
    return session


# ---------------------------------------------------------------------- basics


def test_help_lists_every_command(runner):
    result = invoke(runner, "--help")
    assert result.exit_code == 0
    for command in (
        "setup", "start", "doctor", "devices", "sessions", "show", "export",
        "summarize", "rename", "models", "config", "delete", "anuncio",
    ):
        assert command in result.output


def test_version(runner):
    result = invoke(runner, "--version")
    assert "MeetScribe" in result.output


def test_the_announcement_is_ready_to_read(runner):
    result = invoke(runner, "anuncio")
    assert result.exit_code == 0
    assert "transcribir" in result.output


def test_config_path_shows_where_everything_lives(runner):
    result = invoke(runner, "config", "path")
    assert result.exit_code == 0
    assert str(db_path()) in result.output
    assert "onedrive" not in result.output.lower()


def test_config_init_then_show(runner):
    result = invoke(runner, "config", "init")
    assert result.exit_code == 0
    assert config_path().exists()

    again = invoke(runner, "config", "init")
    assert again.exit_code == 1
    assert "Ya existe" in again.output

    forced = invoke(runner, "config", "init", "--forzar")
    assert forced.exit_code == 0

    shown = invoke(runner, "config", "show")
    assert "large-v3-turbo" in shown.output
    assert "api_key:" not in shown.output
    assert "api_key_configured" in shown.output


# ---------------------------------------------------------------------- setup


def test_setup_with_defaults_writes_a_usable_config(runner):
    """The unattended path the installer uses: no prompts, no network, no key."""
    result = invoke(runner, "setup", "--defaults")
    assert result.exit_code == 0, result.output
    assert config_path().exists()

    written = load_config(config_path())
    written.validate()
    assert written.stt.model == "large-v3-turbo"
    assert written.summarizer.provider == "none"
    # No provider means nothing can leave the machine, so offline is implied.
    assert written.privacy.offline_only is True
    assert written.privacy.first_run_notice_shown is True
    assert written.privacy.default_consent == "announced"
    assert written.capture.retain_audio == "none"


def test_setup_never_writes_the_api_key_to_the_config_file(runner):
    invoke(runner, "setup", "--defaults")
    assert "api_key" not in config_path().read_text(encoding="utf-8")


def test_setup_answers_are_folded_into_the_config():
    """The wizard's logic, tested without driving a single prompt."""
    config = apply_setup_answers(
        Config(),
        model="small",
        language="auto",
        provider="anthropic",
        consent=ConsentMode.ALL_PARTIES,
        retain_audio="session",
        diarization=False,
    )
    assert config.stt.model == "small"
    assert config.stt.language == "auto"
    assert config.summarizer.provider == "anthropic"
    # Anthropic is the one provider that sends text out, so offline must stay off
    # or the config would refuse to validate.
    assert config.privacy.offline_only is False
    assert config.privacy.default_consent == "all_parties"
    assert config.diarization.enabled is False
    config.validate()


def test_setup_is_interactive_when_not_given_defaults(runner):
    """Answers arrive as menu numbers; the last two prompts are confirmations."""
    answers = "\n".join(["2", "1", "3", "2", "1", "n", "n"]) + "\n"
    result = invoke(runner, "setup", input=answers)
    assert result.exit_code == 0, result.output
    written = load_config(config_path())
    assert written.stt.model == "small"          # option 2
    assert written.summarizer.provider == "none"  # option 3
    assert written.privacy.default_consent == "all_parties"  # option 2
    assert written.diarization.enabled is False   # declined
    written.validate()


def test_a_configured_default_consent_stops_start_from_asking():
    config = Config()
    config.privacy.default_consent = "todos"
    assert _resolve_consent(config, None) is ConsentMode.ALL_PARTIES


def test_an_explicit_consent_flag_overrides_the_configured_default():
    """The stored answer is a habit; the flag is a claim about this meeting."""
    config = Config()
    config.privacy.default_consent = "todos"
    assert _resolve_consent(config, "solo-yo") is ConsentMode.SELF_ONLY


def test_an_invalid_default_consent_falls_back_to_asking(runner):
    """A typo in the stored answer must not silently become "not recorded"."""
    config = Config()
    config.privacy.default_consent = "quizas"
    # Click's isolation supplies stdin, so the prompt is answered rather than
    # reading the real terminal.
    with runner.isolation(input="2\n"):
        mode = _resolve_consent(config, None)
    assert mode is ConsentMode.ALL_PARTIES


def test_config_show_reports_a_broken_config_file(runner):
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text("stt:\n  modelo: small\n", encoding="utf-8")
    result = runner.invoke(cli, ["config", "show"], obj={})
    assert result.exit_code != 0


# -------------------------------------------------------------------- archive


def test_sessions_is_helpful_when_empty(runner):
    result = invoke(runner, "sessions")
    assert result.exit_code == 0
    assert "meetscribe start" in result.output


def test_sessions_lists_what_is_there(runner):
    session = build_archive()
    result = invoke(runner, "sessions")
    assert session.session_id in result.output
    assert "Revisión de facturación" in result.output
    assert "acta" in result.output


def test_show_renders_the_acta_and_the_transcript(runner):
    build_archive()
    result = invoke(runner, "show", "abc123")
    assert result.exit_code == 0
    assert "Se revisó el avance" in result.output
    assert "hablar con el contacto comercial" in result.output
    assert "Buenos días" in result.output
    assert "¿quién firma el contrato?" in result.output


def test_show_reports_an_unknown_session(runner):
    result = runner.invoke(cli, ["show", "zzz"], obj={})
    assert result.exit_code == 2
    assert "No hay ninguna sesión" in result.output


@pytest.mark.parametrize("fmt", ["md", "txt", "json", "srt"])
def test_export_writes_every_format(runner, tmp_path, fmt):
    build_archive()
    target = tmp_path / f"out.{fmt}"
    result = invoke(runner, "export", "abc123", "--formato", fmt, "--salida", str(target))
    assert result.exit_code == 0
    text = target.read_text(encoding="utf-8")
    assert "Buenos días" in text


def test_export_defaults_to_the_exports_directory(runner):
    build_archive()
    result = invoke(runner, "export", "abc123")
    assert result.exit_code == 0
    files = list(exports_dir().glob("*.md"))
    assert len(files) == 1
    assert files[0].name.startswith("revision-de-facturacion-")


def test_rename_applies_everywhere(runner):
    build_archive()
    result = invoke(runner, "rename", "abc123", "p1", "María")
    assert result.exit_code == 0
    assert "2 intervenciones" not in result.output  # only one remote line here
    with SqliteStore(db_path()) as store:
        rows = store.get_segments("abc123def456")
    assert any(r.speaker_name == "María" for r in rows)


def test_delete_asks_before_destroying(runner):
    build_archive()
    declined = runner.invoke(cli, ["delete", "abc123"], obj={}, input="n\n")
    assert declined.exit_code == 1
    with SqliteStore(db_path()) as store:
        assert store.get_session("abc123def456") is not None

    accepted = invoke(runner, "delete", "abc123", "--si")
    assert accepted.exit_code == 0
    with SqliteStore(db_path()) as store:
        assert store.get_session("abc123def456") is None


def test_summarize_refuses_a_session_with_no_transcript(runner):
    session = Session(
        session_id="vacia0000000",
        name="Vacía",
        started_at=dt.datetime(2026, 3, 1, tzinfo=dt.UTC),
        state=SessionState.STOPPED,
    )
    with SqliteStore(db_path()) as store:
        store.create_session(session)
    result = runner.invoke(cli, ["summarize", "vacia", "--proveedor", "none"], obj={})
    assert result.exit_code != 0


def test_summarize_honours_an_offline_session(runner):
    build_archive()
    with SqliteStore(db_path()) as store:
        import dataclasses

        store.update_session(
            dataclasses.replace(store.get_session("abc123def456"), offline_only=True)
        )
    result = runner.invoke(
        cli, ["summarize", "abc123", "--proveedor", "anthropic"], obj={}
    )
    assert result.exit_code == 2
    assert "solo-local" in result.output


# --------------------------------------------------------------------- doctor


def test_doctor_reports_and_exits_nonzero_on_a_real_problem(runner, monkeypatch):
    monkeypatch.setattr(
        doctor_module,
        "run_all",
        lambda config, with_audio=True, seconds=2.0: [
            doctor_module.Check("Algo", doctor_module.OK, "bien"),
            doctor_module.Check("Otra cosa", doctor_module.WARN, "regular", "haz esto"),
            doctor_module.Check("Micrófono", doctor_module.ERROR, "silencio digital", "revisa"),
        ],
    )
    result = runner.invoke(cli, ["doctor", "--sin-audio"], obj={})
    assert result.exit_code == 1
    assert "silencio digital" in result.output
    assert "-> revisa" in result.output
    assert "impiden grabar" in result.output


def test_doctor_is_happy_when_everything_passes(runner, monkeypatch):
    monkeypatch.setattr(
        doctor_module,
        "run_all",
        lambda config, with_audio=True, seconds=2.0: [
            doctor_module.Check("Algo", doctor_module.OK, "bien")
        ],
    )
    result = runner.invoke(cli, ["doctor", "--sin-audio"], obj={})
    assert result.exit_code == 0
    assert "Todo listo" in result.output


def test_individual_doctor_checks_are_meaningful():
    config = Config()
    assert doctor_module.check_dependencies().status == doctor_module.OK
    assert doctor_module.check_data_dir().status in (doctor_module.OK, doctor_module.WARN)
    assert doctor_module.check_database().status == doctor_module.OK

    budget = doctor_module.check_cpu_budget(config)
    assert "RTF estimado" in budget.detail

    config.privacy.offline_only = True
    config.summarizer.provider = "ollama"
    assert "solo-local" in doctor_module.check_privacy(config).detail

    config.privacy.offline_only = False
    config.summarizer.provider = "anthropic"
    privacy = doctor_module.check_privacy(config)
    assert privacy.status == doctor_module.WARN
    assert "Anthropic" in privacy.detail


def test_doctor_flags_a_data_dir_inside_onedrive(monkeypatch, tmp_path):
    fake = tmp_path / "OneDrive" / "MeetScribe"
    fake.mkdir(parents=True)
    monkeypatch.setenv("MEETSCRIBE_DATA_DIR", str(fake))
    check = doctor_module.check_data_dir()
    assert check.status == doctor_module.ERROR
    assert "OneDrive" in check.detail


def test_doctor_rejects_an_unsupported_stt_model():
    config = Config()
    config.stt.model = "medium"
    check = doctor_module.check_stt_model(config)
    assert check.status == doctor_module.ERROR


# --------------------------------------------------------------------- export


def test_markdown_export_carries_the_provenance():
    session = build_archive()
    with SqliteStore(db_path()) as store:
        text = export_module.render(
            "md",
            session,
            store.get_segments(session.session_id),
            store.get_speakers(session.session_id),
            store.get_summary(session.session_id),
        )
    assert "# Revisión de facturación" in text
    assert "se anunció en voz alta" in text
    assert "**Audio conservado:** no" in text
    assert "- [ ] **Yo**: hablar con el contacto comercial" in text
    assert "## Transcripción" in text
    assert "claude-opus-5" in text


def test_json_export_is_machine_readable():
    session = build_archive()
    with SqliteStore(db_path()) as store:
        payload = json.loads(
            export_module.render(
                "json",
                session,
                store.get_segments(session.session_id),
                store.get_speakers(session.session_id),
                store.get_summary(session.session_id),
                store.get_action_items(session.session_id),
            )
        )
    assert payload["session"]["session_id"] == session.session_id
    assert payload["session"]["consent"] == "announced"
    assert len(payload["segments"]) == 3
    assert payload["segments"][0]["speaker_name"] == "Yo"
    assert payload["summary"]["model"] == "claude-opus-5"
    assert payload["action_items"][0]["owner"] == "Yo"


def test_srt_export_produces_valid_cues():
    session = build_archive()
    with SqliteStore(db_path()) as store:
        text = export_module.render("srt", session, store.get_segments(session.session_id))
    blocks = [b for b in text.split("\n\n") if b.strip()]
    assert len(blocks) == 3
    assert blocks[0].splitlines()[0] == "1"
    assert "00:00:00,000 --> 00:00:04,000" in blocks[0]
    assert "Yo: Buenos días" in blocks[0]


def test_srt_never_emits_a_zero_length_cue():
    session = Session(
        session_id="x", name="x", started_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    segment = TranscriptSegment(
        segment_id="s", session_id="x", track=TrackKind.MIC, start_s=1.0, end_s=1.0,
        text="ya", speaker_id="yo", speaker_name="Yo", seq=1,
    )
    text = export_module.render("srt", session, [segment])
    assert "00:00:01,000 --> 00:00:01,600" in text


def test_text_export_is_plain_and_complete():
    session = build_archive()
    with SqliteStore(db_path()) as store:
        text = export_module.render(
            "txt", session, store.get_segments(session.session_id),
            summary=store.get_summary(session.session_id),
        )
    assert "RESUMEN" in text
    assert "COMPROMISOS" in text
    assert "TRANSCRIPCIÓN" in text
    assert "[00:09] Yo: Perfecto" in text


def test_an_unknown_format_is_refused():
    session = build_archive()
    with pytest.raises(ValueError, match="formato desconocido"):
        export_module.render("pdf", session, [])


def test_export_of_a_session_with_no_summary_still_works():
    session = Session(
        session_id="y", name="Sin acta", started_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    )
    text = export_module.render("md", session, [])
    assert "no generado" in text
    assert "## Transcripción" in text
