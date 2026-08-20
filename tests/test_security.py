"""Security, privacy, configuration and model integrity.

The redaction tests matter more than they look: they are the proof that "the API
key cannot reach a log file" is a property of the code rather than a promise about
future contributors' discipline.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import types

import pytest

from meetscribe import assets as assets_module
from meetscribe.assets import (
    REQUIRED_MODELS,
    ModelAsset,
    find_model,
    install_model,
    resolve_model,
    sha256_file,
    verify_model,
)
from meetscribe.config import Config, load_config, save_config
from meetscribe.core.models import ConsentMode
from meetscribe.errors import (
    ConfigError,
    EgressBlockedError,
    ModelIntegrityError,
    ModelNotFoundError,
    SecretNotFoundError,
)
from meetscribe.security import consent as consent_module
from meetscribe.security import secrets as secrets_module
from meetscribe.security.paths import (
    audio_dir,
    cloud_sync_root,
    config_path,
    data_dir,
    db_path,
    ensure_data_dirs,
    models_dir,
)
from meetscribe.security.redact import (
    RedactingFilter,
    install_log_redaction,
    redact_pii,
    redact_secrets,
)
from meetscribe.security.retention import (
    delete_session_data,
    purge_expired_sessions,
    shred_file,
    shred_session_audio,
)

# -------------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    "text",
    [
        "usando sk-ant-api03-AbCdEf123456789012345 para la llamada",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "api_key=supersecretvalue123",
        "token: 'abcdefghijklmnop'",
        "abrir http://127.0.0.1:8765/?t=Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5",
        "HF token hf_abcdefghijklmnopqrst",
    ],
)
def test_credentials_are_stripped_from_text(text):
    cleaned = redact_secrets(text)
    assert "[REDACTADO]" in cleaned
    for secret in (
        "AbCdEf123456789012345",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "supersecretvalue123",
        "abcdefghijklmnop",
        "Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5",
        "abcdefghijklmnopqrst",
    ):
        assert secret not in cleaned or secret not in text


def test_ordinary_prose_is_left_alone():
    text = "La reunión duró 45 minutos y asistieron 3 personas."
    assert redact_secrets(text) == text


def test_the_log_filter_scrubs_both_the_message_and_its_arguments(caplog):
    """``log.info("key=%s", key)`` puts the secret in ``record.args``, which a
    message-only filter would miss entirely."""
    logger = logging.getLogger("meetscribe.test.redact.args")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("clave sk-ant-api03-INLINESECRET1234567890 y %s",
                    "sk-ant-api03-ARGSECRET1234567890")
    assert "INLINESECRET" not in caplog.text
    assert "ARGSECRET" not in caplog.text
    assert "[REDACTADO]" in caplog.text


def test_the_log_filter_handles_dict_style_args(caplog):
    logger = logging.getLogger("meetscribe.test.redact.dict")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("%(k)s", {"k": "sk-ant-api03-DICTSECRET1234567890"})
    assert "DICTSECRET" not in caplog.text


def test_installing_redaction_covers_the_logger_and_its_handlers(caplog):
    logger = logging.getLogger("meetscribe.test.redaction")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    redactor = install_log_redaction(logger)
    assert redactor in logger.filters
    assert redactor in handler.filters
    logger.removeHandler(handler)


@pytest.mark.parametrize(
    "text, marker",
    [
        ("escríbeme a ana.perez@empresa.com", "[correo]"),
        ("mi DNI es 12345678Z", "[documento]"),
        ("la cuenta ES91 2100 0418 4502 0005 1332", "[iban]"),
        ("el expediente 123456789012", "[número]"),
    ],
)
def test_pii_is_redacted_before_egress(text, marker):
    assert marker in redact_pii(text)


def test_phone_redaction_can_be_switched_off():
    text = "llámame al 600 123 456"
    assert "[teléfono]" in redact_pii(text)
    assert "[teléfono]" not in redact_pii(text, phones=False)


# ----------------------------------------------------------------------- tokens


def test_session_tokens_are_long_and_unique():
    tokens = {secrets_module.new_session_token() for _ in range(100)}
    assert len(tokens) == 100
    assert all(len(t) >= 40 for t in tokens)


def test_token_comparison_rejects_everything_but_the_exact_value():
    token = secrets_module.new_session_token()
    assert secrets_module.token_matches(token, token)
    assert not secrets_module.token_matches(token, token[:-1])
    assert not secrets_module.token_matches(token, "")
    assert not secrets_module.token_matches(token, None)
    assert not secrets_module.token_matches("", token)


# ---------------------------------------------------------------- key resolution


def test_the_environment_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-from-env  ")
    assert secrets_module.resolve_api_key("anthropic") == "sk-ant-from-env"


def test_the_keyring_is_used_before_the_config_file(monkeypatch):
    stub = types.SimpleNamespace(
        get_password=lambda service, user: "sk-ant-from-keyring",
        set_password=lambda *a: None,
        delete_password=lambda *a: None,
        get_keyring=object,
    )
    monkeypatch.setitem(sys.modules, "keyring", stub)
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: True)
    assert (
        secrets_module.resolve_api_key("anthropic", config_value="sk-ant-from-file")
        == "sk-ant-from-keyring"
    )


def test_the_config_file_is_the_last_resort(monkeypatch):
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: False)
    assert (
        secrets_module.resolve_api_key("anthropic", config_value="sk-ant-from-file")
        == "sk-ant-from-file"
    )


def test_a_missing_key_explains_both_ways_to_provide_one(monkeypatch):
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: False)
    with pytest.raises(SecretNotFoundError) as excinfo:
        secrets_module.resolve_api_key("anthropic")
    assert "config set-key" in excinfo.value.user_message
    assert "ANTHROPIC_API_KEY" in excinfo.value.user_message


def test_a_missing_key_can_be_optional(monkeypatch):
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: False)
    assert secrets_module.resolve_api_key("anthropic", required=False) is None


def test_storing_an_empty_key_is_refused(monkeypatch):
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: True)
    with pytest.raises(SecretNotFoundError):
        secrets_module.store_api_key("anthropic", "   ")


def test_storing_without_a_backend_says_what_to_do_instead(monkeypatch):
    monkeypatch.setattr(secrets_module, "keyring_available", lambda: False)
    with pytest.raises(SecretNotFoundError) as excinfo:
        secrets_module.store_api_key("anthropic", "sk-ant-x")
    assert "ANTHROPIC_API_KEY" in excinfo.value.user_message


# ---------------------------------------------------------------------- consent


@pytest.mark.parametrize(
    "value, expected",
    [
        ("anunciado", ConsentMode.ANNOUNCED),
        ("ALL", ConsentMode.ALL_PARTIES),
        ("solo-yo", ConsentMode.SELF_ONLY),
        ("ninguno", ConsentMode.NOT_RECORDED),
        (None, ConsentMode.NOT_RECORDED),
    ],
)
def test_consent_parsing(value, expected):
    assert consent_module.parse_consent(value) is expected


def test_an_unrecognised_consent_value_lists_the_options():
    with pytest.raises(ValueError, match="anunciado"):
        consent_module.parse_consent("quizá")


def test_consent_sufficiency_gates_recording():
    assert consent_module.consent_is_sufficient(ConsentMode.ANNOUNCED, require=True)
    assert consent_module.consent_is_sufficient(ConsentMode.ALL_PARTIES, require=True)
    assert consent_module.consent_is_sufficient(ConsentMode.SELF_ONLY, require=True)
    assert not consent_module.consent_is_sufficient(ConsentMode.NOT_RECORDED, require=True)
    assert consent_module.consent_is_sufficient(ConsentMode.NOT_RECORDED, require=False)


def test_every_consent_mode_has_a_spanish_description():
    for mode in ConsentMode:
        assert consent_module.describe_consent(mode)


def test_the_announcement_helper_is_ready_to_read_aloud():
    assert "transcribir" in consent_module.ANNOUNCEMENT_ES
    assert "equipo" in consent_module.ANNOUNCEMENT_ES


def test_the_first_run_notice_names_the_only_egress_path():
    assert "anthropic" in consent_module.FIRST_RUN_NOTICE_ES
    assert "ollama" in consent_module.FIRST_RUN_NOTICE_ES
    assert "consentimiento" in consent_module.FIRST_RUN_NOTICE_ES


# --------------------------------------------------------------------- retention


def test_shredding_overwrites_and_removes(tmp_path):
    path = tmp_path / "secreto.wav"
    path.write_bytes(b"contenido muy privado" * 100)
    assert shred_file(path)
    assert not path.exists()


def test_shredding_a_missing_file_is_not_an_error(tmp_path):
    assert not shred_file(tmp_path / "nope.wav")


def test_shredding_a_session_clears_its_directory(tmp_path):
    directory = tmp_path / "sess"
    directory.mkdir()
    for name in ("mic.wav", "loopback.wav"):
        (directory / name).write_bytes(b"x" * 1000)
    assert shred_session_audio(directory) == 2
    assert not directory.exists()


def test_retention_is_off_by_default(store, session):
    assert purge_expired_sessions(store, audio_dir(), retention_days=0) == []


def test_retention_removes_only_expired_sessions(store, session, tmp_path):
    import dataclasses

    old = dataclasses.replace(
        session, session_id="viejo", started_at=dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    )
    store.create_session(old)
    store.create_session(session)
    (tmp_path / "viejo").mkdir()
    (tmp_path / "viejo" / "mic.wav").write_bytes(b"x" * 100)

    removed = purge_expired_sessions(
        store, tmp_path, retention_days=30, now=dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    )
    assert removed == ["viejo"]
    assert store.get_session("viejo") is None
    assert store.get_session(session.session_id) is not None
    assert not (tmp_path / "viejo").exists()


def test_deleting_a_session_removes_rows_and_audio(store, session, tmp_path):
    store.create_session(session)
    directory = tmp_path / session.session_id
    directory.mkdir()
    (directory / "mic.wav").write_bytes(b"x" * 100)

    delete_session_data(store, tmp_path, session.session_id)
    assert store.get_session(session.session_id) is None
    assert not directory.exists()


# ------------------------------------------------------------------------ paths


def test_the_data_directory_honours_the_override_and_is_not_in_onedrive():
    root = data_dir()
    assert root.name == "data"
    assert "onedrive" not in str(root).lower()
    assert config_path().parent == root
    assert db_path().parent == root


@pytest.fixture
def no_sync_env(monkeypatch):
    """Neutralise the real machine's sync variables so only the path is judged."""
    for variable in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize(
    "relative",
    [
        "OneDrive/MeetScribe",
        "OneDrive - Contoso/data",
        "Dropbox/meetings",
        "Google Drive/notes",
    ],
)
def test_a_synced_folder_is_detected_by_component(no_sync_env, tmp_path, relative):
    target = tmp_path.joinpath(*relative.split("/"))
    target.mkdir(parents=True)
    assert cloud_sync_root(target) is not None


@pytest.mark.parametrize(
    "relative",
    [
        # The bug this guards: a substring match called every one of these synced.
        "temp/C--Users-me-OneDrive-Documents-project/data",
        "OneDriveBackupTool/data",
        "my-onedrive-notes/data",
        "local/MeetScribe",
    ],
)
def test_a_name_that_merely_contains_a_sync_word_is_not_synced(
    no_sync_env, tmp_path, relative
):
    target = tmp_path.joinpath(*relative.split("/"))
    target.mkdir(parents=True)
    assert cloud_sync_root(target) is None


def test_the_sync_root_comes_from_the_environment_when_windows_declares_it(
    no_sync_env, monkeypatch, tmp_path
):
    """The env var is authoritative: a synced folder need not be *named* OneDrive."""
    root = tmp_path / "Documentos"
    target = root / "reuniones"
    target.mkdir(parents=True)
    monkeypatch.setenv("OneDrive", str(root))
    assert cloud_sync_root(target) == root.resolve()
    assert cloud_sync_root(tmp_path / "otro") is None


def test_ensure_data_dirs_creates_the_whole_tree():
    ensure_data_dirs(harden=False)
    for path in (data_dir(), models_dir(), audio_dir()):
        assert path.is_dir()
    ensure_data_dirs(harden=False)  # idempotent


# --------------------------------------------------------------------- config


def test_defaults_are_the_documented_ones():
    config = Config()
    assert config.stt.model == "large-v3-turbo"
    assert config.stt.language == "es"
    assert config.diarization.threshold == 0.60
    assert config.vad.min_silence_duration == 0.45
    assert config.capture.retain_audio == "none"
    assert config.panel.host == "127.0.0.1"
    assert config.privacy.require_consent is True
    assert config.summarizer.model == "claude-opus-5"


def test_an_unknown_key_is_an_error_with_the_valid_alternatives():
    """A silently ignored typo leaves the user sure they configured something
    they did not."""
    with pytest.raises(ConfigError) as excinfo:
        Config.from_dict({"diarisation": {"enabled": False}})
    assert "diarisation" in excinfo.value.user_message
    assert "diarization" in excinfo.value.user_message


def test_an_unknown_nested_key_names_its_full_path():
    with pytest.raises(ConfigError, match=r"stt\.modelo"):
        Config.from_dict({"stt": {"modelo": "small"}})


def test_the_panel_cannot_be_exposed_by_accident():
    config = Config.from_dict({"panel": {"host": "0.0.0.0"}})
    with pytest.raises(ConfigError) as excinfo:
        config.validate()
    assert "127.0.0.1" in excinfo.value.user_message

    allowed = Config.from_dict({"panel": {"host": "0.0.0.0", "allow_non_local": True}})
    allowed.validate()  # explicit opt-in is honoured


def test_offline_mode_conflicts_with_a_remote_provider():
    config = Config.from_dict(
        {"privacy": {"offline_only": True}, "summarizer": {"provider": "anthropic"}}
    )
    with pytest.raises(ConfigError, match="offline_only"):
        config.validate()


def test_invalid_enumerations_are_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"capture": {"retain_audio": "siempre"}}).validate()
    with pytest.raises(ConfigError):
        Config.from_dict({"summarizer": {"provider": "openai"}}).validate()
    with pytest.raises(ConfigError):
        Config.from_dict({"panel": {"port": 99999}}).validate()


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("MEETSCRIBE_STT_MODEL", "small")
    monkeypatch.setenv("MEETSCRIBE_PANEL_PORT", "9000")
    monkeypatch.setenv("MEETSCRIBE_OFFLINE", "1")
    monkeypatch.setenv("MEETSCRIBE_PROVIDER", "ollama")
    config = Config().apply_env()
    assert config.stt.model == "small"
    assert config.panel.port == 9000
    assert config.privacy.offline_only is True


def test_the_api_key_never_appears_in_a_dump():
    config = Config()
    config.summarizer.api_key = "sk-ant-should-not-appear"
    public = config.to_public_dict()
    assert "api_key" not in public["summarizer"]
    assert public["summarizer"]["api_key_configured"] is True
    assert "sk-ant" not in str(public)


def test_saving_a_config_omits_the_key_and_reloads(tmp_path):
    config = Config()
    config.summarizer.api_key = "sk-ant-secret"
    config.stt.model = "small"
    path = save_config(config, tmp_path / "config.yaml")
    assert "sk-ant-secret" not in path.read_text(encoding="utf-8")

    reloaded = load_config(path)
    assert reloaded.stt.model == "small"
    assert reloaded.summarizer.api_key is None


def test_loading_a_broken_yaml_file_explains_where(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("stt: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_loading_a_missing_file_yields_the_defaults(tmp_path):
    assert load_config(tmp_path / "absent.yaml").stt.model == "large-v3-turbo"


def test_the_bundled_default_config_is_valid_yaml():
    import yaml

    from meetscribe.config import DEFAULT_CONFIG_YAML

    parsed = yaml.safe_load(DEFAULT_CONFIG_YAML)
    Config.from_dict(parsed).validate()


# ---------------------------------------------------------------------- models


def make_asset(tmp_path, content: bytes = b"modelo falso") -> tuple[ModelAsset, object]:
    source = tmp_path / "origen"
    source.mkdir(exist_ok=True)
    path = source / "fake_model.onnx"
    path.write_bytes(content)
    asset = ModelAsset(
        name="fake",
        filename="fake_model.onnx",
        sha256=sha256_file(path),
        size_bytes=len(content),
        url="https://example.invalid/fake_model.onnx",
        purpose="pruebas",
    )
    return asset, source


def test_a_verified_local_copy_is_installed_without_downloading(tmp_path, monkeypatch):
    asset, source = make_asset(tmp_path)
    monkeypatch.setattr(
        assets_module, "search_paths", lambda: [models_dir(), source]
    )
    path, how = install_model(asset)
    assert how.startswith("copiado")
    assert path.parent == models_dir()
    verify_model(asset, path)


def test_installing_twice_is_a_no_op(tmp_path, monkeypatch):
    asset, source = make_asset(tmp_path)
    monkeypatch.setattr(assets_module, "search_paths", lambda: [models_dir(), source])
    install_model(asset)
    _, how = install_model(asset)
    assert how == "ya estaba instalado"


def test_a_tampered_model_is_refused_because_onnx_is_executable(tmp_path):
    asset, source = make_asset(tmp_path)
    (source / "fake_model.onnx").write_bytes(b"contenido distinto")
    with pytest.raises(ModelIntegrityError) as excinfo:
        verify_model(asset, source / "fake_model.onnx")
    assert "SHA-256" in excinfo.value.user_message


def test_a_corrupt_installed_copy_is_replaced(tmp_path, monkeypatch):
    asset, source = make_asset(tmp_path)
    monkeypatch.setattr(assets_module, "search_paths", lambda: [models_dir(), source])
    models_dir().mkdir(parents=True, exist_ok=True)
    (models_dir() / asset.filename).write_bytes(b"basura")
    path, how = install_model(asset)
    assert how.startswith("copiado")
    verify_model(asset, path)


def test_a_missing_model_says_exactly_how_to_get_it(tmp_path, monkeypatch):
    asset, _ = make_asset(tmp_path)
    monkeypatch.setattr(assets_module, "search_paths", lambda: [tmp_path / "vacio"])
    with pytest.raises(ModelNotFoundError) as excinfo:
        resolve_model(asset)
    assert "meetscribe models --download" in excinfo.value.user_message


def test_the_pinned_hashes_are_real_sha256_digests():
    for asset in REQUIRED_MODELS:
        assert len(asset.sha256) == 64
        assert set(asset.sha256) <= set("0123456789abcdef")
        assert asset.url.startswith("https://")
        assert asset.size_bytes > 0


def test_the_shipped_models_match_their_pinned_hashes_when_present():
    """If a developer already ran the spikes, the files on disk must be the exact
    ones the measurements were taken with."""
    for asset in REQUIRED_MODELS:
        path = find_model(asset)
        if path is not None:
            verify_model(asset, path)


def test_egress_error_names_the_local_alternative():
    error = EgressBlockedError("x")
    assert "ollama" in error.user_message
