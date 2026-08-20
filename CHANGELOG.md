# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-20

First complete release. Dual-track capture, CPU transcription, speaker separation, live panel,
minutes and exports all work end to end, verified by installing from a wheel on a clean prefix
and recording a session.

### Added

- **Dual-track WASAPI capture** — loopback and microphone as two independent tracks, never
  pre-mixed, with a per-track silent-device watchdog.
- **CPU-only transcription** — `faster-whisper` on CTranslate2, int8, one shared model
  instance behind a bounded queue, with a hallucination filter after the decoder.
- **Speaker separation** — Silero VAD plus NeMo TitaNet embeddings through `sherpa-onnx`,
  online clustering during the meeting and an offline refinement pass at the end. Speakers can
  be renamed retroactively across the whole transcript.
- **Live web panel** — loopback-only HTTP server, Server-Sent Events with a `Last-Event-ID`
  replay cursor, level meters, detected commitments. WCAG AA in light and dark themes,
  keyboard navigable, no build step and no external requests.
- **Structured minutes** — Claude or Ollama behind one port, returning a validated structure:
  summary, topics, decisions, action items with owner and due date, open questions.
- **Exports** — Markdown, plain text, JSON and SRT.
- **`meetscribe setup`** — guided first-run configuration, with `--defaults` for unattended
  installs. The unattended path selects no summarizer, so nothing the installer writes can
  send text off the machine.
- **`meetscribe doctor`** — verifies by doing: opens the audio endpoints and measures signal
  level, checks the model cache by hash, inspects data directory permissions and cloud-sync
  parentage, tests summarizer reachability.
- **`install.ps1`** — one-command install, in-place upgrade and uninstall for a target
  machine. No admin rights, no developer tools. Reuses a healthy environment, rebuilds a
  broken one, and preserves recordings across both.
- **Consent recording** — per-session, with `meetscribe anuncio` printing a sentence to read
  aloud.
- **Security posture** — 256-bit panel token compared in constant time, Host/Origin
  validation, CSRF header on state-changing requests, SHA-256 pinned ONNX models with no
  bypass, API key in Windows Credential Manager and omitted from every dump, log redaction
  filter installed before the first log call, ACL-hardened data directory, cloud-sync
  detection.
- Documentation: [`docs/architecture.md`](docs/architecture.md) for the decisions and the
  alternatives that were rejected, [`docs/benchmarks.md`](docs/benchmarks.md) for the
  measurements behind every number.

### Fixed

Found by running the application rather than by reading it:

- **HTTP request desynchronisation in the panel.** A rejected `POST` left its body unread, so
  on a keep-alive connection the server parsed that body as the start of the next request —
  one rejected request broke the following one. The body is now read before any check that can
  reject the request.
- **Cloud-sync false positive.** The data directory check used a substring match, so any path
  merely containing the word "OneDrive" was flagged. It now consults the Windows environment
  variables and compares whole path components.
- **`install.ps1` failed on stock Windows.** Without a UTF-8 BOM, PowerShell 5.1 read the file
  as ANSI; the first accented character terminated a string and the script would not parse.
- **The installer was not idempotent.** `uv venv` refuses an existing directory, so the
  documented upgrade path failed. It now reuses a healthy environment and rebuilds a broken
  one.
- **Reinstalling the same version silently did nothing**, so rebuilding a wheel without
  bumping the version deployed the old code.
- **Redirected installs aborted.** `$ErrorActionPreference = 'Stop'` combined with `uv`
  writing progress to stderr killed the script mid-install, leaving a runtime with no working
  command.
- **Level meters caused page-wide horizontal scrolling** below roughly 650 px.
- **Roughly fifty incorrect Spanish accents** in user-facing text, and a consent notice that
  referenced a command-line flag that does not exist.
- **Protocol drift** between `core/ports.py` and the real adapters, now asserted by a test.

### Notes

- Not yet validated against a live Teams call; functional validation is a synthetic two-voice
  replay plus `doctor` against real hardware. See
  [Status and limitations](README.md#status-and-limitations).
- `install.ps1` is unsigned.
- Whisper weights are not pinned by revision, unlike the ONNX models.

[1.0.0]: https://github.com/LittlePitza/meetscribe/releases/tag/v1.0.0
