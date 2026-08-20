# Security Policy

## Reporting a vulnerability

Please report security issues **privately** through GitHub's
[private vulnerability reporting](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
on this repository, rather than opening a public issue.

Include what you did, what happened, and what you expected. A proof of concept helps but is
not required.

## What this tool guards

MeetScribe records private conversations, so the interesting attack surface is not remote code
execution — it is **anything that causes a transcript to reach somewhere it should not.** The
following are treated as security properties, and a bug in any of them is a vulnerability
rather than a defect:

| Property | Where it is enforced |
|---|---|
| The panel binds to loopback only; a non-loopback bind requires explicit configuration | `web/server.py`, `config.py` |
| Every data endpoint requires a 256-bit session token, compared in constant time | `web/server.py` |
| `Host` and `Origin` are validated (DNS rebinding); no CORS header is ever emitted | `web/server.py` |
| State-changing requests require the token in a custom header (CSRF) | `web/server.py` |
| Static file serving cannot escape the panel's asset directory | `web/server.py` |
| `--offline` refuses a remote summarizer rather than downgrading silently | `summarize/factory.py` |
| The API key is never written to `config.yaml`, and is omitted — not masked — from every dump | `security/secrets.py` |
| Secrets are stripped from log records by a filter installed before the first log call | `security/redact.py` |
| Model files are verified against a pinned SHA-256, with no bypass | `assets.py` |
| The data directory is never inside a cloud-synced folder | `security/paths.py` |
| The data directory's ACL is restricted on creation | `security/paths.py` |

Each of these has tests in `tests/test_security.py` and `tests/test_bus_web.py`.

## Known limitations

These are documented rather than hidden, and are not what to report:

- **`install.ps1` is unsigned.** SmartScreen will warn, and `-ExecutionPolicy Bypass` is
  needed to run it. Verify the file before running it on a machine you care about.
- **Whisper weights are not pinned by revision.** The ONNX models (VAD, speaker embeddings)
  are verified against a pinned SHA-256. The Whisper model is downloaded by `faster-whisper`
  from its upstream repository and resolves to whatever that repository currently serves.
- **ACL hardening is best-effort.** Where `icacls` is unavailable the application still runs
  and reports that hardening did not happen, rather than refusing to start.
- **Dependencies are resolved at install time**, not from a lock file, so two machines
  installed at different times may not receive identical transitive dependency versions.
- **The panel's session token appears in the URL.** It is printed once to the terminal and
  never logged, but it will be present in the browser's history for that session.
- **Choosing the Claude summarizer sends the transcript text to Anthropic.** This is the
  documented purpose of that setting, is never the installer's default, and is refused
  outright under `--offline`.

## Scope

In scope: anything in `src/`, `install.ps1`, and the packaging in `pyproject.toml`.

Out of scope: vulnerabilities in third-party dependencies (report those upstream, though
telling us is appreciated so the pin can move), and the security of a model file a user
deliberately placed in the cache themselves.
