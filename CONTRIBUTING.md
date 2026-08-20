# Contributing

## Getting set up

```bash
uv sync --extra dev
uv pip install -e .
pytest
```

The test suite needs no audio hardware, no model files and no network. If it does not pass on
a clean checkout, that is a bug worth reporting on its own.

## Conventions

These are not style preferences; each one exists because the alternative caused a concrete
problem.

**User-facing text is Spanish. Code, identifiers, comments, docstrings and documentation are
English.** Contributors read the code; users read the output. Mixing the two makes both worse.
A pull request that adds an English string to the CLI, or a Spanish identifier to a module,
will be asked to pick a side.

**Ports and adapters.** Every heavy dependency sits behind a `typing.Protocol` in
`core/ports.py`. Dependencies point inward: `core/` imports nothing from the project, adapters
import from `core/`, and an adapter never imports the protocol it satisfies — conformance is
structural, and `tests/test_core.py` asserts it. If you add an adapter, add it to that test.

**No silent failures.** Every error carries two messages: a technical one for the log, and a
Spanish `user_message` that says what to do next. See `errors.py`. An exception that reaches
the user without a `user_message` is an incomplete error path.

**Comments explain *why*.** What the code does is already in the code. A comment that
restates the next line is noise; a comment that records the measurement or the failure that
drove a decision is the most valuable thing in the file.

**New behaviour comes with a test that would have caught the bug.** Not a test that exercises
the happy path — one that fails before the fix. Several tests in this suite were written by
first breaking the code and confirming the test caught it.

## Formatting

`ruff check` is enforced. **`ruff format` is deliberately not part of the workflow.**

The layout here is hand-tuned: aligned section dividers, comment blocks wrapped to sit against
the code they explain, tables kept in visual columns. Reflowing all of it would cost more
readability than the consistency would buy. Lint rules are mechanical and enforced; layout is
reviewed by a human.

```bash
ruff check src tests tools benchmarks
```

Every exemption in `pyproject.toml` carries a comment explaining why it is there. If you need
a new one, explain it the same way.

## Running the real thing without a real meeting

```bash
meetscribe start --replay recording.wav --solo-remoto   # the full pipeline, on a file
python tools/panel_preview.py                           # the panel, on a scripted meeting
```

`panel_preview.py` runs the production server and event bus against a scripted session, so the
page you inspect is the page a meeting produces. It holds the microphone at digital silence on
purpose: "your microphone is producing no signal" is the most important thing the panel ever
says, and it should not be the least-exercised path.

## Benchmarks

Every performance number in `docs/` came from a script in `benchmarks/`. If you change
something that could move one of those numbers, re-run the relevant script and update the
document in the same pull request — a stale benchmark is worse than none, because it is
believed.

```bash
python benchmarks/bench_capture.py              # WASAPI loopback + mic, real devices
python benchmarks/bench_stt_throughput.py       # RTF per model size
python benchmarks/bench_stt_accuracy.py         # RTF and WER on speech with ground truth
python benchmarks/bench_diarization_2spk.py     # VAD + speaker embeddings, 2 speakers
python benchmarks/bench_diarization_3spk.py     # the same, 3 speakers (harder case)
```

The speech fixtures are committed so the numbers stay reproducible. They are deterministic
Windows SAPI output and can be regenerated:

```powershell
powershell -File benchmarks/make_fixtures_3spk.ps1
powershell -File benchmarks/make_fixtures_2spk.ps1
```

The ONNX models the diarization benchmarks need are not committed — they are downloaded and
hash-verified by `meetscribe models --download`, and a second unverified copy in the tree
would defeat the point of pinning them.

## Editing `install.ps1`

Two things about this file will break silently if you are not careful:

1. **It must keep its UTF-8 BOM.** PowerShell 5.1 — the version that ships with Windows —
   reads a `.ps1` without a BOM as ANSI. A single accented character then terminates a string
   and the whole script fails to parse. `.gitattributes` keeps the line endings CRLF, but the
   BOM is file content: if your editor strips it, the file is broken.
2. **Never let `$ErrorActionPreference = 'Stop'` see a native command's stderr.** `uv` writes
   progress to stderr; under `Stop` that aborts the script mid-install, leaving a runtime with
   no working command. Call native executables through the `Invoke-Native` / `Assert-Native`
   helpers, which check `$LASTEXITCODE` instead.

After any edit, verify both:

```powershell
powershell -NoProfile -Command "[void][System.Management.Automation.Language.Parser]::ParseFile('install.ps1', [ref]$null, [ref]$null); 'parse ok'"
```

## Pull requests

- One concern per pull request.
- `pytest` and `ruff check` pass.
- The commit message says *why*, not *what* — the diff already says what.
