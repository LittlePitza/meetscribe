# Architecture

Why MeetScribe is shaped the way it is. Every number quoted here was measured on the target
machine; the measurements themselves are in [`benchmarks.md`](benchmarks.md).

The organising idea: **a tool that records private conversations has to be boring in the
places where it could be clever.** No bot in the meeting, no cloud service, no event loop
bridging a blocking pipeline, no dependency that could not be justified twice over.

---

## 1. The shape

```
┌─ capture ─────────────┐   ┌─ segmentation ──┐   ┌─ recognition ────────────┐
│ WASAPI loopback ──────┼──▶│ VAD (silero)    │──▶│                          │
│   = the other people  │   │  per track      │   │  ONE shared transcriber  │
│                       │   │                 │   │  (faster-whisper, int8)  │
│ microphone ───────────┼──▶│ VAD (silero)    │──▶│            │             │
│   = you               │   └─────────────────┘   └────────────┼─────────────┘
└───────────────────────┘         ChunkQueue(30 s)   UtteranceQueue(32)
                                                                │
                          ┌─────────────────────────────────────▼──────────┐
                          │ diarize ▸ assemble ▸ hallucination filter      │
                          │ ▸ persist (SQLite) ▸ publish (event bus)       │
                          └─────────────────────────┬──────────────────────┘
                                                    │
                              ┌─────────────────────▼──────────────────────┐
                              │ SSE ▸ local panel      summarizer ▸ minutes│
                              └────────────────────────────────────────────┘
```

Six long-lived threads, a bounded queue at every seam, and one direction of flow.

## 2. Two tracks, never pre-mixed

The loopback endpoint carries everything the remote participants say; the microphone carries
you. They are captured separately and stay separate all the way to the database.

This is the decision the whole design rests on. Speaker attribution for half the transcript
becomes a property of the *wiring* rather than an inference: the mic track is you, with
certainty, and no clustering algorithm can be wrong about it. Only the loopback track — which
genuinely is several people mixed by Windows before we ever see it — needs diarization.

A single pre-mixed stream would have made every line a guess.

## 3. Ports and adapters

Every heavy dependency — WASAPI, CTranslate2, onnxruntime, SQLite, HTTP — sits behind a
`Protocol` in [`core/ports.py`](../src/meetscribe/core/ports.py). The pipeline is written
against the Protocols only.

Two consequences that pay for the indirection:

- **The whole orchestration is unit-testable.** 322 tests run with in-memory fakes: no audio
  hardware, no model files, no network, and an injected clock. The threading, the shutdown
  ordering, the backpressure policy — all exercised deterministically.
- **The dependency arrow points inward.** Protocols are structural, so an adapter never
  imports the port it satisfies. `WasapiSource` has no idea `AudioSource` exists.

The ports are kept honest by a conformance test: `runtime_checkable` `isinstance` assertions
against the *real* adapters. A port that omits a method the pipeline calls is worse than no
port at all, because a fake built from it type-checks and then fails at runtime.

## 4. Threads, not asyncio, not processes

The expensive stages all release the GIL in C++: CTranslate2 while decoding, onnxruntime
during VAD and embedding, PortAudio while blocked on a read. Threads therefore run genuinely
in parallel here, and the Python between them is trivial.

- **asyncio would buy nothing.** Every stage is a blocking C call, not an I/O wait. There is
  nothing to await.
- **Processes would cost a lot.** Audio buffers over a pipe, and a 1.6 GB model duplicated.

The web server is a plain threaded `http.server` for the same reason: with no event loop in
the process there is no event loop to starve, and the blocking-pipeline-to-async-server
bridge — the classic source of subtle deadlocks in exactly this kind of program — does not
exist to get wrong.

## 5. One transcriber, behind a queue

Measured: two `large-v3-turbo` instances would take ~3.2 GB and put two decoders on the same
eight cores. Meeting speech is turn-taking, so a single shared instance behind one queue is
both cheaper and faster in practice; the queue absorbs the moments when both tracks do speak
at once.

The whole pipeline sums to RTF ≈ 0.32 — a 3× margin.

| Stage | RTF |
|---|---:|
| STT (`large-v3-turbo`, int8, 8 threads) | 0.305 |
| VAD × 2 tracks | 0.008 |
| Speaker embedding | 0.009 |
| **Total** | **≈ 0.32** |

## 6. Why not streaming STT (LocalAgreement-2)

The obvious way to get low-latency transcription is to run the decoder on a sliding window and
emit the prefix that two successive runs agree on. MeetScribe does not do this. It cuts audio
at VAD boundaries and transcribes each complete utterance once.

The reasons, in order of weight:

1. **Whisper is not a streaming model.** It was trained on 30-second windows and behaves badly
   on partial ones — it hallucinates continuations, and its timestamps drift. Feeding it a
   natural pause is feeding it the input it was built for.
2. **The latency problem it solves is not the problem here.** At 3.3× real time an utterance is
   transcribed in under a third of the time it took to say. The interim-draft mechanism
   (`interim_after_s`) covers the one case that matters — a speaker who has been going for
   more than eight seconds — by publishing provisional text that the final replaces.
3. **It multiplies the decoder cost** by the overlap factor, spending the 3× headroom that
   currently absorbs both tracks talking at once.
4. **It makes correctness much harder to test.** Agreement-based prefix emission has to be
   right about re-anchoring after every drop, gap and device reconnect. The current design has
   one rule instead: an utterance is transcribed when it is complete.

The cost is that a very long uninterrupted monologue appears in fewer, larger chunks. That was
judged the better trade.

## 7. Drafts, and what gets sacrificed first

The `UtteranceQueue` between segmentation and recognition is where backpressure bites, because
recognition is the only stage that can fall behind — a screen share, a build, thermal
throttling.

Its policy is explicit:

- Enqueueing a **final** *removes* any pending **draft** with the same key. The draft's audio
  is superseded; spending the decoder on it while behind is precisely wrong.
- When the queue is full, the oldest **draft** goes first. Nothing permanent is lost.
- Only when no drafts remain is a **final** dropped — and that is counted and surfaced as a
  warning, because it is a real hole in the transcript.

Unbounded growth was the alternative, which turns a slow machine into a two-hour memory leak.

## 8. Hallucination is a first-class failure mode

Whisper's decoder is autoregressive and was trained on subtitle corpora. Given audio with
little speech — a hold tone, music, a cough — it emits what subtitle files usually contain. In
Spanish that means *"Subtítulos realizados por la comunidad de Amara.org"*.

Writing that into a meeting transcript is worse than dropping the segment, because the user
cannot tell it from a real line and the summarizer downstream will treat it as content.

Four independent filters, each catching a different failure: `no_speech_prob` (the model's own
estimate), `avg_logprob` (confidence), `compression_ratio` (repetition loops compress
extremely well), and a blocklist plus repetition detection on the text. Every rejection is
logged with its reason — a filter silently eating real speech would be its own kind of bug.

## 9. Diarization: online, then corrected

The loopback track is clustered online so labels appear during the meeting, then **refined
offline at the end** over every stored embedding, and the corrections applied retroactively.

Online clustering has to decide with only the past; the refinement pass sees the whole meeting
and can merge two clusters that turned out to be one person. Only the segments that actually
moved are republished — resending the whole transcript would be correct and would also
overflow every subscriber's queue on a long meeting, disconnecting the panel at the worst
possible moment.

The operating threshold is cosine 0.60, chosen mid-gap between a measured same-speaker minimum
of 0.868 and a different-speaker maximum of 0.434. That gap is wide, so the threshold sits in
a large safe window rather than on a knife edge — and it narrows on real, compressed,
crosstalk-heavy audio, which is why it is configurable and why diarization degrades to a
single `Remoto` label rather than failing the session.

## 10. Shutdown is a drain, not an abort

`Ctrl-C` mid-meeting must not lose the last sentence. The sequence is ordered and every step
has a deadline:

> stop capture → flush resampler tails → close chunk queues → segmenters flush their VADs →
> close the utterance queue → STT worker drains the backlog → refine diarization → drop
> unconfirmed drafts → summarise → close

Exceeding a timeout logs and moves on rather than hanging the process. A second `Ctrl-C`
exits immediately, because someone pressing it twice means it now.

Segments are committed per utterance, so a crash at minute 40 of a 60-minute meeting loses at
most the utterance in flight. A session the process never finished is marked `FAILED` at next
startup rather than being left `RUNNING` forever, looking like a live session that stopped
producing lines.

## 11. Security posture

The threat model is "a tool that records private conversations". These are properties, not
settings — the details are in the [README](../README.md#privacy-and-security); what follows is
why each exists.

| Control | The failure it prevents |
|---|---|
| Data in `%LOCALAPPDATA%`, never the repo | The repo is under OneDrive. Writing transcripts beside the source uploads every meeting, silently, as a side effect of where you cloned |
| ACL hardening on creation | `%LOCALAPPDATA%` is readable by other administrators by default |
| Audio discarded by default | The audio is the most sensitive artifact and the transcript is the one worth keeping |
| Loopback-only bind, refused otherwise | A live transcript on a LAN interface is a leak, not a preference |
| 256-bit token, constant-time compare | Loopback stops the network; the token stops another *account on the same machine* |
| Host + Origin validation | DNS rebinding |
| Token required in a custom header for POSTs | Forces a CORS preflight this server never answers, so form-style CSRF is structurally impossible |
| Body read before any rejection | An undrained body on a keep-alive connection is parsed as the next request line — one denied POST would break the following legitimate one |
| Pinned SHA-256 on model files | An ONNX file is a program the runtime executes; a mismatch means corruption or a compromised mirror, and there is no benign third case |
| Secret excluded from every dump | Not masked — *absent*, so no code path can format it into output |
| `--offline` refuses, never downgrades | A privacy switch that silently degrades is not a privacy switch |

Cloud-sync detection deserves a note, because it is the one check that has to be right in
*both* directions. It asks the environment first (`OneDrive`, `OneDriveCommercial`,
`OneDriveConsumer`), then matches whole path *components*. A substring test — the obvious
implementation — reports a false positive for any unrelated directory whose name merely
contains the word, and telling someone their data directory is unsafe when it is not trains
them to ignore the warning that matters.

## 12. Where each decision lives

| Concern | Module |
|---|---|
| Domain types, no I/O | `core/models.py`, `core/timeline.py` |
| The seams | `core/ports.py` |
| Device discovery and reconnection | `capture/devices.py`, `capture/track.py` |
| Backend adapters | `capture/sources.py` |
| Segmentation and drift handling | `stt/vad.py` |
| Recognition | `stt/transcriber.py` |
| Invented-speech defences | `stt/hallucination.py` |
| Online clustering, offline refinement | `diarize/online.py` |
| Topology, threading, shutdown | `pipeline/session.py` |
| Backpressure policy | `pipeline/queues.py` |
| Minutes, two adapters, one port | `summarize/` |
| Paths, ACLs, secrets, redaction, retention, consent | `security/` |
| Loopback HTTP, SSE, the panel | `web/` |
