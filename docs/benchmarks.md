# Spike findings — measured on the target machine

Everything below was **measured**, not estimated. Every design decision in this project
traces back to a number on this page. Re-run the spikes in `benchmarks/` to reproduce.

**Target machine:** Windows 11 Home 26200 · AMD Ryzen AI 7 350 (8C/16T, Zen5) ·
Radeon 860M iGPU (**no CUDA**) · 15.2 GB RAM · Python 3.12.14

---

## 1. Audio capture — WASAPI loopback works

`benchmarks/bench_capture.py`

| Endpoint | Device | Format |
|---|---|---|
| Render (default) | `Altavoces (Realtek(R) Audio)` | 48 kHz, 2 ch |
| **Loopback** | `[10] Altavoces (Realtek(R) Audio) [Loopback]` | 48 kHz, 2 ch |
| Capture (mic) | `Microphone Array (AMD Audio Device)` | 48 kHz, 2 ch |

Captured 3 s from the loopback endpoint: **−31.6 dBFS, peak 0.078** — real signal, not a
stream that merely opened. `soundcard` independently confirmed loopback works, so it is a
viable fallback backend behind the same port.

> **Open issue:** the microphone captured **digital silence** (−240 dBFS, peak 0.0000).
> The stream opened without error. This is exactly the failure mode that silently produces
> an empty half of the transcript, so the capture stage ships a **silent-track watchdog**
> and `meetscribe doctor` checks it explicitly. Likely cause is Windows microphone privacy
> settings or a muted array — to be confirmed with the user before first real use.

**Decision:** `PyAudioWPatch` as the primary capture backend, `soundcard` as fallback,
both behind an `AudioSource` port so tests inject a WAV/synthetic source instead.

---

## 2. Speech-to-text — CPU is comfortably fast enough

`benchmarks/bench_stt_accuracy.py`

### Method note — the first measurement was wrong

The initial benchmark ran against synthetic babble and reported RTF 0.009–0.045.
Whisper classified that audio as non-speech and returned **`segments: 0`**, so the
autoregressive decoder never ran; the numbers measured the encoder alone and were
meaningless as a capacity estimate.

The benchmark was redone against **real Spanish speech** generated locally with the
Windows SAPI engine (`benchmarks/make_fixtures_3spk.ps1`), with known ground truth so both
throughput *and* accuracy are measured. Fixture: 71.3 s, 10 turns, 151 reference words.

### Results (int8, 8 CPU threads, `beam_size=5`, VAD on)

| Model | RTF | Speed | WER | Verdict |
|---|---:|---:|---:|---|
| tiny | 0.024 | 42.5× | 8.6 % | fastest, least accurate |
| base | 0.045 | 22.3× | 7.3 % | |
| **small** | **0.124** | **8.1×** | **6.0 %** | **throughput choice** |
| medium | 0.348 | 2.9× | 7.3 % | **dominated — never use** |
| **large-v3-turbo** | **0.305** | **3.3×** | **4.6 %** | **accuracy choice, default** |

Two findings worth keeping:

1. **`medium` is strictly dominated.** It is slower than `large-v3-turbo` *and* less
   accurate (7.3 % vs 4.6 % WER). It is removed from the offered model list entirely.
2. **Even the largest viable model runs at 3.3× real time.** The CPU-only constraint that
   looked like the project's main risk is not a constraint at all.

> **WER caveat, stated plainly:** SAPI speech is clean, articulate, noise-free and
> single-accent. Real meeting audio — Teams' lossy codec, background noise, overlapping
> speech, varied accents, cheap laptop mics — will be **substantially worse**. These
> figures are a *relative* ranking between models, not an absolute accuracy promise.

---

## 3. VAD and diarization — sherpa-onnx covers both, cheaply

`benchmarks/bench_diarization_3spk.py`, `benchmarks/bench_diarization_2spk.py`

`sherpa-onnx` (**2.2 MB wheel, no PyTorch, no HuggingFace gated licences**) provides
`VoiceActivityDetector`, `SpeakerEmbeddingExtractor`, `SpeakerEmbeddingManager`,
`OfflineSpeakerDiarization` and `FastClustering` — the entire VAD + diarization stack
behind one lightweight ONNX dependency.

| Component | Model | Size | RTF |
|---|---|---:|---:|
| VAD | `silero_vad.onnx` | 0.6 MB | **0.0040** (250×) |
| Speaker embedding | `nemo_en_titanet_small.onnx` | 38 MB | **0.0091** (110×) |

### Speaker separation

The first attempt used **one SAPI voice with varied prosody** as three speakers. TitaNet
correctly refused to split it — margin +0.040, everything into one cluster. That is the
embedder working as designed (it models the vocal tract, not the pitch), and it made the
fixture invalid rather than the model.

Redone with **two genuinely different voices** (Helena es-ES, Zira en-US):

| Metric | Value |
|---|---|
| Same-speaker cosine | mean **0.909**, min 0.868 |
| Different-speaker cosine | mean **0.384**, max 0.434 |
| Separation margin | **+0.524** |
| Distribution overlap | **none** — cleanly separable |
| Clustering purity | **100 %** at every threshold from 0.25 to 0.55 |

The gap between 0.434 and 0.868 is wide, so the operating threshold sits in a large safe
window rather than on a knife edge. **Chosen starting point: cosine 0.60** (distance 0.40),
mid-gap, to be recalibrated against real meeting audio.

> Two clean TTS voices are an easy case. Real speakers on a compressed call with crosstalk
> will narrow that margin. The threshold is therefore configurable, and diarization degrades
> to a single `Remoto` label rather than failing the session.

---

## 4. CPU budget — the whole pipeline fits with room to spare

Summing the measured real-time factors for the worst-case model choice:

| Stage | RTF |
|---|---:|
| STT — one shared `large-v3-turbo` instance | 0.305 |
| VAD × 2 tracks | 0.008 |
| Speaker embedding | 0.009 |
| **Total** | **≈ 0.32** |

**Key architectural consequence:** run **one shared transcriber behind a work queue**, not
one model instance per track. Two instances would double both RAM (~1.6 GB each) and thread
contention for no benefit, because meeting speech is largely turn-taking — the mic and
loopback tracks rarely need the decoder at the same instant. The queue absorbs the overlap
that does occur, and the 3× headroom covers it.

---

## 5. Test infrastructure gained for free

`benchmarks/make_fixtures_3spk.ps1` and `make_fixtures_2spk.ps1` generate deterministic Spanish speech via
the Windows SAPI engine — **no network, no licensing, no recorded humans, byte-identical on
every run**. This becomes the project's fixture generator, giving the test suite real speech
with exact ground-truth text and speaker labels for WER and diarization regression tests.
