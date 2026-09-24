# Implementation Plan: Code-Switching Language Detection and Speaker Diarization

**Track ID:** codeswitch-diarization_20260922
**Spec:** [spec.md](./spec.md)
**Created:** 2026-09-22
**Status:** [ ] Not Started

## Overview

Build the measurement before the thing being measured. Phase 1 turns the spike's throwaway eval
scripts into a committed harness with a recorded baseline, because without it no later change can
be shown to help. Phase 2 fixes the chunking, which the prototype got wrong in two ways — too
coarse, and dropping speech. Phase 3 puts per-chunk language detection behind backend config and
proves it against the baseline. Phase 4 adds diarization, gated on the ground-truth question
settled in Phase 1. Phase 5 documents and tunes.

Nothing in this track touches the pipeline, the schema, the interface, or the manifest. Both
fields already exist; this fills them. If a change here appears to need a contract change, that is
a signal to stop and reconsider the approach, per the approval rules in `workflow.md`.

## Phase 1: Eval Harness and Baseline

Make the improvement measurable. The spike's numbers live in a scratch directory and are not
reproducible; this phase makes them so.

### Tasks

- [x] Task 1.1: Add an `eval` optional extra (`jiwer`, `pyarrow`) — never a core or backend dependency
- [x] Task 1.2: Write the dataset fetcher: download the `mixed` config on demand into a gitignored cache, verify it, and never commit audio
- [x] Task 1.3: Implement reference cleaning for CHAT markup — `(.)` pauses, `<>` retracing, `+` continuations, `mani(cure)` partials — with tests over hand-written examples
- [x] Task 1.4: Implement scoring: WER via `jiwer`, plus a language-mix measure reporting transcribed Spanish share against the reference share
- [x] Task 1.5: Write the eval runner — takes a backend name and config, runs the `mixed` subset, writes a JSON report
- [x] Task 1.6: Record the per-file baseline for all 5 recordings and commit the report as the number to beat
- [x] Task 1.7: **Settle the diarization ground-truth question** (see Risks in the spec): TalkBank credentials, another labelled corpus, or qualitative-only. Record the decision here before Phase 4 is planned in detail

### Verification

- [x] `pytest` passes; reference cleaning is covered by tests (`tests/test_evaluation_text.py`)
- [x] The eval runs end to end and reproduces a WER near 0.44 for `zeledon4` under the current default
  — *superseded:* 0.44 was one unseeded draw. Seeded, `zeledon4` scores 0.375 and reproduces
  byte-for-byte across runs (re-verified 2026-09-23 on all five recordings)
- [x] The baseline report is committed, and Task 1.7's decision is written into this plan

### Task 1.7 decision — diarization ground truth

**Decided 2026-09-22: two corpora, two questions.**

What was checked:

| Source | Speaker labels | Code-switched | Reachable |
| ------ | -------------- | ------------- | --------- |
| `BrunoHays/Bangor-Miami-…` (this track's corpus) | No — CHAT speaker tiers stripped | Yes | Yes |
| TalkBank (original Bangor source) | Yes, plus word-level language IDs | Yes | **No — authentication modal** |
| `ksheth2/miamibangor` | — | — | Empty stub, README only |
| `diarizers-community/voxconverse`, `/ami` | Yes | No — English only | Yes, ungated |

**Decision:** use `diarizers-community/voxconverse` to measure DER and prove the diarizer works at
all, which unblocks Phase 4 without an external dependency. Pursue TalkBank credentials in
parallel for the integrated bilingual evaluation.

**Consequence for Phase 4:** diarization quality is measured on English conversation, not on this
project's actual audio. Until TalkBank lands, no claim is made about diarization accuracy on
code-switched speech, and the backend doc page must say so.

**Bonus if TalkBank arrives:** its word-level language annotations would replace the aggregate
`spanish_share` this track currently scores against, turning the language-mix measure into true
per-segment language accuracy.

## Phase 2: Chunking

Fix what the prototype got wrong. This phase changes no behaviour visible to a user — it produces
correct chunk boundaries and proves they lose nothing.

### Tasks

- [x] Task 2.1: Implement VAD-based chunk grouping with a configurable target length, replacing the prototype's logic which produced ~68 s chunks against a 28 s target
- [x] Task 2.2: Write tests asserting chunk lengths stay within the target and boundaries land on speech gaps
- [x] Task 2.3: **Fix the content-loss regression.** The prototype produced 3683 words against a 4269-word reference, with 666 deletions; measure what VAD discards and close the gap
- [x] Task 2.4: Add a coverage assertion to the eval — transcribed audio duration against total speech duration — so a future change cannot buy WER by dropping speech
- [x] Task 2.5: Fix decoding reproducibility (unplanned; see deviations)
- [x] Task 2.6: Fix silently dropped backend config (unplanned; see deviations)

### Verification

- [x] Chunk lengths respect the configured target (`tests/test_chunking.py`)
- [x] Transcribing chunk-by-chunk under the file-level language scores no worse than the baseline; this isolates chunking from detection
  — measured 2026-09-23 as chunked mode with `language_threshold=1.01`, so every chunk keeps the
  file-level language. With span merging that tests the current design's chunking — VAD
  grouping, no VAD inside a span — rather than per-chunk decoding, which no longer exists. Mean
  WER −0.010, retained +0.017. On the four recordings whose file-level language matched the
  baseline's: −0.035, −0.020, +0.002, +0.020, all better or within the 0.022 seed noise floor
  (Task 5.1). `zeledon4` is excluded: restricting detection to `{en, es}` flipped its file-level
  language from `es` to `en`, so it does not isolate chunking
- [x] Coverage assertion in place and passing — now `retained`, per Task 5.8

### Deviations

**2026-09-22 — content loss was the chunking bug, not VAD.** Task 2.3 assumed VAD was discarding
speech. With chunk length bounded, speech coverage measured 0.97 and 0.99 on the two recordings
checked, so VAD discards very little. The prototype's deletions came from single language
decisions spanning minutes of bilingual audio. Task 2.4's guard is satisfied by the `coverage`
field added to `Score` in Phase 1, which is asserted per recording in `Comparison`.

**2026-09-22 — two unplanned fixes to the shipped backend (Tasks 2.5, 2.6).** Phase 1's harness
found both; neither was visible to the existing test suite.

*Reproducibility.* faster-whisper samples above temperature 0.0 and the backend never pinned a
seed, so the same recording gave WER 0.409 vs 0.441 across runs — a 0.127 spread, larger than any
improvement this track could plausibly show. A first attempt defaulted to greedy decoding; it was
measured and reverted, because removing the ladder cost 0.185 WER and 15 points of coverage, and
no configuration between the two extremes beat either. The ladder earns its place: 14.5% of
segments fall back and some need temperature 1.0 to escape repetition loops. Seeding CTranslate2's
RNG gives accuracy and reproducibility together, verified byte-identical through the full pipeline.

The default seed of 0 is deliberately untuned. Seeds 42 and 7 scored 0.259 and 0.394 WER on the
same recording; picking the better one would fit noise and inflate every subsequent measurement.

*Silently dropped config.* The backend accepted any keyword, recorded it in the manifest, and
forwarded only those `transcribe()` named inline. Two runs configured differently produced
byte-identical output while the manifest claimed the option was applied. Decode options are now
derived from the config so recording and forwarding cannot diverge, unknown options are rejected,
and a test asserts every recorded option reaches the decoder.

**Consequence:** the first committed baseline was invalid — unseeded, and taken at an unknown
point in a 0.127 spread. It has been replaced.

## Phase 3: Per-Chunk Language Detection

The measured win. Opt-in, restricted to `{en, es}`.

### Tasks

- [x] Task 3.1: Implement per-chunk detection taking the best of `{en, es}` from `all_language_probs`, not the global argmax
- [x] Task 3.2: Implement the confidence threshold and its fallback to the file-level language; choose the value from the eval rather than by intuition
- [x] Task 3.3: Add `language_detection` backend config (`file` default, `chunked` opt-in) and the allowed-language set as config
- [x] Task 3.4: Populate `spoken_language` per segment from its chunk's detected language
- [x] Task 3.5: Record the detection mode, threshold, and allowed languages in the effective config, so a run's manifest explains its own output
- [x] Task 3.6: Write tests — detection restricted to the allowed set, threshold fallback, and `spoken_language` varying within one transcript
- [x] Task 3.7: Run the eval and compare against the Phase 1 baseline

### Verification

- [x] `pytest -k conformance` still passes in both modes — *corrected 2026-09-23:* this was ticked
  when the suite only ran each backend's defaults. It now runs chunked mode explicitly, and passes
- [ ] **Not met:** WER improves on the baseline and Spanish share lands within 0.05 of the reference
- [x] Coverage assertion still passing — the win is not bought with deletions
- [x] The default path is byte-for-byte unchanged when `language_detection=file`

### Phase 3 result

Measured against the reproducible baseline, span-merged:

| recording | ΔWER | Δcoverage | mix error base → chunked |
| --------- | ---- | --------- | ------------------------ |
| herring17 | −0.052 | +0.019 | 0.341 → 0.083 |
| maria20   | −0.156 | +0.076 | 0.348 → 0.092 |
| sastre8   | +0.081 | −0.055 | 0.013 → 0.037 |
| zeledon14 | +0.009 | −0.011 | 0.001 → 0.006 |
| zeledon4  | +0.030 | −0.059 | 0.118 → 0.034 |

Mean: WER −0.018, coverage −0.006, language-mix error −0.114.

**All five recordings now report both languages**, where the baseline reported one for each.
Language-mix error improved on every recording. WER improved on two.

**The acceptance criterion "WER improves across all five" is not met, and will not be.** Chunked
detection helps where file-level detection guessed wrong (`herring17`, `maria20`: mix errors of
0.34) and costs where it guessed right (`sastre8`, `zeledon14`: mix errors of 0.013 and 0.001 —
nothing to fix, only overhead to add). That shape is inherent, not a tuning failure.

Two variants were measured. Decoding each chunk separately scored mean ΔWER −0.027 but regressed
coverage by 0.040, because a word split across two decodes is transcribed badly in both. Merging
consecutive chunks that agree on a language into one decode fixed coverage (−0.006) and gave back
some WER (−0.018). Neither dominates; span merging was kept because a coverage regression is the
failure mode the eval exists to catch.

**Consequence for Task 5.6:** `chunked` stays opt-in. It is the right setting for genuinely
bilingual audio and a net loss for monolingual audio, and the backend cannot know which it has
before transcribing.

## Phase 4: Speaker Diarization

Scope depends on Task 1.7. If no ground truth is available, this phase ships the capability with
qualitative evaluation only, and says so plainly in the doc page.

### Tasks

- [x] Task 4.1: Select a diarization model and record why, including its licence and gating
- [x] Task 4.2: Add it as its own optional extra, loaded only when diarization is enabled
- [x] Task 4.3: Implement requirement checks — model availability, credentials, device — reported up front like every other backend requirement
- [x] Task 4.4: Assign speaker labels to segments, stable within a transcript
- [x] Task 4.5: Make `capabilities.diarization` reflect whether it is actually active, rather than what the engine could do
- [x] Task 4.6: Record the diarization model and its version in the manifest, so a transcript's speaker labels are traceable
- [x] Task 4.7: Evaluate against whatever Task 1.7 settled on, and write the result down — **done 2026-09-23 against TalkBank, see below**
  - [x] 4.7a: CHAT loader — timed utterances, speaker, per-utterance language from `@Languages`, `@s` and `[- lang]`; choose the right file per recording
  - [x] 4.7b: DER scoring via `pyannote.metrics`, 0.25 s collar, excluding regions that hold untimed reference speech
  - [x] 4.7c: Per-segment language accuracy — time-weighted confusion against reference utterances, mixed speech reported separately
  - [x] 4.7d: Harness option `--talkbank`, results in the same JSON reports; docs with access, licence and citation
  - [x] 4.7e: Run file and chunked with diarization on; record DER and language accuracy

### Task 4.1 decision — diarization model

**Decided 2026-09-23: `pyannote/speaker-diarization-3.1`.**

Every full diarization pipeline available is either gated or non-commercial; the ungated
permissively-licensed options are speaker-embedding models that would need a segmentation
front-end and clustering built around them.

| Model | Gated | Licence | Commercial use |
| ----- | ----- | ------- | -------------- |
| `pyannote/speaker-diarization-3.1` | Yes, accept conditions + token | MIT | Yes |
| `soniqo/Sortformer-Diarization-4spk-ONNX` | No | CC-BY-NC-4.0, inherited from `nvidia/diar_streaming_sortformer_4spk-v2.1` | **No** |
| `speechbrain/spkrec-ecapa-voxceleb` | No | Apache-2.0 | Yes, embeddings only |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | No | CC-BY-4.0 | Yes, embeddings only |

Sortformer was the tempting option — ungated, good quality, and it runs on the `onnxruntime`
already installed — but it inherits NVIDIA's non-commercial licence, and `product.md` describes
internal team workflows. That is commercial use.

**Revised during implementation:** the pinned `pyannote.audio<4.0` resolved to 3.4.0, which
references `torchaudio.AudioMetaData` — removed in torchaudio 2.11. Upgrading to `>=4.0,<5.0`
fixes it, and 4.x ships `pyannote/speaker-diarization-community-1` as its recommended pipeline.
That is now the default: it bundles its embedding and scoring weights, so only its own gate needs
accepting, where `speaker-diarization-3.1` references its segmentation model externally and needs
two. The required gates are therefore per model rather than a fixed list.

**Consequence:** the model cannot be fetched until someone accepts the conditions at
`huggingface.co/pyannote/speaker-diarization-community-1` and sets `HF_TOKEN`. Everything else in
this phase is built and tested against that; the requirement check names the pages for whichever
pipeline is configured.

### Task 4.7 — reopened 2026-09-23 against TalkBank

TalkBank access was obtained (option 1 below). The original CHAT transcripts of the Bangor Miami
corpus carry speaker tiers, per-utterance timestamps and word-level language marks, and their
timestamps align with the Hugging Face audio already in use: matching transcript words against
our own segments peaks at a 0 s offset on all five recordings (0.56–0.84 of words), falling to
about 0.1 at ±30 s. So diarization and per-segment language can be scored on this project's own
bilingual audio, and VoxConverse is no longer needed.

**Licence, decided by the user:** TalkBank data is CC BY-NC-SA 3.0; its ground rules keep
downloaded data on the local device, forbid redistribution, and require citing Deuchar & Parafita
Couto (2011), doi:10.21415/T5J01D. Proceeding was the user's decision. The data lives only in
`~/.cache/audio-transcriber/talkbank/`, is never committed, and only aggregate scores are.

### Task 4.7 result — TalkBank

Both committed reports (`eval/baseline.json`, `eval/chunked.json`) now carry TalkBank scores. They
were run with `diarization=true`. Their text scores are identical to the runs without it, as they
should be: diarization changes no text.

**Speakers.** Speaker confusion, the share of reference speech labelled with the wrong speaker:

| recording | file | chunked | DER (chunked) |
| --------- | ---- | ------- | ------------- |
| herring17 | 0.038 | 0.037 | 0.186 |
| maria20   | — | — | — |
| sastre8   | 0.086 | 0.083 | 0.218 |
| zeledon14 | 0.056 | 0.041 | 0.208 |
| zeledon4  | 0.097 | 0.060 | 0.383 |

About 94% of labelled speech goes to the right speaker. DER is 19–38%, mostly `missed` speech:
TalkBank's utterance times span pauses that Whisper's segments do not, so DER measures
segmentation as much as speaker attribution. `maria20` cannot be scored: its reference times only
one of its two speakers. That first read as 46% confusion, which led to the scorability rule.

**Languages.** Per-segment language accuracy, by time, against TalkBank's word-level annotation:

| recording | file | chunked | chunked en / es recall | mixed speech |
| --------- | ---- | ------- | ---------------------- | ------------ |
| herring17 | 0.629 | 0.778 | 0.88 / 0.60 | 9% |
| maria20   | 0.732 | 0.817 | 0.84 / 0.74 | 24% |
| sastre8   | 0.662 | 0.762 | 0.89 / 0.50 | 26% |
| zeledon14 | 0.872 | 0.885 | 0.24 / 0.98 | 21% |
| zeledon4  | 0.436 | 0.859 | 0.90 / 0.80 | 12% |

- **Chunked labels segments more correctly on all five**, mean 0.666 → 0.820. That includes
  `sastre8`, whose WER got worse: its loss is in decoding, not labelling. This is the first
  measure on which chunked mode improves every recording.
- **Detection leans toward the majority language.** Minority-language recall is 0.50–0.80 on
  English-base recordings, and 0.24 for English on the Spanish-base `zeledon14`.
- **18% of speech is in mixed utterances** on average (9–26%), which no chunk-level label can get
  right. That replaces the 5.7% function-word estimate from Task 5.5, which undercounted.

**How the reports were produced.** Both runs used the code as of `f3fbb98`. The scorability rule
(`7c4de2c`) landed while the chunked run was in progress, so both reports' TalkBank sections were
recomputed from the saved segment files with the committed code. The scores are pure functions of
transcript and reference, so this is exact. Transcripts were checked byte-identical to the
committed runs.

### Task 4.7 — deferred 2026-09-23 (superseded above)

**Not done, and deliberately so.** Task 1.7 settled on `diarizers-community/voxconverse` for DER,
because the Bangor Miami dataset used everywhere else in this track has no speaker labels: the
published version strips CHAT markup including the speaker tiers, so there is nothing to score
against. VoxConverse is a different corpus that answers a different question.

| | Corpus | Measures | Describes this project's audio |
| --- | ------ | -------- | ------------------------------ |
| Phases 1–3 | Bangor Miami | WER, coverage, language mix | Yes — Spanish-English code-switching |
| Task 4.7 | VoxConverse | DER | No — English-only conversation |

A DER figure on English meeting recordings is published, unactionable, and does not describe
bilingual speech. It carries no decision, and the harness to produce it is roughly a day's work.

**What was verified instead:** diarization runs end to end on real audio and populates `speaker`
on every segment, with the model and library version recorded in the manifest. That establishes
the feature works, not how well.

**What is therefore unknown:** diarization accuracy, on any audio. No claim is made about it, and
the backend doc page must say so rather than cite pyannote's published benchmarks — which describe
`speaker-diarization-3.1`, not the `community-1` pipeline this uses.

**To close it, in order of value:**

1. **TalkBank credentials** (free, researcher registration). Gets speaker tiers *and* word-level
   language IDs on the recordings already in use, upgrading both evaluations at once. The real fix.
2. **Qualitative check on our own audio.** Twenty minutes, answers "is this usable for us", which
   VoxConverse does not.
3. **VoxConverse DER.** Regression protection so a future change cannot silently degrade
   diarization. Worth having once something depends on it.

### Verification

- [x] `speaker` is populated and stable within a transcript when enabled, `null` when not — tests, plus
  the 2026-09-23 end-to-end run (two labels over 554 segments; `null` only where no turn overlaps)
- [x] Disabled by default; nothing about the Phase 3 output changes when it is off — final `chunked`
  report differs from Phase 3's only by the one segment Task 5.7 drops
- [x] `pytest -k conformance` passes with diarization both on and off — four configurations, 2026-09-23

> **Phase 4 closes with Task 4.7 done against TalkBank.** Speaker labels are measured on four
> two-speaker recordings of this project's own kind of audio; see the Task 4.7 result.

## Phase 5: Tuning and Documentation

### Tasks

- [x] Task 5.1: Tune chunk length and confidence threshold against the eval; record the chosen values and what they cost
- [x] Task 5.2: Measure and document throughput for each mode — chunked detection adds a pass, diarization adds a model
- [x] Task 5.3: Update `docs/backends/faster-whisper.md` with the new options, their cost, and their limits
- [x] Task 5.4: Document the eval harness: how to run it, what it measures, and how to read the report
- [x] Task 5.5: Record in the spec's known limits that intra-sentential switching is not expressible at chunk granularity
- [x] Task 5.6: Decide whether `chunked` should become the default, with the eval as the argument either way
- [x] Task 5.7: **Drop implausibly dense segments.** Whisper emitted "For more info, see the links in the description." — eight words — into a 15 ms segment, after the duration clamp squashed a hallucination past the end of speech rather than discarding it. No real segment fits eight words into 15 ms, so words-per-second is a reliable discriminator. Filter in normalization, with the threshold chosen from the corpus rather than guessed, and a test built from the observed case
- [x] Task 5.8: **Investigate over-generation.** `sastre8` scored coverage 1.019 at baseline: the transcript is longer than the reference. Establish whether that is hallucinated content, reference gaps, or normalization, since coverage is the metric guarding every other measurement in this track and a systematic bias in it would matter
- [x] Task 5.9: Reject options the detection mode would silently ignore (unplanned; see deviations)
- [x] Task 5.10: Run diarization on the resolved device, not always the CPU (unplanned; see deviations)
- [x] Task 5.11: Stop seeded runs holding two copies of the model on the device (unplanned; see deviations)

### Task 5.1 result — tuning

**Final: `language_threshold=0.5` kept, `chunk_seconds` changed 25 → 15** — both by full
evaluation, below. The section starts with the detection-only probe that came first, when only the
required evaluations were being run. That probe measured detection alone: every chunk's best-allowed language and probability, for
chunk lengths 10–30 s across all five recordings, with thresholds replayed offline.

At 25 s chunks (flipped = chunks whose language changes to the file-level fallback):

| threshold | flipped, per recording | decode spans, per recording |
| --------- | ---------------------- | --------------------------- |
| 0.3 | 0–1 | 5–35 |
| 0.5 (default) | 0–6 | 5–35 |
| 0.7 | 1–17 | 3–23 |
| 0.9 | 2–31 | 1–12 |

- **The threshold barely acts at 0.5.** Only 3.7% of 25 s chunks score below it; the median
  chunk scores 0.79. 0.3 and 0.5 are near-identical.
- **Raising it pushes toward the file-level language**, which is decided from the first 30 s of
  the recording and was wrong on exactly the recordings chunked mode helps most (`herring17`,
  `maria20`). A higher threshold would re-import the error chunked mode exists to fix. That is an
  argument, not a measurement.
- **Shorter chunks roughly double decode spans** on bilingual recordings (15 s: 32–55 spans
  where 25 s gives 19–35, excluding the 98%-Spanish `zeledon14`). Phase 3 measured that every boundary costs content.

**What this costs:** the defaults are reasoned, not measured against alternatives. Per-recording
seed noise (WER 0.259 vs 0.394 on one recording) is larger than any effect either knob is likely
to have, so a tuning claim would need a noise-floor run beside each candidate — about 25 minutes
of GPU each. Reopen if chunked mode becomes the default or a new corpus arrives.

### Task 5.6 result — default

**`file` stays the default; `chunked` stays opt-in.** Final reports, `eval/baseline.json` against
`eval/chunked.json` (15 s chunks, per Task 5.1), both on the finished code:

| recording | ΔWER | Δretained | mix error file → chunked |
| --------- | ---- | --------- | ------------------------ |
| herring17 | −0.076 | +0.016 | 0.341 → 0.096 |
| maria20   | −0.108 | +0.053 | 0.348 → 0.019 |
| sastre8   | +0.045 | −0.035 | 0.013 → 0.050 |
| zeledon14 | −0.008 | −0.005 | 0.001 → 0.010 |
| zeledon4  | −0.023 | −0.013 | 0.118 → 0.044 |

Mean: WER −0.034, retained +0.003, language-mix error −0.120. At 25 s chunks this was 2 of 5
better; at 15 s it is 4 of 5, with `zeledon14` within noise. The remaining loss is where it was:
`sastre8`, whose language file-level detection already had right, loses 0.045 WER beyond noise,
and mix error rises on the two recordings `file` mode got nearly right. The backend cannot tell
which kind of recording it has before transcribing, so `chunked` stays opt-in. **This is now a
closer call than when first decided**, and worth reopening with a larger corpus; flipping it is
out of scope for this track per the spec.

**Throughput (Task 5.2),** same runs: `file` 20.5 min for 150 min of audio (7.3x), `chunked`
23.3 min (6.4x), about 14% more. Diarization 22.9x on CUDA, 1.8x on CPU, on one 21.8-min
recording. Documented in the doc page.

**Measured after all (2026-09-23).** The full runs above were then made. Each compares against the
committed `eval/chunked.json` (threshold 0.5, 25 s, seed 0):

| run | ΔWER per recording (herring17, maria20, sastre8, zeledon14, zeledon4) | mean ΔWER | mean mix error |
| --- | ---------------------------------------------------------------------- | --------- | -------------- |
| `seed=1` — noise floor | +0.000, −0.001, −0.022, −0.012, +0.000 | −0.007 | 0.050 → 0.048 |
| `language_threshold=0.7` | −0.012, +0.030, +0.009, −0.006, +0.046 | +0.014 | 0.050 → 0.122 |
| `chunk_seconds=15` | −0.024, **+0.048**, −0.036, −0.017, −0.053 | −0.016 | 0.050 → 0.044 |

- **Noise floor: 0.022 WER per recording, 0.007 on the mean.** Far tighter than the 0.259 vs 0.394
  seed spread quoted elsewhere, which was measured on an older configuration.
- **Threshold 0.7 is worse**, as argued above: mix error doubles to quintuples on four
  recordings, WER rises beyond noise on two. 0.5 stays, now measured.
- **15 s chunks are better on 4 of 5** — three beyond noise — at the same wall-clock (23.3 vs
  23.0 min). Against the file-mode baseline they improve WER on 4 of 5 recordings (mean −0.034)
  where 25 s improves 2 (mean −0.018). The exception is `maria20`: +0.048 WER and −0.026
  retained, which trips the comparison guard. It is the recording with the sparsest,
  least trustworthy reference. **Decided 2026-09-23: default changed to 15 s**, on the 4-of-5
  result against a single seed. `eval/chunked.json` is now that run.

### Task 5.7 result — density filter

**Rule: drop a segment of ≥ 4 words packed faster than 50 words/s.** Chosen from the five
recordings in both modes (6684 segments), segment timings kept by the harness for the purpose.

| | max words/s seen |
| --- | --- |
| Real segments, ≥ 4 words | 28.6 |
| Real segments, 1–3 words | 100 — 20 ms timestamp resolution plus overlap clamping |
| The Phase 4 hallucination | 533 — 8 words in 15 ms |

Speech rate was not usable as the threshold. Whisper's timestamps sometimes collapse into
uniform 1.000 s slots regardless of length, putting 17–18 real, correctly transcribed words in
one second on `zeledon14`, `zeledon4` and `sastre8`. Any threshold near human speech rate deletes
real content.

The rule drops one segment in the corpus, in chunked mode: "She's a sweet girl." in 20 ms,
repeating its predecessor, where the reference has the line once. A true positive.

**Follow-up, not in scope:** at Whisper's minimum 20 ms duration, 10 of 11 chunked-mode segments
duplicate a neighbour — a seven-fold "No." loop on `zeledon4`, "That's my dream." twice on
`sastre8`. That is a repetition failure, and mostly one to three words, below what density can
judge without also dropping squashed real backchannels. Deduplicating 20 ms segments against their
neighbours would address it; it needs its own measurement.

### Task 5.8 result — over-generation

**Not over-generation. The 1.019 was a superseded measurement; the finding is that coverage
hides deletions.**

The figure comes from the baseline committed in `6f61356`, taken before the per-file model
rebuild. `sastre8` was the third file of that run, so it decoded with RNG state left by the two
before it — the order dependence that commit fixed. The same recording scored 531 insertions
there against 362 in the valid baseline. Reproducibility was re-verified: a fresh file-mode run
today produced transcripts byte-identical to the committed baseline's on all five recordings.
No recording now exceeds coverage 1.0.

Of the three candidate causes:

- **Hallucination** — present but minor. Repetition loops ("frantitude" ×30, "jean and a polo
  shirt" repeated) account for 4–133 inserted words per recording; most insertions are short
  scattered runs.
- **Reference gaps** — no evidence in `sastre8`: no insertion run outside a loop exceeds seven
  words. `maria20` is a different story — 51 reference words per minute against about 200
  elsewhere, 8% of them `xxx` — and is flagged as the least trustworthy recording in
  `docs/evaluation.md`.
- **Normalization** — `xxx` (unintelligible) and fillers survive reference cleaning and can never
  be matched. They bias every report equally, so they do not distort comparisons.

**What mattered:** coverage is a length ratio, so insertions offset deletions. At baseline:

| recording | coverage | retained (1 − deletions/reference) | gap |
| --------- | -------- | ---------------------------------- | --- |
| herring17 | 0.761 | 0.749 | 0.012 |
| maria20   | 0.861 | 0.806 | 0.055 |
| sastre8   | 0.987 | 0.924 | 0.063 |
| zeledon14 | 0.979 | 0.922 | 0.057 |
| zeledon4  | 0.862 | 0.833 | 0.029 |

The comparison guard now uses `retained`. Phase 3 re-checked: chunked-vs-baseline retention
deltas have the same sign as the coverage deltas on all five recordings, so its verdicts stand.

### Verification

- [x] Doc page current, per gate 6
- [x] A real run over bilingual audio produces varying `spoken_language` and populated `speaker`
  — 2026-09-23, CLI, `zeledon4` (21.8 min), `large-v3`, chunked + diarization, CUDA: 554
  segments, `en` 343 / `es` 211, `SPEAKER_00` 248 / `SPEAKER_01` 299 / `null` 7 (no overlapping
  turn). 241 s wall-clock, peak VRAM 6.7 GB. Manifest records mode, threshold, languages,
  chunk length, diarization model and pyannote version; token not written
- [x] No segment in the eval output exceeds the words-per-second threshold from Task 5.7 — 0 of 6683
  in the final reports' transcripts

### Deviations

**2026-09-23 — order.** Tasks 5.7 and 5.8 run before 5.1. The density filter changes what
normalization emits in both modes, so tuning measured before it would be tuning a pipeline that
no longer exists.

**2026-09-23 — unplanned fix (Task 5.9).** Writing the doc page for 5.3 surfaced a false claim:
that a forced `language` overrides chunked detection. It did not — chunked mode detected anyway
and overwrote it. The mirror case held too: file mode never read `languages`,
`language_threshold` or `chunk_seconds`. All four were recorded in the manifest as if applied,
the defect Task 2.6 fixed for decode options. Contradictions are now rejected at construction.

**2026-09-23 — unplanned fix (Task 5.10).** Measuring throughput for 5.2 found that the pyannote
pipeline was never moved off the CPU. On `zeledon4` (21.8 min) it ran 724 s on CPU against 57 s
on CUDA, with identical speaker turns — so on CPU, diarization took longer than transcription.
It now runs on the device transcription resolved to.

**2026-09-23 — unplanned fix (Task 5.11).** The first end-to-end run with both models on the GPU
took 22 minutes for a 22-minute recording and was stopped. VRAM read 7.9 GB within 30 s. Cause:
the per-file model rebuild that applies the seed (Task 2.5) dropped its reference to the old
model and assumed that freed it. It did not — one copy survived even an explicit `gc.collect()` —
so every seeded run held two copies of `large-v3`. That fitted on 8 GB for transcription alone;
adding diarization overflowed it, WSL2 paged GPU memory to system RAM, and a 5-minute excerpt
transcribed at 2.3x realtime instead of 6–8x. The old model is now unloaded explicitly before the
next is built.

## Final Verification

- [ ] All acceptance criteria in [spec.md](./spec.md) met — **one not met:** WER does not improve on
  all five recordings (2 of 5), and Spanish share is not within 0.05 on every one. See Phase 3 and
  Task 5.6: inherent to the approach, which is why `chunked` stays opt-in
- [x] Gate 1 — `pytest` passing
- [x] Gate 2 — detection, chunking, and normalization covered by tests
- [x] Gate 3 — `pytest -k conformance` passing in every mode
- [x] Gate 4 — `ruff check .` clean
- [x] Gate 5 — `ruff format --check .` clean
- [x] Gate 6 — the `faster-whisper` doc page covers every new option
- [x] Gate 7 — manifest records detection mode, threshold, languages, and diarization model
- [x] Eval report committed, showing the improvement over the recorded baseline — language-mix
  error improves on all five; WER on two
- [ ] Manual sign-off recorded here

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
