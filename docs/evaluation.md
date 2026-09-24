# Evaluation harness

Measures transcription quality against a reference corpus, so a change to a backend can be shown
to help rather than assumed to. It is a development tool: nothing in the pipeline imports it, and
it is not part of transcribing audio.

## Install

```bash
.venv/bin/pip install -e ".[eval,faster-whisper]"
```

The `eval` extra adds `jiwer` (word error rate) and `pyarrow` (reading the corpus). It is never a
dependency of the core package or of any backend.

## The corpus

[`BrunoHays/Bangor-Miami-Spanish-English-Corpus`](https://huggingface.co/datasets/BrunoHays/Bangor-Miami-Spanish-English-Corpus),
CC BY-SA 3.0: naturalistic Spanish-English conversation recorded in Miami. The harness uses the
`mixed` configuration — five recordings, about 2.5 hours, each between 20% and 80% Spanish —
because a recording that is 98% English says nothing about code-switching.

| Recording | Length | Reference words | Words / min | Notes |
| --------- | ------ | --------------- | ----------- | ----- |
| `herring17` | 30.0 min | 6042 | 201 | |
| `maria20` | 31.9 min | 1615 | 51 | Sparse reference; 8% of it is `xxx` (unintelligible). See below |
| `sastre8` | 33.4 min | 5791 | 173 | |
| `zeledon14` | 33.1 min | 6758 | 204 | |
| `zeledon4` | 21.8 min | 4225 | 193 | |

The 116 MB parquet downloads on first use into `~/.cache/audio-transcriber/bangor-miami`, is
checked for size and format, and is never committed. Set `AUDIO_TRANSCRIBER_EVAL_CACHE` to use a
different directory, e.g. one shared between users on a machine.

**No speaker labels in this copy.** The Hugging Face dataset strips CHAT speaker tiers and
language marks. TalkBank's original annotations keep both; see
[Speakers and languages: TalkBank](#speakers-and-languages-talkbank).

## Running

```bash
# Evaluate one configuration over all five recordings
.venv/bin/python -m audio_transcriber.evaluation \
    --backend faster-whisper --label chunked \
    --option model=large-v3 \
    --option language_detection=chunked \
    --option 'languages=["en","es"]'

# Limit to some recordings while iterating
.venv/bin/python -m audio_transcriber.evaluation -b faster-whisper -l try.local \
    --option model=large-v3 --only zeledon4 --only maria20

# Compare two existing reports, recording by recording
.venv/bin/python -m audio_transcriber.evaluation --compare eval/baseline.json eval/chunked.json
```

`--option` values are parsed as JSON where they parse, and kept as strings otherwise, so
`beam_size=1`, `vad_filter=false` and `languages=["en","es"]` all mean what they look like. An
option the backend does not recognize is rejected, not ignored.

Output:

| Path | Contents | Committed |
| ---- | -------- | --------- |
| `eval/<label>.json` | The report | Yes — reports are the numbers to beat |
| `eval/<label>.local.json` | A report you do not intend to keep | No, gitignored |
| `eval/transcripts/<recording>.<label>.txt` | The normalized hypothesis that was scored | No |
| `eval/transcripts/<recording>.<label>.json` | The hypothesis as segments, with timings | No |

Use a `.local` label for exploratory runs so they cannot be committed by accident.

`--compare` exits 0 only when the candidate improved on **every** recording without dropping
content (see `retained` below), and 1 otherwise, so it can gate a script.

### How long it takes

`large-v3` on an RTX 3060 runs this corpus at roughly 5–10x realtime, so a full evaluation of the
2.5 hours takes 15–30 minutes per configuration. `maria20` is faster than the rest because it is
mostly silence. On CPU, budget hours, or use `--only` and a smaller model while iterating — but
make any claim about a change against all five recordings on the model you will ship.

## What it measures

Every score is computed after normalizing both sides: lowercase, punctuation stripped, accents
folded. The reference first has its CHAT markup removed — `(.)` pauses, `<>` retracing brackets,
`+` continuations, `&=laughs` event codes, `[/]` inline codes — and partial words like
`mani(cure)` are written out whole.

| Field | Meaning | Better |
| ----- | ------- | ------ |
| `wer` | Word error rate from `jiwer`: (substitutions + deletions + insertions) / reference words | Lower |
| `retained` | Reference words transcribed as anything (hit or substitution) rather than deleted, / reference words | Higher |
| `coverage` | Hypothesis words / reference words | Near 1.0 |
| `reference_spanish_share` | Spanish share of the reference, by function words | — |
| `hypothesis_spanish_share` | The same, for the transcript | Near the reference |
| `language_mix_error` | \|hypothesis − reference\| Spanish share | Lower |
| `substitutions`, `deletions`, `insertions`, `hits` | The WER breakdown | |

**Why `retained` exists.** WER rewards emitting less. A deletion costs one error; a wrong guess
costs a substitution and risks an insertion as well. A change that silently drops a third of the
audio can therefore lower WER. `--compare` treats a WER gain with `retained` more than 0.02 below
the baseline's as no gain.

**Why coverage is not the guard.** Coverage is a length ratio, so invented words offset dropped
ones: `sastre8` scores coverage 0.987 at baseline with 7.6% of its reference deleted. Read it as
the over-generation signal — well above 1.0 means the transcriber is inventing — and read
`retained` for dropped content. The gap between the two is roughly the insertion rate.

**How the language measure works.** Spanish share is the fraction of marker words — short, common
function words (`que`, `pero`, `the`, `because`) — that are Spanish. It measures orthography, not
meaning: it detects English written out as Spanish under the wrong language token, and the
reverse. It is aggregate, not per segment; the corpus's word-level language IDs are not in the
published dataset.

Each recording also reports `detected_languages` — the distinct `spoken_language` values in the
transcript — and `realtime_factor`, audio duration over wall-clock time.

## Reading a report

**Judge per recording, never on the mean.** The prototype this harness was built to check
improved one recording by 23% and worsened another by 36%. A mean would have called that a wash.
`--compare` reports each recording's deltas and counts how many improved.

**Know the noise floor.** Decoding samples above temperature 0.0, and the backend pins
CTranslate2's seed so a configuration gives byte-identical output run to run. But that makes the
result reproducible, not representative: a different seed gives a different, equally valid
transcript. Measured on the committed chunked configuration, changing only the seed moved WER by
up to 0.022 per recording and 0.007 on the mean. A per-recording difference smaller than that is
not evidence on its own. (An older configuration showed a far wider spread — 0.259 against 0.394
on one recording — so re-measure the floor when the configuration changes.) Look for changes that move most
recordings the same way, and never tune the seed.

**Compare like with like.** The model, device and every backend option are in the report's
`config`. Two reports with different `model_version` or `config.device` are not comparable.

### Known biases

These are present in every report equally, so they do not distort a *comparison* between two
configurations. They do mean absolute numbers are pessimistic.

- **`xxx` stays in the reference.** CHAT marks unintelligible speech `xxx` (and `yyy`, `www`).
  Cleaning leaves them in, so each is a reference word no transcriber can match: 0.8–2.6% of the
  reference on four recordings, and **8.0% on `maria20`**.
- **Fillers.** The reference transcribes `um`, `ah`, `aha`; Whisper mostly omits them. 0.5–2.6%
  of reference words.
- **`maria20` is sparsely transcribed.** 51 words per minute against about 200 for the others.
  Its coverage and WER describe a small, partly unintelligible reference and are the least
  trustworthy of the five.
- **`no` counts as Spanish** in the language measure, though it is equally common in English.
  It inflates both shares, so the mix *error* is less affected than either share.

## Speakers and languages: TalkBank

`--talkbank` adds two scores per recording, taken against TalkBank's original annotations of the
same five recordings: **who spoke** (diarization) and **which language each stretch of speech is
in**. The Hugging Face copy cannot answer either. TalkBank's timestamps align with the Hugging Face
audio: matching transcript words against our segments peaks at a 0 s offset on all five.

```bash
.venv/bin/python -m audio_transcriber.evaluation -b faster-whisper -l chunked --talkbank \
    --option model=large-v3 --option diarization=true \
    --option language_detection=chunked --option 'languages=["en","es"]'
```

### Access and licence

1. Create a free TalkBank account: the green **Login** button at <https://talkbank.org/biling/>,
   then **New User**.
2. While logged in, download the transcripts: <https://talkbank.org/data/biling/Bangor/Miami?f=zip>
   (3 MB, 56 CHAT files). The audio is not needed; the harness already has it.
3. Save the zip as `~/.cache/audio-transcriber/talkbank/Miami.zip`, or point
   `AUDIO_TRANSCRIBER_TALKBANK_ARCHIVE` at it.

TalkBank data is **CC BY-NC-SA 3.0**, and its [ground rules](https://talkbank.org/0share/rules.html)
keep downloaded data on the local device and forbid passing it on. So the archive is never
committed and never copied into the repo, test fixtures included. Only aggregate scores go into
reports. Anyone re-running this needs their own account. Any use must cite:

> Deuchar, M., & Parafita Couto, M. (2011). The Bangor Miami Corpus. TalkBank.
> doi:10.21415/T5J01D

Reports carry this citation whenever TalkBank scores are present.

### What it scores

Each CHAT utterance has a speaker, a timestamp, and a language derived from the file's base
language, `[- spa]` precodes and `@s` word switches: `en`, `es`, or `mixed` when it holds both.

**Diarization** (`talkbank.diarization`, only when the run diarized):

| Field | Meaning |
| ----- | ------- |
| `confusion` | Reference speech attributed to the wrong speaker. **The number to read for speaker-label quality** |
| `missed` | Reference speech with no labelled segment over it |
| `false_alarm` | Labelled segments over no reference speech |
| `der` | The sum: diarization error rate |
| `reference_seconds` | Reference speech scored |

Our labels (`SPEAKER_00`) are mapped onto the reference speakers by best agreement, as standard. A
0.25 s collar either side of each reference boundary is forgiven. What is scored is the
transcript's segments, the labels a user receives, not the diarizer's raw turns. So `missed` and
`false_alarm` mostly measure where Whisper's segments start and stop: TalkBank's utterance times
span pauses that Whisper's segments do not. They also count speech Whisper never transcribed.
That makes `der` pessimistic, and why `confusion` is the diarization-quality number.

**Language** (`talkbank.language`, always):

| Field | Meaning |
| ----- | ------- |
| `accuracy` | Share of English and Spanish reference speech whose overlapping segment carries the right `spoken_language`, by time |
| `recall.en`, `recall.es` | The same, per reference language: how much of each language was labelled as itself |
| `mixed_share` | Share of speech in utterances that mix both languages. No single label is right for them, so they are left out of accuracy |
| `confusion` | Seconds, as reference language → segment language |

This is stronger than `language_mix_error`. That compares aggregate Spanish share, so a transcript
whose segments are individually wrong can still score well as long as the totals balance.
`accuracy` cannot be fooled that way.

`mixed_share` measures the limit that chunk-level detection cannot cross: speech in utterances
holding both languages. Per recording it runs from 9% to 25% of speech time.

### Limits

- **`maria20` is partly timed.** 255 of its 380 utterances carry timestamps. The gaps holding the
  untimed ones are left out of diarization scoring rather than counted as false alarms, so its
  scores cover part of the recording.
- **Overlapping speech** is scored against whichever speakers TalkBank timed there. Crosstalk is
  common in this corpus and slightly inflates `confusion`.
- **Five recordings, two speakers each.** Enough to judge whether speaker labels are usable, not
  to rank diarization models finely.
- **Utterances with no known language** (only `xxx`, fillers) count as speech for diarization and
  are ignored for language.

