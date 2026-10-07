# Evaluation harness

Measures transcription quality against a reference corpus, so a change to a backend can be shown
to help rather than assumed to. It is a development tool: nothing in the pipeline imports it, and
it is not part of transcribing audio.

The corpus is a fixed benchmark, but it is not our audio. To score a backend on the team's own
recordings, against a ground truth you write, see
[Your own recordings: a custom ground truth](#your-own-recordings-a-custom-ground-truth).

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

# Every model side by side, recording by recording, in one table
.venv/bin/python -m audio_transcriber.evaluation --table

# What each metric in that table means
.venv/bin/python -m audio_transcriber.evaluation --metrics
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
| `eval/comparison.md` | The side-by-side table `--table` writes | No, regenerated from the reports |

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

### Every model side by side: `--table`

`--compare` judges two reports. `--table` lays out any number of them, corpus and custom alike,
model by model and recording by recording. It reads reports only: no backend runs, so nothing is
spent.

```bash
# Every kept report: eval/*.json and eval/custom/*.json, without the *.local.json ones
.venv/bin/python -m audio_transcriber.evaluation --table

# Only these reports, local ones included
.venv/bin/python -m audio_transcriber.evaluation --table eval/chunked.json eval/scribe.json \
    eval/custom/*.json
```

The table is printed and written to `<out>/comparison.md` (gitignored; regenerate it). It has:

| Section | Contents |
| ------- | -------- |
| `Corpus`, `Custom ground truth` | One per source with reports. A block of rows per recording, one row per report |
| `Means over each model's scored recordings` | Under each source: one row per report, with how many recordings it scored |
| `Models` | Each report's source, backend, full model version, and whether it diarized |
| `Metrics` | A one-line legend; `--metrics` has the full definitions |

Every row names the report (`Model`, its label) and the backend and model that produced its
scores (`Backend`, such as `faster-whisper large-v3` or `elevenlabs-scribe scribe_v2`; `unknown`
for transcripts scored without a run manifest). The metric columns are:

| Column | Report field | Better |
| ------ | ------------ | ------ |
| `WER` | `score.wer` | Lower |
| `Retained` | `score.retained` | Higher |
| `Coverage` | `score.coverage` | Nearest 1.0 |
| `Lang mix err` | `score.language_mix_error` | Lower |
| `DER`, `Missed`, `False alarm`, `Spk conf` | `<talkbank\|ground_truth>.diarization.der`, `.missed`, `.false_alarm`, `.confusion` | Lower |
| `Lang acc` | `<talkbank\|ground_truth>.language.accuracy` | Higher |
| `Mixed` | `<talkbank\|ground_truth>.language.mixed_share` | Neither: a property of the recording |
| `RTF` | `realtime_factor` | Higher |

The headers carry the direction, `↓` lower is better and `↑` higher is better. Columns are padded
so the table lines up as plain text too, in a terminal or an editor, and still renders as
Markdown. Metrics show three decimals, the real-time factor one.

- A `–` is a metric that was not measured: no diarization, a corpus run without `--talkbank`, or
  no timing for a re-scored transcript. A recording the backend failed on reads `error`.
- Means cover only the recordings each report scored — the `Scored` column — so compare them
  only where those counts match. Judge per recording, as above.
- Two reports from the same source with the same label are told apart by file name.
- To put a model's transcripts of your recordings in the table, score them first with
  `--transcripts` (see
  [Scoring a transcript you already have](#scoring-a-transcript-you-already-have)), once per run
  folder.
- `--table` takes no run flags (`--backend`, `--label`, `--option`, `--only`, `--talkbank`, the
  custom inputs or `--compare`): misuse exits 2. No reports, or a file that is not a report,
  exits 3.

### What each metric means: `--metrics`

```bash
.venv/bin/python -m audio_transcriber.evaluation --metrics
```

Prints, for every column of the table, its definition, which value is better and the report field
it is read from. It takes no other mode and exits 0; combined with one, it exits 2. The
definitions live beside the columns in `evaluation/table.py`, so the two cannot drift apart.

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

- **`maria20` cannot score speakers.** 255 of its 380 utterances carry timestamps, and all 255
  belong to one speaker: the other speaker is entirely untimed. Anything the diarizer does with
  that speaker's speech would be scored against the first speaker's turns, which read as 46%
  confusion and measured nothing. So diarization is skipped wherever a speaker has no timed speech,
  and the report records why. Its language scores still count.
- **Partly timed references** otherwise have the gaps holding untimed utterances left out of
  diarization scoring, rather than counted as false alarms.
- **Overlapping speech** is scored against whichever speakers TalkBank timed there. Crosstalk is
  common in this corpus and slightly inflates `confusion`.
- **Five recordings, two speakers each.** Enough to judge whether speaker labels are usable, not
  to rank diarization models finely.
- **Utterances with no known language** (only `xxx`, fillers) count as speech for diarization and
  are ignored for language.

## Your own recordings: a custom ground truth

The corpus decides nothing on its own: what matters is how a backend does on the team's recordings,
English with Spanish speakers who code-switch. So the harness also scores against a ground truth
you write for one of them, either by running a backend on the recording or by scoring a transcript
the pipeline already made. The scores are the corpus's, and the reports have the same shape, so
`--compare` works on them.

### Writing a ground truth

A ground truth is the pipeline's own transcript format: a JSON array of segments, one file per
recording, named after it (`standup.mp4` → `standup.json`).

```json
[
  {
    "start_time": 0.0,
    "end_time": 4.2,
    "speaker": "Bob",
    "spoken_language": "en",
    "text": "So the meeting is at three, right?"
  },
  {
    "start_time": 3.9,
    "end_time": 6.5,
    "speaker": "Ana",
    "spoken_language": "es",
    "text": "Sí, a las tres. ¿Y tú qué piensas?"
  }
]
```

| Key | Required | Notes |
| --- | -------- | ----- |
| `start_time`, `end_time` | Yes | From the start of the recording: seconds (`1192.5`) or a clock time (`"19:52"`, `"1:02:03"`). `end_time` not before `start_time` |
| `text` | Yes | What was said. Write it as spoken: punctuation, case and accents do not count |
| `speaker` | No | Any name, used consistently. `null` or left out means unlabelled |
| `spoken_language` | No | `en` or `es`; `mixed` for a segment that holds both. `null` or left out means unlabelled |

- **Overlap is allowed.** Two people talking at once is two segments over the same time. A
  pipeline transcript never overlaps; a ground truth may.
- **Order does not matter.** Segments are sorted by time when read.
- **Times need not be exact.** Diarization forgives 0.25 s either side of each boundary. Times
  written to the second, as `MM:SS` is, are coarser than that, so `missed` and `false_alarm`
  read high; `confusion` is still the number to trust.
- **Label speakers on every segment or none.** A segment with no speaker is speech that belongs to
  nobody, so speakers are only scored when all segments have one. Languages can be partly
  labelled: unlabelled segments are left out of language scoring.
- **Keep the files out of the repo.** Recordings and ground truths are the team's data. Reports,
  which hold scores and each ground truth's checksum, may be committed.

### A clip cut from a longer video

Recordings to evaluate are usually a clip, about 20 minutes, cut from a longer class video, and
each clip starts somewhere else in its video. Write the ground truth with the times you see in the
**full video**, and say where the clip starts with `clip_start`. The file is then an object instead
of a bare array:

```json
{
  "clip_start": "19:53",
  "segments": [
    {
      "start_time": "19:58",
      "end_time": "20:08",
      "speaker": "profesor",
      "spoken_language": "en",
      "text": "Is it pizza?"
    }
  ]
}
```

| Key | Required | Notes |
| --- | -------- | ----- |
| `segments` | Yes | The segments, exactly as in a bare array, timed against the full video |
| `clip_start` | No | Where the clip starts in that video: seconds or a clock time, as segment times are. Left out means `0` |

Every time is moved back by `clip_start` when read, so the segment above is scored as `5`–`15` s
of the clip. Each file carries its own `clip_start`, so a folder of clips that start at different
points is scored in one run. A bare array is still read as before, as a clip starting at `0`.

**Speech outside the clip is cut.** Speech before `clip_start`, or after the clip's end, is not in
the recording: neither its words nor its time are scored. A segment that crosses either edge keeps
the part inside. A clip cut a second off from its ground truth loses a word or two this way.

**A ground truth for the wrong stretch of time is refused.** If more than 20% of its speech falls
outside the recording, the run stops before anything is transcribed:

```
error: ground truth truth/week8.json: 93% of its speech falls outside the audio (0–1216 s) with no
clip_start. It is written for 1192–2415 s: is clip_start missing or wrong? See docs/evaluation.md.
```

This is the mistake to guard against: a 20-minute clip scored against a ground truth in the full
video's times has the same WER, since words carry no times, but its speaker and language scores
are measured over the seconds where the two happen to overlap. They can look perfect, a speaker
`confusion` of `0.000`, while measuring nothing.

When [re-scoring a transcript](#scoring-a-transcript-you-already-have), the clip's length is read
from the run's `manifest.json`. Without one, the ground truth is checked against where the
transcript's speech ends, and nothing after it is cut.

### Mistakes in a ground truth

A ground truth is checked before anything is transcribed. A file that is not a JSON array (or a
clip's object), is empty, or has a segment with a missing key or its end before its start stops
the run, with the file, the segment's index and the problem. A clip's times are reported as
written, in the full video's times:

```
error: ground truth truth/standup.json, segment 12: segment end_time (61.0) is before its start_time (62.5)
```

### Running a backend on your recordings

```bash
# One recording and its ground truth (audio or video; ffmpeg decodes either)
.venv/bin/python -m audio_transcriber.evaluation -b faster-whisper -l large-v3 \
    --option model=large-v3 --option diarization=true \
    --audio ~/recordings/standup.mp4 --reference ~/recordings/standup.json

# A folder of recordings, each beside its ground truth: name.<ext> + name.json
.venv/bin/python -m audio_transcriber.evaluation -b faster-whisper -l large-v3 \
    --option model=large-v3 --option diarization=true \
    --references ~/recordings

# Limit a folder to some recordings
.venv/bin/python -m audio_transcriber.evaluation -b faster-whisper -l try.local \
    --references ~/recordings --only standup
```

In a folder, every recording needs a ground truth and every ground truth a recording; a file
without its partner, or two recordings with the same name, is an error naming it. Other files are
ignored, and only the top level is searched. `--only` is applied first, so an unrelated unpaired
file does not block a run.

### Scoring a transcript you already have

To re-score a transcript, say from a hosted backend, without transcribing again:

```bash
# One transcript
.venv/bin/python -m audio_transcriber.evaluation -l scribe \
    --transcript out/2026-10-02T10-00-00Z/standup.json --reference ~/recordings/standup.json

# A pipeline run's output folder, matched to the ground truths by name
.venv/bin/python -m audio_transcriber.evaluation -l scribe \
    --transcripts out/2026-10-02T10-00-00Z --references ~/recordings
```

No backend runs and no `--backend` is given. The backend, model version and config are read from
the run's `manifest.json` beside the transcripts; without one, they are reported as `unknown`.
Every transcript in the folder needs a ground truth (use `--only` to pick some), but ground truths
without a transcript are errors too. The scores are exactly those a backend run gives for the same
transcript.

**This is the default way to score a hosted backend.** About 20 minutes of `elevenlabs-scribe`
costs roughly a seventh of a free-tier month: transcribe once with the pipeline, then score and
re-score the output.

### Where the output goes

| Path | Contents | Committed |
| ---- | -------- | --------- |
| `eval/custom/<label>.json` | The report | Yes |
| `eval/custom/<label>.local.json` | A report you do not intend to keep | No, gitignored |
| `eval/custom/transcripts/<name>.<label>.txt` | The normalized hypothesis that was scored | No, gitignored |
| `eval/custom/transcripts/<name>.<label>.json` | The hypothesis as segments | No, gitignored |
| `eval/custom/transcripts/<name>.reference.txt` | Your ground truth, normalized as scored | No, gitignored: it is your text |

`--out` moves all of it: reports go under `<out>/custom/`. To see which words cost the WER:

```bash
git diff --no-index --word-diff eval/custom/transcripts/standup.reference.txt \
    eval/custom/transcripts/standup.large-v3.txt
```

### What the scores mean on your data

**Words.** `wer`, `retained`, `coverage` and the language-mix fields mean what they do for the
corpus (see [What it measures](#what-it-measures)). Both sides are normalized the same way, but
there is no CHAT markup to strip, so everything in your `text` counts: write `um` only if you want
fillers scored, and never write placeholders like `[inaudible]` or `xxx`, which no transcriber can
match.

**Speakers and languages** are under `ground_truth` in the report, where corpus reports have
`talkbank`, and are scored the same way (see [What it scores](#what-it-scores)):

| Field | Meaning |
| ----- | ------- |
| `ground_truth.diarization.confusion` | Your speech attributed to the wrong speaker. The number to read for speaker labels |
| `ground_truth.diarization.missed`, `false_alarm`, `der` | Mostly where segments start and stop against yours; pessimistic, as for TalkBank |
| `ground_truth.language.accuracy` | Share of your `en` and `es` speech whose overlapping segment has the right `spoken_language`, by time |
| `ground_truth.language.recall.en`, `.es` | The same, per language |
| `ground_truth.language.mixed_share` | Share of speech you labelled `mixed`; left out of accuracy |
| `ground_truth.language.confusion` | Seconds, as your label → the transcript's |

Speaker scores need `pyannote.metrics`, which the `diarization` extra installs; a run that
diarizes has it already.

**Only `en` and `es` count towards accuracy.** That suits the team's audio. Other codes, if you
use them, appear in the confusion table but not in accuracy.

**Skipped scores say why.** When your ground truth has no speaker labels (or only some), or no
language labels, that score is `null` and `diarization_skipped` or `language_skipped` gives the
reason. So does a transcript with no speaker labels, from a run that did not diarize.

**Traceability.** Each recording in the report records its ground truth's `reference.file` and
`reference.checksum`, and for a clip its `reference.clip_start`. `reference.outside_s`, when
present, is the seconds of ground-truth speech cut for falling outside the recording. Compare only
reports scored against the same ground truths: if the checksums differ, the reference changed and
so did what the scores mean.

