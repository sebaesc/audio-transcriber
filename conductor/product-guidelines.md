# Product Guidelines

## Voice & Tone

### Brand Voice

Concise and direct. Short, plain sentences. Documentation and CLI output state what happened
and what to do next, with no filler. Written for a technical reader who wants the answer, not
an introduction.

### Voice Attributes

- **Direct:** Lead with the fact or the action. No preamble.
- **Precise:** Name the file, the backend, the flag. Avoid vague nouns like "the system".
- **Plain:** Ordinary words over jargon, except for domain terms (diarization, WER, VAD) used correctly.

### Tone Variations by Context

| Context        | Tone                    | Example                                                        |
| -------------- | ----------------------- | -------------------------------------------------------------- |
| Success states | Factual, brief          | "Transcribed 12 files in 4m18s → out/2026-09-22/"               |
| Error states   | Specific, actionable    | "ffmpeg not found. Install it: apt install ffmpeg"              |
| Onboarding     | Step-by-step, minimal   | "1. Create the venv. 2. Install. 3. Run the CLI on a sample."   |
| Empty states   | Neutral, with next step | "No audio files matched. Check the input path or --glob."       |

### Words We Use

- backend (a transcription model implementation)
- run (one invocation of the pipeline over an input set)
- transcript (the normalized output object)

### Words We Avoid

- "simply", "just", "easy" — they hide real setup cost
- "AI-powered" — say which model
- "engine" and "provider" as synonyms for backend; pick one term and keep it

## Messaging Guidelines

### Core Messages

**Primary Message:**

> The transcription model is a plug-in. The pipeline does not change when the model does.

**Supporting Messages:**

1. One transcript schema across every backend.
2. Every backend documents its own requirements before you need them.
3. Runs are reproducible: same input and config produce the same output.

### Message Hierarchy

1. **Must Communicate:** Backends are swappable behind a stable interface.
2. **Should Communicate:** Output is normalized and structured (timestamps, speakers).
3. **Could Communicate:** Comparing backends on the same audio is possible as a result.

### Audience-Specific Messaging

| Audience                  | Key Message                                              | Proof Points                                     |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------ |
| Internal team             | Change models without downstream breakage                 | Shared schema; conformance tests per backend     |
| Future integrators        | Implement one interface to add an engine                  | Documented backend contract; per-backend doc page |

## Design Principles

### Principle 1: Modularity / Clean Interfaces

Model backends sit behind a stable interface. Pipeline code never depends on a specific
engine — no engine-specific imports, branches, or output handling outside the backend module.

**Do:**

- Define the backend contract (inputs, outputs, errors, capabilities) and keep it stable
- Normalize every backend's output to the shared transcript schema inside that backend
- Register backends by name so selection is config, not code

**Don't:**

- Branch on the backend name in pipeline or CLI logic
- Let an engine's native output shape reach downstream consumers
- Add a pipeline feature that only one backend can support

### Principle 2: Reproducibility

The same input plus the same config produces the same output. Configs and model versions are
recorded alongside results, so any transcript can be traced back to what produced it.

**Do:**

- Write a run manifest with backend name, model version, config, and input checksums
- Pin model and dependency versions explicitly
- Set seeds and deterministic options where the backend exposes them

**Don't:**

- Depend on ambient state (unpinned "latest" model tags, unset env vars, wall-clock defaults)
- Overwrite prior run outputs in place
- Leave the effective config implicit in CLI flag defaults

### Principle 3: Documented Requirements

Each backend is documented before it is considered done: how to use it, and what it needs.

**Do:**

- Give each backend a doc page: usage, dependencies, credentials, hardware, known limits
- State system-level requirements (ffmpeg, CUDA, RAM) explicitly
- Note cost and rate limits for hosted backends

**Don't:**

- Assume the reader already has the model weights or an API key
- Document only the happy path

## Accessibility Standards

Not applicable in the usual sense — this is a CLI tool with no UI. The equivalent commitments:

- Output is plain text and machine-readable (JSON) so it can be consumed by any tool
- No information conveyed by color alone; terminal color is decoration, never meaning
- Progress and errors go to stderr; transcript data goes to stdout or files, so piping works
- Long-running operations report progress rather than appearing hung

## Error Handling Philosophy

### Error Prevention

- Validate config and input paths before starting a run, not per file
- Check backend requirements (binaries, credentials, model files) up front and fail fast
- Confirm before overwriting an existing output directory

### Error Communication

#### Principles

1. **Be specific:** Name the file, backend, and failing step.
2. **Be helpful:** Give the command or config change that fixes it.
3. **Be plain:** Technical is fine; cryptic is not.
4. **Be timely:** Fail at validation time when the problem is knowable then.

#### Error Message Structure

```
[What happened] + [Why it happened (if relevant)] + [How to fix it]
```

#### Examples

| Bad                       | Good                                                                        |
| ------------------------- | --------------------------------------------------------------------------- |
| "Invalid input"           | "meeting.m4a: unsupported codec. Convert it: ffmpeg -i meeting.m4a out.wav" |
| "Backend error"           | "whisper-api: 401 unauthorized. Set OPENAI_API_KEY."                        |
| "Failed"                  | "3 of 12 files failed. See out/2026-09-22/errors.json"                      |

### Error States

| Severity | Visual Treatment              | User Action Required |
| -------- | ----------------------------- | -------------------- |
| Info     | Plain line on stderr          | Optional             |
| Warning  | `warning:` prefix on stderr   | Recommended          |
| Error    | `error:` prefix, non-zero exit | Required             |
| Critical | `error:` prefix, abort run    | Immediate            |

### Recovery Patterns

- Per-file failures do not abort the batch; they are collected and reported at the end
- Runs are resumable: already-transcribed files are skipped unless forced
- Partial results are kept, never discarded on failure
- Exit codes distinguish "all succeeded", "partial failure", and "run aborted"
