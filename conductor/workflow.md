# Development Workflow

## Core Principles

1. **plan.md is the source of truth** — all task status and progress tracked in the track's plan
2. **Tests where they earn their keep** — flexible TDD; required for complex logic, not everywhere
3. **Conventional Commits** — machine-readable history
4. **Incremental progress** — small, verifiable commits with a clear purpose

## Testing Policy

**Strictness: Flexible.**

Tests are written for complex logic, not as a blanket requirement. Write tests when the code
is one of:

- The **backend interface** and its contract (every backend implementation)
- **Output normalization** — mapping an engine's native output to the shared transcript schema
- **Run manifest / reproducibility** logic
- The **CLI contract** — flags, exit codes, output layout
- Any **bug being fixed** — reproduce it with a test first

Tests are optional for thin wrappers, glue code, and one-line pass-throughs.

There is **no fixed coverage percentage gate**. A new backend must pass the shared backend
conformance tests before it is considered done.

Tests may be written before or after the implementation. Writing the failing test first is
encouraged for bug fixes and for the backend interface.

## Task Lifecycle

### Step 1: Task Selection

- Review plan.md for the next pending task
- Verify dependencies are complete
- Confirm understanding of the acceptance criteria

### Step 2: Progress Marking

- Update task status in plan.md from `[ ]` to `[~]`

### Step 3: Implement

- Write the code for the task
- For complex logic (see Testing Policy), write the test first and watch it fail

### Step 4: Test

- Add or update tests per the Testing Policy
- Run the suite: `pytest`

### Step 5: Refactor

- Improve structure without changing behavior
- Apply `conductor/code_styleguides/python.md` and `general.md`

### Step 6: Lint & Format

- `ruff format .`
- `ruff check --fix .`

### Step 7: Deviation Documentation

- If the implementation differs from the spec, document why in the plan
- Update the spec if the change is permanent

### Step 8: Backend Documentation (when applicable)

- A task that adds or changes a model backend is **not complete** without its doc page:
  usage, dependencies, credentials, hardware requirements, known limits

### Step 9: Code Commit

- Stage related changes only
- Conventional Commits format (see below)

### Step 10: Plan Update

- Mark the task `[x]` in plan.md
- Note blockers or follow-ups

### Step 11: Plan Commit

- Commit plan.md separately: `chore(plan): mark <task> complete [track-id]`

## Commit Convention

**Conventional Commits.**

```
<type>(<scope>): <description> [track-id]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`

Examples:

```
feat(backends): add faster-whisper backend [core-pipeline]
fix(normalize): preserve segment timestamps when merging [core-pipeline]
docs(backends): document whisper-api credentials and limits [core-pipeline]
chore(plan): mark task 2.3 complete [core-pipeline]
```

Scope is the module or area (`backends`, `cli`, `normalize`, `manifest`, `docker`, `plan`).

> **Note:** this project is not yet a git repository. Run `git init` before the first commit.

## Code Review

**Policy: Optional — self-review is acceptable.**

Before committing, self-review the diff for:

- Engine-specific logic leaking outside a backend module (violates modularity)
- Unrecorded config or model versions (violates reproducibility)
- A new or changed backend without its doc page

Request another reviewer when a change alters the backend interface or the transcript schema.

## Verification Checkpoints

**Policy: Manual verification at track completion only.**

Phases and tasks run through without pausing for sign-off. Verification happens once, when the
track is complete:

1. All tasks in plan.md are `[x]`
2. Full suite passes: `pytest`
3. Lint and format clean: `ruff check . && ruff format --check .`
4. All backends touched by the track have current documentation
5. A real run over sample audio produces the expected output
6. Manual sign-off recorded in the track's plan.md

An exception applies regardless of the policy: pause for approval before proceeding on

- Changes to the **backend interface**
- Changes to the **transcript output schema**
- Changes to the **run manifest format**

These are the project's contracts; breaking them breaks every backend and consumer.

## Quality Assurance Gates

Before a track is marked complete:

| Gate           | Requirement                                     | Command                       |
| -------------- | ----------------------------------------------- | ----------------------------- |
| 1. Tests       | Suite passing                                   | `pytest`                      |
| 2. Complex logic | Interface, normalization, CLI covered by tests | Manual review                 |
| 3. Conformance | New backends pass shared conformance tests      | `pytest -k conformance`       |
| 4. Lint        | No lint errors                                  | `ruff check .`                |
| 5. Format      | Formatted                                       | `ruff format --check .`       |
| 6. Docs        | Every touched backend has a current doc page    | Manual review                 |
| 7. Repro       | Run manifest records backend, version, config   | Manual review                 |

## Development Commands

### Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Run the Pipeline

```bash
.venv/bin/python -m audio_transcriber --input ./audio --backend <name> --out ./out
```

### Pre-Commit Checks

```bash
ruff format . && ruff check --fix . && pytest
```

### Full Validation

```bash
ruff check . && ruff format --check . && pytest
```

> Commands reflect the intended layout; they become real as the first track lands.

## Workflow Diagram

```
┌─────────────┐
│ Select Task │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Mark [~]    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Implement   │◄── test first for complex logic / bug fixes
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Test        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Refactor    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Lint/Format │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Doc backend │ (if a backend changed)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Commit Code │ Conventional Commits
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Mark [x]    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Commit Plan │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Track complete?  │
│ → manual verify  │
└──────────────────┘
```
