# AGENTS.md

RLinf is a Hydra-configured, Ray-orchestrated RL stack. The real entrypoints are `examples/embodiment/train_embodied_agent.py`, `examples/embodiment/train_async.py`, `examples/reasoning/main_grpo.py`, `examples/sft/train_vla_sft.py`, and `examples/sft/train_vlm_sft.py`.

## Use These Commands

- Preferred local setup is `uv`, not ad-hoc `pip install`.
- Fast unit-test env matches CI:
  - `uv venv && source .venv/bin/activate`
  - `UV_TORCH_BACKEND=auto uv sync`
  - `uv pip install transformers`
- Repo install script is the source of truth for heavyweight stacks:
  - `bash requirements/install.sh embodied --model <model> --env <env>`
  - `bash requirements/install.sh agentic`
  - `bash requirements/install.sh docs`
- `requirements/install.sh` reuses `.venv` only if its Python minor version matches `3.11`; otherwise it recreates it.

## Focused Verification

- Lint/format: `pre-commit run --all-files`
- Install commit-msg hook before making commits that must pass repo hooks: `pre-commit install --hook-type commit-msg`
- Unit tests: `export PYTHONPATH=$(pwd):$(pwd)/tests/unit_tests && pytest tests/unit_tests`
- Scheduler doctests are separate: `pytest --doctest-modules rlinf/scheduler`
- Package structure check: `python tests/unit_tests/check_missing_init.py`
- Import smoke test used in packaging CI: `python tests/unit_tests/test_import_rlinf_package.py`

## E2E Shortcuts

- E2E scripts expect `REPO_PATH=$(pwd)`.
- Embodied train/e2e entrypoints wire EGL defaults and simulator env vars in shell scripts; prefer the existing wrappers over inventing new commands.
- Single embodied config: `export REPO_PATH=$(pwd) && bash tests/e2e_tests/embodied/run.sh <config>`
- Async embodied config: `export REPO_PATH=$(pwd) && bash tests/e2e_tests/embodied/run_async.sh <config>`
- Reasoning config: `export REPO_PATH=$(pwd) && bash tests/e2e_tests/reasoning/run.sh <config>`
- SFT config: `export REPO_PATH=$(pwd) && bash tests/e2e_tests/sft/run_vla_sft.sh <config>`

## Architecture Boundaries

- `rlinf/config.py` is the central validation and registration choke point: supported task types, model enums, rollout backends, and config normalization all pass through it.
- Embodied flow is `Cluster` -> `HybridComponentPlacement` -> actor/rollout/env/reward workers -> `EmbodiedRunner`.
- Reasoning flow is `Cluster` -> `ModelParallelComponentPlacement` -> rollout/inference/reward/actor/critic/scheduler workers -> `ReasoningRunner`.
- CI treats these paths as shared infra with broad blast radius: `rlinf/algorithms`, `rlinf/scheduler`, `rlinf/utils`, `rlinf/data`, `rlinf/hybrid_engines`, `rlinf/workers/rollout`, `rlinf/config.py`.
- CI treats embodied work as mostly `rlinf/envs`, `rlinf/models/embodiment`, embodied actor/env workers, `rlinf/runners/*embodied*.py`, and `examples/embodiment`.
- CI treats agent/reasoning work as `rlinf/agents`, megatron/critic/inference workers, `rlinf/runners/{agent_runner,reasoning_runner,coding_online_rl_runner}.py`, and `examples/agent` or `examples/reasoning`.

## Repo Conventions That Matter

- Config YAML is expected to stay static. Do not add computed values or silently overwrite user-settable config fields in code.
- In workers, use `self.log_info`, `self.log_warning`, `self.log_error`; outside workers use `rlinf.utils.logging.get_logger()`. Avoid `print` for normal logging.
- Google-style docstrings and type hints are expected on public APIs. Ruff currently enforces docstring rules mainly under `rlinf/scheduler/**`, so do not mistake limited enforcement for repo policy.
- User-facing behavior changes are expected to ship with tests and docs.

## Ray And Multi-Node Gotchas

- Multi-node scheduling depends on `RLINF_NODE_RANK` being set before `ray start` on each node.
- `rlinf/scheduler/cluster/config.py` treats `node_groups[*].node_ranks` as those `RLINF_NODE_RANK` values; `env_configs` can also inject per-node env vars and Python interpreter paths.
- `ray_utils/start_ray.sh` uses `RANK`, not `RLINF_NODE_RANK`; do not assume the helper script covers the scheduler requirement by itself.

## CI / PR Workflow

- PR CI intentionally fails unless the PR has the `run-ci` label.
- Commits must be signed off: use `git commit -s`.
- Commit and PR titles are expected to follow Conventional Commits: `<type>(<scope>): <description>`.

## Core principle: repository-grounded only
When explaining code, answering questions about the project, or generating/modifying code, always ground the response in the repository contents first.

Do not guess based on general knowledge when the repository can be checked.
Do not invent functions, classes, files, APIs, config keys, or behaviors that are not supported by the codebase.

If the repository does not provide enough evidence, explicitly say so.
Use phrases like:
- "I could not find evidence for this in the repository."
- "This is a likely interpretation, but I could not verify it from the current source."
- "I need to inspect more files before making this change confidently."

## Required repo-reading behavior
Before explaining a module, function, class, workflow, or architecture:
1. Read the directly relevant file(s).
2. Also inspect nearby evidence when relevant, such as:
   - imports
   - base classes
   - callers / call sites
   - interfaces and type definitions
   - config files
   - tests
   - related scheduler / worker / engine / launcher files
3. Prefer project-local patterns over generic best practices.

Before generating or modifying code:
1. Inspect the target file and surrounding code.
2. Inspect existing implementations of similar logic elsewhere in the repo.
3. Check names, signatures, control flow, error handling, logging style, typing style, and config conventions.
4. Reuse existing project patterns whenever possible.

## Evidence standard for explanations
When explaining code, every important claim should be traceable to the source.

Always cite concrete repository locations using relative paths and line numbers:
- `path/to/file.py:123`
- `path/to/file.py:123-148`

Do not use vague statements like:
- "This probably does..."
- "Usually this kind of code means..."
unless you clearly label them as inference.

Distinguish clearly between:
- what the code explicitly does
- what can be inferred from surrounding code
- what is still uncertain

## Evidence standard for generated code
Any generated code must be consistent with the existing repository.

Requirements:
- Do not introduce new dependencies unless the user explicitly asks for them or the repo already uses them.
- Do not call nonexistent functions or import nonexistent modules.
- Do not create new abstractions if an existing project abstraction already fits.
- Match the project's existing naming, typing, logging, config, and error-handling style.
- If a requested feature cannot be implemented cleanly with current repo structures, say so explicitly before proposing a workaround.

Before finalizing generated code, verify:
- referenced files exist
- referenced symbols exist
- imports are plausible in this repo
- the change matches existing architecture
- the change does not contradict config, tests, or current call patterns

## No fake certainty
Never present an unverified assumption as a fact.

If evidence is incomplete:
- say exactly what was checked
- say what is still missing
- say what additional files should be inspected

Prefer:
- "From `rlinf/.../worker.py:88-121`, this function queues tasks into ..."
over:
- "This function is responsible for the whole scheduling pipeline."

## File reference format
Always use repository-relative paths.

Allowed:
- `rlinf/scheduler/worker/worker.py`
- `rlinf/scheduler/worker/worker.py:99`
- `rlinf/scheduler/worker/worker.py:99-120`

Not allowed:
- absolute Windows paths like `/E:/...`
- local absolute file URIs
- markdown links to local files such as `[worker.py](/E:/.../worker.py#L99)`

When citing code locations, use plain text in `path:line` or `path:start-end` format.

## Output style for code explanations
When the user asks for an explanation:
1. Start from the exact file/function/class requested.
2. Explain only what is supported by the code.
3. Cite the relevant source locations.
4. If needed, then expand to upstream/downstream dependencies.
5. If there are multiple possible interpretations, state which one is best supported by the repository.

## Output style for code generation
When asked to generate code:
1. First infer the intended integration point from the existing codebase.
2. Base the implementation on nearby repo patterns.
3. Mention the files/functions used as evidence.
4. If confidence is limited, say what part is grounded and what part is a best-effort proposal.

## Preference for local consistency over generic elegance
Prefer code that matches the current repository style over code that is more abstract, more modern, or more elegant in the abstract.

Do not refactor unrelated code unless the user asks for it.

## Conflict rule
If the user's request conflicts with the repository's current structure, constraints, or naming conventions, point out the conflict explicitly instead of silently forcing a generic solution.

## Example expected behavior
Good:
- "The worker pulls tasks from X and forwards them to Y based on `rlinf/scheduler/worker/worker.py:99-140` and `rlinf/scheduler/dispatcher.py:42-68`."
- "I did not find any retry logic in the current worker path. I checked `rlinf/scheduler/worker/worker.py:99-180` and `rlinf/scheduler/base.py:10-72`."

Bad:
- "This should use a retry queue."  
- "The framework probably has a central dispatcher for fault tolerance."  
- generating code that references symbols not found in the repo

## Default behavior
When in doubt:
- inspect more source files first
- cite the repo
- avoid guessing
- be explicit about uncertainty