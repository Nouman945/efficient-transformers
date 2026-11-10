# Server task file

This file is the hand-off between two agents. One agent (remote, no access to this server)
writes it. You, the agent on the server, read it after each `git pull` and do the task
under "Current task". The user photographs your final report and sends it back, so the
report must be short and complete on one screen.

This file is working material. It is not part of any pull request.

## Standing rules (apply to every task)

- Work in `/usr2/nrasheed/efficient-transformers-GPTOSS-120B`, branch `gpt-oss-ttft-breakdown`.
- Do not commit, push, or open pull requests. Do not edit `pyproject.toml`.
- Do not modify, upgrade or uninstall anything in the venv
  `/local/mnt/workspace/nrasheed/qeff_env_latest`.
- Do not edit files under `/usr2/nrasheed/efficient-transformers-latest`.
- Do not change library or example code to make a step pass. If a step fails, stop that
  step, report the error, and continue with the next independent step.
- The login shell is tcsh. Use tcsh syntax in anything you ask the user to run.
- Before a compile, check free disk space. Put large outputs under
  `/local/mnt/workspace/nrasheed`. Stop and report if space is short.
- Before a run, check that enough devices are free with `/opt/qti-aic/tools/qaic-util -q`.
- Reuse the existing Hugging Face cache and `QEFF_HOME` (see `~/.cshrc`). Do not download
  a model that is already cached.
- Do not put customer names, server addresses or internal project names in any file.

## Report format (every task)

End with one block titled `REPORT task <id>`, at most 30 lines, plain text, no tables wider
than 100 characters. Give numbers, not descriptions. For any error give the exception line
and the last 10 lines of the traceback. Say clearly which steps were done, skipped or failed.

---

## The bug this work is for

This repository copy is public. The description here is kept generic on purpose. The full
bug report is confidential and must stay out of git: no customer name, no report numbers,
no report text in any tracked file, commit message or branch name.

**Full details are on this machine, not in this file.** At the start of a task, ask the user
for the path of the bug report if you have not read it in this session, and read it there.
Use it to understand the work. Do not copy from it into the repository.

Generic statement of the bug:

- Model and mode: `openai/gpt-oss-120b`, disaggregated serving (separate prefill and decode
  workers, KV handed from prefill to decode).
- Symptom: every request pays a fixed first-token cost that does not depend on prompt
  length, on top of the per-chunk prefill time.
- Question to answer: where does the fixed part go (scheduling, KV hand-off), and can it be
  reduced?

What has been found so far on this server (public example code, gpt-oss-20b, reduced depth):

- The KV hand-off moves the full compiled context for every slot, whatever the prompt
  length. Its cost grows with context length and slot count.
- Handing off only the joining slot's row, inputs only, and only the prompt's positions
  removed almost all of that cost with identical tokens for 1, 3 and 10 chunk prompts.
  This is mode `prompt_ctx` in `examples/disagg_serving/gpt_oss_ttft_breakdown.py`.
- Re-wiring all rows only when a request joins (mode `all_once`) gives wrong tokens for the
  request that was already running. Do not use it.
- Prefill time is the same per chunk at every prompt length, so prefill has no fixed cost
  of its own in these runs.

Result of task 2026-11-10-b, step 2: with six prompts on four slots, the old and new example
code each give repeatable text, but prompts 3 and 5 differ between them late in the answer.
Speed went from about 32 to about 43 tok/sec. Which version is right is not known yet.
One suspicion: rows that are idle for a step (inputs set to -1) may not be harmless and may
disturb another row, for example the last one.

Results of task 2026-11-10-c:

- The example change at the branch head is wrong when several requests are in flight.
  Single-prompt runs of prompts 3 and 5 give run A's text with both old and new code.
- In `gpt_oss_ttft_breakdown.py`, `prompt_ctx` gives exactly the tokens of `every_step` in
  every run, including with the requests in different slots. So the hand-off itself is not
  shown to be wrong; the join timing in the example differs from upstream (one slot per
  step, with a pending slot idle for a step).
- On the 8-layer model, tokens depend on the slot in both modes. This may be numeric noise
  on a cut-down model. Not seen on the 24-layer model.
- Memory per request: about 257 MB with `every_step`, about 0.08 MB with `prompt_ctx`
  (100-token prompt), about 6 MB at 5000 tokens.
- Four prefill chunks in flight is 3.0x faster but gives wrong tokens. Not usable as is.

Results of task 2026-11-10-d:

- Run C (upstream wiring, new join timing) and run D (new wiring, full-context row) both
  give run B's text. So the wiring change and the prompt-sized row change no text. The
  difference between A and B comes only from when each request joins.
- Run E (old code, prompts reordered) changes the answers of prompts 0 and 1. So upstream
  itself gives different text when the neighbours change. Run A is not a fixed reference.
- Speed comes from the wiring: about 32 tok/sec with upstream wiring, about 43 with the new.

Results of task 2026-11-10-e:

- Two requests can join decode on the same step with the new hand-off (two wire calls, then
  one decode step). `new_in` and `prompt_ctx` give exactly the tokens of `every_step`.
- Runs F and G (new code, prompts reordered) differ only on prompt 3, at the last token. The
  top-2 margins there are 0.27 and 0.97, so the gap between the two logits moved by at least
  1.23 with identical earlier tokens. That is not a near-tie. Decode output depends on slot
  or schedule in upstream code too. Per-row drift or a cross-row effect: not known.

Important limit of all results above: they are from the single-process Python example in
this repository. The deployment in the bug report is a different code path: a vLLM based
disaggregated serving container with separate prefill and decode worker processes, a
request proxy, and a KV hand-off service between them, at decode batch size 1. None of that
code is in this repository. It is not yet shown that anything measured here applies there.

Open questions the current and next tasks are meant to answer:

1. Does the example-level change give the same answers as the old code when slots refill?
2. Does memory grow per request with the old hand-off, and does the new one stop it?
3. Can prefill chunks overlap across pipeline stages, and how much faster is that?
4. How much of the fixed cost in the real serving stack is outside this repository
   (the hand-off between worker processes)? That code is not here.

---

## Current task

**Task id: 2026-11-11-f**

Read-only discovery. No compile, no model run, no device use. Do not pull or download
images or packages. Do not start a serving container. Do not copy any source code you find
into this repository or into the report; give paths, names and short answers only.

Goal: find out whether the code of the real serving stack from the bug report is on this
machine, and if so, how the first token is produced there. Read the bug report first (ask
the user for its path) to get the exact image name, container name, ports and flags.

### Step 1. Is the serving stack here?

Look for the disaggregated serving package and image named in the bug report:

- Is docker (or podman) installed, and can this user run it? List local images whose name
  matches `vllm`, `disagg` or `cloud_ai`. List running containers.
- Search for a directory or Python package with the container name from the report, and for
  a `proxy/server.py` under it. Places: `/opt/qti-aic`, the home directory,
  `/usr2/nrasheed`, `/local/mnt/workspace`, and the site-packages of any venv (read only).
  Keep searches bounded (`-maxdepth`), do not scan network mounts.
- Is a `vllm` package installed in any venv here? Which version, and does it contain QAIC
  specific files (search its tree for `qaic`)?
- If an image is present locally but not unpacked, you may list its files with a container
  that has no devices attached and exits at once. Do not run its entrypoint.

If nothing is found, stop here and report exactly where you looked.

### Step 2. If the code is found, answer these from the code

1. Does it import `QEfficient`, or call `set_data_for_kv_handoff`, `setDataWithSlices` or
   `qaicrt` directly? Name the files.
2. How does KV get from a prefill worker to a decode worker? What does the hand-off service
   do: copy bytes over a socket, shared memory, or pass device buffer handles?
3. How much is handed off per request: the full compiled context, or only the prompt's
   positions? Inputs only, or inputs and retained-state outputs?
4. Are hand-off buffers allocated per request? Where are they freed?
5. What happens, in order, between the proxy receiving a request and the first streamed
   token? Note any fixed waits: `sleep`, poll intervals, timeouts, health checks, retries.
6. What does the prefill capture-length flag from the report control?
7. Is there a switch (flag, environment variable, log level) that prints per-stage timing?

### Report for this task

- Step 1: found or not; image names and tags; package paths; vllm version.
- Step 2: one or two lines per question, with file and function names. Say "not found" where
  the code does not answer it.
- One line: which of the seven answers you are least sure of.
