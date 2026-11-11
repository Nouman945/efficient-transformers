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

Results of task 2026-11-11-f (read from code of nearby versions, not the deployed build):

- The exact serving build from the bug report is not on this machine. A newer disaggregated
  serving image and the SDK's serving patches (an older and a newer version) are.
- The serving stack uses this repository only to export and compile. Its KV hand-off is its
  own code. It already hands off only the joining slot's row and the prompt's positions,
  inputs only, through shared memory. So the hand-off change made in this repository is not
  a fix for the bug report. It only speeds up the example here.
- The first-token path there is: proxy sends the request to a prefill worker and waits for
  the full reply, then sends it to a decode worker, which fetches the KV, binds it and runs
  the first step. Two HTTP round trips in a row, a new connection per hop. No long sleep
  was seen in the code.
- Shared-memory segments that a decode worker never fetches are freed only when the
  hand-off service shuts down. Possible link to the memory growth. Not checked on a
  running system.

Open questions the current and next tasks are meant to answer:

1. Does the example-level change give the same answers as the old code when slots refill?
2. Does memory grow per request with the old hand-off, and does the new one stop it?
3. Can prefill chunks overlap across pipeline stages, and how much faster is that?
4. How much of the fixed cost in the real serving stack is outside this repository
   (the hand-off between worker processes)? That code is not here.

---

## Current task

**Task id: 2026-11-11-g**

Goal: measure the first-token path on a running disaggregated serving stack on this
machine, with `openai/gpt-oss-20b`, one prefill worker and one decode worker, and split the
time into its hops. This uses the newer serving image that task f found locally, so the
numbers are for that version, not for the deployed one.

### Rules for this task (in addition to the standing rules)

- **Before you start any container, show the user the exact command and wait for a yes.**
  The image was not built by this user. If the user says no, stop and report that.
- Do not pull or build images. Do not touch containers of other users.
- Use only devices that `qaic-util -q` shows as free. Give the container its own name. No
  restart policy. Pick ports that are not in use.
- Find the launch procedure from the launcher and docs inside the image, and from the launch
  command in the bug report (ask the user for its path). Do not guess flags.
- Keep the report's settings where the small model allows: 256-token prefill chunk, decode
  batch size 1, 4 devices per worker. Use context 32768 if the compile for it finishes in
  about an hour; otherwise 8192, and say which. Check disk space first.
- Put the client script and logs under `results-logs/`. Nothing new in tracked files.
- At the end stop and remove your container and confirm the devices are free again.

### Step 1. Bring up 1 prefill + 1 decode + hand-off service + proxy

Report compile time (if any), start-up time, and the versions in use.

### Step 2. First-token time through the proxy

Three prompt sizes: about 100, 500 and 1000 tokens. Temperature 0, a unique prefix per
request, one request at a time, 5 requests per size, streaming. Record time to the first
streamed token. Send one non-streamed request per size to get the server's own
`prompt_tokens`, and report that number, not the target size.

### Step 3. Split the time by hop

For the same three sizes, 5 requests each, call the workers directly in the same order the
proxy does:

1. The prefill worker alone, the way the proxy calls it. Time the full reply.
2. Then the decode worker alone with the same request, streaming. Time to the first token.

Also turn on the per-stage timing switches you listed under question 7 of task f, if they
do not change the request path, and read out what they print for one request per size.

### Report for this task

- Step 1: done or failed, versions, context used, compile and start-up time.
- One table, one row per size: server prompt tokens, chunks, proxy first token (median),
  prefill hop (median), decode hop to first token (median), sum of the two hops.
- A fit of proxy first-token time as fixed + per chunk.
- Any per-stage timing lines that show where the decode hop spends its time before the
  first token.
- Cleanup: container removed, devices free.

---

## Separate task, only when the user asks for it by id

**Task id: 2026-11-11-h**

This is not part of the bug work. It checks a clean branch meant for a pull request to the
public repository: the example speed-up without any debug code.

Use the venv `/local/mnt/workspace/nrasheed/qeff_env_gptoss`. No new compile should be
needed; say so if one starts.

1. `git fetch`, then take two files from the branch `gpt-oss-cb-kv-handoff-once` into the
   working tree without switching branch:
   `QEfficient/generation/_kv_dma_handoff.py` and
   `examples/disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py`.
2. Run H: the example with the six prompts in the usual order, same arguments as run A.
3. Run I: the same with prompt order 4, 5, 0, 1, 2, 3, same arguments as run E.
4. Run the device test
   `tests/transformers/disaggregated/test_gpt_oss_disagg_cb_kv_share_w_hf_fp32.py` if it can
   run here; report pass, fail or why it cannot run.
5. Restore the two files to the branch head and confirm `git diff HEAD` is empty.

Report: tok/sec of H and I; for each prompt, whether H's text equals run A's (`before.txt`)
and whether I's text equals run E's (`run_e.txt`), with the first differing character if
not; the test result. Expected: all texts equal, about 43 tok/sec.
