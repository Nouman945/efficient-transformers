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

Open questions the current and next tasks are meant to answer:

1. Does the example-level change give the same answers as the old code when slots refill?
2. Does memory grow per request with the old hand-off, and does the new one stop it?
3. Can prefill chunks overlap across pipeline stages, and how much faster is that?
4. How much of the fixed cost in the real serving stack is outside this repository
   (the hand-off between worker processes)? That code is not here.

---

## Current task

**Task id: 2026-11-10-d**

Goal: find which part of the example change produces the wrong text for prompts 3 and 5.
The change has three parts: (1) slots join one per step, (2) only the joining slot's row is
wired, inputs only, and rows stay on the device, (3) the row covers only the prompt's
positions. Two debug switches were added to the example to separate them.

Use the venv `/local/mnt/workspace/nrasheed/qeff_env_gptoss`. Use the example at the branch
head. No new compile should be needed; say so if one starts.

Run `examples/disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py` with the
same six prompts as before, each as its own `--prompt`, in the same order:

| Run | Extra argument | Output file | What it isolates |
|---|---|---|---|
| C | `--debug-full-rewire` | `run_c.txt` | part (1) alone: upstream wiring, new join timing |
| D | `--debug-full-ctx-row` | `run_d.txt` | parts (1) and (2), without (3) |

Compare the generated text of all six prompts in C and in D with `before.txt` (run A) and
`after.txt` (run B).

If you have time left, also run the six prompts in this order with **old** code
(`64b1b410` for the example file and `QEfficient/generation/_kv_dma_handoff.py`), output
`run_e.txt`: prompts 4, 5, 0, 1, 2, 3 (so "Why is the sky blue?" first). Then restore the
branch head. Compare each prompt's text with run A. This checks whether upstream itself
changes an answer when the neighbours change.

### Report for this task

- Run C: tok/sec line; for each of the six prompts: equal to A, equal to B, or neither.
- Run D: the same.
- Run E, if done: for each prompt, equal to A or not, with the first differing character
  index.
- One line: which of parts (1), (2), (3) the evidence points to.
