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

Open questions the current and next tasks are meant to answer:

1. Does the example-level change give the same answers as the old code when slots refill?
2. Does memory grow per request with the old hand-off, and does the new one stop it?
3. Can prefill chunks overlap across pipeline stages, and how much faster is that?
4. How much of the fixed cost in the real serving stack is outside this repository
   (the hand-off between worker processes)? That code is not here.

---

## Current task

**Task id: 2026-11-10-c**

Use the venv `/local/mnt/workspace/nrasheed/qeff_env_gptoss`. No new compile should be
needed in steps 1 and 2. If one starts, say so in the report.

### Step 1. Which example run is right? Run the two differing prompts alone

Run `examples/disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py` with a
single `--prompt` each time, all other arguments default. One prompt means one busy slot
and no refill, so nothing else can disturb it.

| Run | Code | Prompt | Output file |
|---|---|---|---|
| R1 | old (`64b1b410` for the example file and `QEfficient/generation/_kv_dma_handoff.py`) | `Name three primary colors.` | `ref_old_p3.txt` |
| R2 | old | `Define entropy in one line.` | `ref_old_p5.txt` |
| R3 | new (branch head) | `Name three primary colors.` | `ref_new_p3.txt` |
| R4 | new | `Define entropy in one line.` | `ref_new_p5.txt` |

Compare the generated text of each with the text for the same prompt in `before.txt`
(run A) and `after.txt` (run B) from the last task. Leave the tree clean at the branch head.

### Step 2. Does the slot a request sits in change its tokens?

Same model, same prompt, different slot. The tokens must be the same. Run:

```
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --steps 40 --modes every_step prompt_ctx --slot-a 0 --slot-b 1 --out slot01.json
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --steps 40 --modes every_step prompt_ctx --slot-a 3 --slot-b 0 --out slot30.json
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --steps 40 --modes every_step prompt_ctx --slot-a 1 --slot-b 3 --out slot13.json
```

From the three JSON files compare, for mode `every_step` and again for mode `prompt_ctx`:
`tokens_a` across the three runs, and `tokens_b` across the three runs. For any pair that
differs, give the index of the first differing token.

### Step 3. Two timing runs (carried over, not started last time)

```
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --prompt-tokens 5000 --out in1.json
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --prompt-tokens 5000 --prefill-inflight 4 --out in4.json
```

The second run may fail or hang in prefill. If it hangs for more than 5 minutes after any
compile is done, stop it and report that.

### Report for this task

- Step 1: for R1 to R4, one line each: equal to run A's text, equal to run B's text, equal
  to both, or equal to neither. Also whether R1 equals R3 and R2 equals R4.
- Step 2: for each mode, `tokens_a` same or different across the three runs, and the same
  for `tokens_b`, with the first differing index.
- Step 3: for each run, the table rows for `every_step` and `prompt_ctx` (all columns,
  including `rss_mb_per_request`), and whether `tokens_a` of `prompt_ctx` is identical
  between `in1.json` and `in4.json`.
