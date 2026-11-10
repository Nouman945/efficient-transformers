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

## Current task

**Task id: 2026-11-10-a**

### Step 1. Give this checkout its own environment

Background: the active venv has QEfficient installed in editable mode from
`/usr2/nrasheed/efficient-transformers-latest`, which is older code. Scripts run from this
checkout therefore load the old library. Pointing `PYTHONPATH` at this checkout fails with
`ImportError: cannot import name 'GraniteMoeExperts'` because the venv's `transformers` is
older than this checkout needs.

1. If `/usr2/nrasheed/efficient-transformers-latest/QEfficient/generation/_kv_dma_handoff.py.bak`
   exists, copy it back over `_kv_dma_handoff.py` in that folder and say so in the report.
   This is the one allowed edit in that folder.
2. Create a new venv `/local/mnt/workspace/nrasheed/qeff_env_gptoss` with the same Python
   version as `qeff_env_latest`.
3. In the new venv run `pip install -e .` from this checkout.
4. Make `import qaicrt` and `import QAicApi_pb2` work in the new venv. Find how
   `qeff_env_latest` gets them (a `.pth` file, `PYTHONPATH`, or a copy in site-packages)
   and do the same. Do not reinstall or modify the QAIC SDK.
5. Check: `python3 -c "import QEfficient, qaicrt; print(QEfficient.__file__)"` must print a
   path under `efficient-transformers-GPTOSS-120B`.

If step 1 cannot be completed, stop here and report. Steps 2 and 3 need it.

### Step 2. Validate the hand-off change in the example

Use the new venv. Run
`examples/disagg_serving/gpt_oss_disagg_mode_cb_chunking_with_kv_share.py` twice with these
six prompts, each as its own `--prompt` argument, all other arguments default:

1. `Explain quantum computing in simple terms.`
2. `What is the capital of France?`
3. `Explain photosynthesis in one sentence.`
4. `Name three primary colors.`
5. `Why is the sky blue?`
6. `Define entropy in one line.`

- Run A (old behaviour): first check out the example file and
  `QEfficient/generation/_kv_dma_handoff.py` from commit `64b1b410`. Save all output to
  `before.txt`.
- Run B (new behaviour): check out both files from the branch head
  (`git checkout gpt-oss-ttft-breakdown -- <the two files>`). Save all output to `after.txt`.

Leave the working tree clean at the branch head when done (`git status` shows no changes
other than untracked output files).

### Step 3. Two timing runs

Use the new venv. Both runs reuse the same compiled QPCs.

```
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --prompt-tokens 5000 --out in1.json
python3 examples/disagg_serving/gpt_oss_ttft_breakdown.py --num-hidden-layers 8 --mxfp6 --prompt-tokens 5000 --prefill-inflight 4 --out in4.json
```

The second run may fail or hang in prefill. If it hangs for more than 5 minutes after the
compile is done, stop it and report that.

### Report for this task

- Step 1: new venv path, `transformers` version in it, the `QEfficient.__file__` line, and
  whether the `.bak` file was restored.
- Step 2: the `decode steps=... tok/sec=...` line of run A and of run B. For each of the six
  prompts: `same` or `different`. For a different one, the first 150 characters of both
  outputs.
- Step 3: for each run, the printed table rows for modes `every_step` and `prompt_ctx` only
  (all columns), the `first tokens` line, and whether `same_tokens_a` / `same_tokens_b` are
  True for `prompt_ctx`.
