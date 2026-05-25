"""P2.2 Agent Loop Ablation eval harness.

CLI entry: `python -m app.eval.p2_2_agent_ablation.run_eval --output output/results.jsonl`

Schemas:
  - RunSpec: one row in the experiment matrix (model × mode × query × turn × run).
  - results.jsonl record: one row per executed RunSpec; schema validated by
    `single_run.validate_record_schema`.
"""
