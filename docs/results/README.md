# Result Snapshots

These are small CSV snapshots from the latest solver comparisons that are worth
keeping in git. Large checkpoints, figures, detailed logs, and regenerable bulk
outputs belong in the ignored `outputs/` directory.

- `anchor_solver_weighted_ppo_curriculum_10h_combined_stage_summary.csv`:
  combined stage summary for the long PPO curriculum run.
- `anchor_solver_curriculum_final_vs_graph_shortest_80cases_20260628_summary.csv`:
  bucket-level comparison between the final curriculum checkpoint and the
  graph-shortest baseline on 80 matched cases.
- `anchor_solver_curriculum_final_vs_graph_shortest_80cases_20260628_paired.csv`:
  paired-case comparison details for the same 80-case evaluation.
- `new_algorithms_simple_benchmark_summary.csv` and
  `new_algorithms_simple_benchmark_detail.csv`: fixed-constant smoke benchmark
  comparing graph scaffold, visibility branching, and visibility SDP on 12
  matched 16-24 anchor cases.
