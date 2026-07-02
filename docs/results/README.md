# Result Snapshots

These are compact snapshots from the latest solver comparisons that are worth
keeping in git. Large checkpoints, detailed logs, and regenerable bulk outputs
belong in the ignored `outputs/` directory.

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
- `visibility_stage1_high_sample_sweep_*` and
  `visibility_stage2_local_refinement_*`: two-stage visibility-parameter sweep
  leaderboards and bucket summaries. Candidates were ranked by a tail-weighted
  score: p95 max offset plus smaller median/max-offset terms and folded/error
  penalties.
- `visibility_two_stage_selected_params.json`: selected tuned parameters from
  the two-stage sweep.
- `visibility_tuned_validation_summary.csv` and
  `visibility_tuned_validation_detail.csv`: fresh 90-case validation comparing
  graph scaffold, tuned visibility branching, and tuned visibility SDP.
- `visibility_tuned_validation_worst_cases_all_methods.png` and
  `visibility_tuned_validation_worst_cases_all_methods_metrics.csv`: worst-case
  visualization and metrics for the selected methods, cross-solved on each
  method's hardest validation case.
