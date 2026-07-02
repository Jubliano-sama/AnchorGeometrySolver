# Long Training Memory

Objective: train a significantly bigger / better anchor distance-completion model for up to 9 hours, using concurrent experiments, and only stop near the time limit unless a model is proven better across the tracked metrics.

Start time from tool: 2026-06-27T02:44:11.5711057+02:00
Nine-hour deadline: 2026-06-27T11:44:11+02:00

Current best checkpoint:
- `outputs/anchor_solver_ml_distance_completion_fair_diag_cuda.pt`
- Architecture: hidden=144, layers=5, batch=96, 8 epochs x 120 steps.
- Active cap sweep recommendation: weak-polish with 5 closest predicted missing distances per anchor is best for grid p95; office tail remains bad.

Baseline fair eval summary from `anchor_solver_ml_distance_completion_fair_diag_cuda_metrics.csv`:
- Grid ML weak: median max offset 0.132 m, p90 0.428 m, under1m 97.1%.
- Grid ML pure: median max offset 0.133 m, p90 0.545 m, under1m 94.3%.
- Random ML weak: median max offset 0.189 m, p90 8.083 m, under1m 68.9%.
- Random ML pure: median max offset 0.191 m, p90 7.974 m, under1m 68.9%.

Baseline cap sweep summary from `anchor_solver_ml_closest_cap_sweep_summary.csv`:
- Random 16-32: p95 about 0.277 m, under1m 100%, mostly insensitive to cap.
- Grid >=16: weak-polish best p95 0.708 m at cap=5, under1m 96.9%.
- Office >=16: weak-polish best p95 9.317 m at cap=6, under1m 76.6%; still bad.

Initial concurrent experiments:
- `bigA_h224_l8`: hidden=224, layers=8, batch=32, lr=0.0015, weight_decay=1e-4, edm=0.035, cap=5 eval, long CUDA run.
- `bigB_h192_l10`: hidden=192, layers=10, batch=32, lr=0.0016, weight_decay=1.5e-4, edm=0.045, cap=5 eval, long CUDA run.

Progress log:
- 2026-06-27T02:44+02: Start. No Python training process running. GPU available, RTX 3090 with display memory around 5.5 GB used.
- 2026-06-27T02:45+02: Attempted to launch two long concurrent CUDA experiments (`bigA_h224_l8`, `bigB_h192_l10`). Approval reviewer rejected the escalation because long concurrent GPU jobs need explicit user approval in this session. Do not attempt workaround execution; ask user for explicit approval or continue with non-running setup work.
- 2026-06-27T02:48+02: User explicitly approved GPU use. Helper scripts compile. GPU state before launch: about 5.8 GB used / 24.6 GB total. Launching `bigA_h224_l8` and `bigB_h192_l10` via `work/start_long_training_experiments.ps1`.
- 2026-06-27T02:49+02: Both experiments launched and logging. Observed Python worker processes present. GPU after launch: ~12.3 GB / 24.6 GB used, ~93% utilization, 58 C. Initial losses: bigA step1 loss=0.71765; bigB step1 loss=0.61888.
- 2026-06-27T02:51+02: bigA and bigB stable through step 100/7200. Throughput ~26 cases/s each, GPU ~12.1 GB used, ~84% util, 61 C. Losses in mid/high 0.3s, normal early training.
- 2026-06-27T02:55+02: bigA epoch1 mean=0.39284, still in epoch2 at step220 with recent loss=0.19546. bigB epoch1 mean=0.40875, epoch2 mean=0.34734. Stderr clean. Continue both.
- 2026-06-27T02:59+02: bigA epoch means 0.39284 -> 0.34107 -> 0.29520, in epoch4 step460. bigB epoch means 0.40875 -> 0.34734 -> 0.30362, in epoch4 step460. GPU ~12.1 GB used, ~89% util, 63 C. Continue both.
- 2026-06-27T03:04+02: bigA epoch means through 6: 0.39284, 0.34107, 0.29520, 0.29800, 0.28164, 0.27301. bigB through 6: 0.40875, 0.34734, 0.30362, 0.28397, 0.27507, 0.27315. bigB logged strong epoch7 batch loss=0.14983. GPU stable ~12.1 GB, 63 C.
- 03:05: user explicitly approved GPU use outside sandbox. bigA completed epoch 7 mean_loss=0.27266; bigB completed epoch 7 mean_loss=0.25141. GPU monitor: 86% util, 12341/24576 MiB, 63C, 202.87W.
- 03:07: added work/anchor_solver_fixed_holdout_compare.py and py_compile passed. It freezes layouts and noisy measured graph batches, then evaluates multiple checkpoints on identical data with detail/summary/gate CSVs and a comparison PNG.
- 03:08: fixed-heldout smoke test passed on CUDA with 2 cases/bucket. Outputs: outputs/anchor_solver_fixed_holdout_smoke_detail.csv, _summary.csv, .png. No code errors; only matplotlib font fallback warnings.
- 03:09: both training jobs reached epoch 9, step 1000/7200. bigA epoch8 mean_loss=0.26717; bigB epoch8 mean_loss=0.26132. GPU monitor: 93% util, 12061/24576 MiB, 65C, 215.29W.
- 03:10: launched third concurrent CUDA experiment bigC_h256_l6_edm07 (hidden=256, layers=6, edm_weight=0.07, lr=0.00125, predicted_sigma=0.50, slope=0.80, cap=5, seed=2026062703). New python process started at 03:09:33; logs created in outputs/anchor_solver_ml_distance_completion_bigC_h256_l6_edm07.train.*.log.
- 03:10: bigC running cleanly: step 20/7200 loss=0.49411, cases_per_s=19.0. No stderr. GPU after three jobs: 96% util, 15349/24576 MiB, 63C, 225.21W.
- 03:12: A/B now around epoch 10 step 1140/7200 after adding bigC; throughput dipped to ~26.4 cases/s but all runs healthy. bigA epoch9 mean=0.26054; bigB epoch9 mean=0.26155. bigC reached step80/7200, loss=0.59015. GPU 96% util, 15365/24576 MiB, 64C, 201.16W.
- 03:14: updated work/anchor_solver_ml_distance_completion.py for future runs: optional --checkpoint-every-epochs saves <prefix>_best_loss.pt on improved epoch mean and <prefix>_latest.pt at the requested interval. py_compile passed. Existing A/B/C processes are unaffected.
- 03:16: all stderr logs are 0 bytes. A/B reached epoch 11 step1280; bigA epoch10 mean=0.26005, bigB epoch10 mean=0.26131. bigC completed epoch1 mean=0.39370 and reached step220. GPU 98% util, 15013/24576 MiB, 63C, 218.43W.
- 03:18: A/B completed epoch11. bigA epoch11 mean=0.25869; bigB epoch11 mean=0.25459 (best recent). bigC epoch2 mean=0.32207, step280. No final .pt checkpoints yet. GPU 97% util, 15202/24576 MiB, 64C, 214.30W.
- 03:20: launched fourth CUDA experiment bigD_h176_l5_fast (hidden=176, layers=5, batch=48, lr=0.002, edm_weight=0.02, cap=4, checkpoint_every_epochs=2, seed=2026062704). New python started at 03:19:38. GPU after launch: 98% util, 17962/24576 MiB, 65C, 225.16W.
- 03:21: bigD startup OK: step1/8640 loss=0.73734, no stderr. A/B completed epoch12: bigA mean=0.26303, bigB mean=0.25147. A/B throughput now ~24.5 cases/s with four jobs. GPU 96% util, 17984/24576 MiB, 64C, 215.42W.
- 03:23: four-job aggregate still useful. A/B at epoch13 step1500; bigA epoch12 mean=0.26303, bigB epoch12 mean=0.25147. bigC epoch3 mean=0.30743 and in epoch4. bigD in epoch1 step40, loss=0.42489. No bigD checkpoint yet. GPU 95% util, 17960/24576 MiB, 63C, 194.82W.
- 03:24: added work/monitor_long_training.py; py_compile passed. Current monitor: bigA step1520 epoch13 loss0.336 cps23.6 best_epoch_mean0.259; bigB step1520 epoch13 loss0.280 cps23.7 best0.251; bigC step460 epoch4 loss0.243 cps18.3 best0.307; bigD step60 epoch1 loss0.374 cps15.1.
- 03:25: monitor: bigA step1560 epoch13 done loss0.261 best_epoch_mean0.259; bigB step1560 epoch13 done loss0.238 best0.251; bigC step500 epoch5 inner20 loss0.301, completed epoch4 mean=0.268 (sharp drop from 0.307); bigD step100 epoch1 loss0.299. GPU 96% util, 18040/24576 MiB, 63C, 204.43W.
- 03:27: bigD completed epoch1 mean=0.38799 last_loss=0.23445 and successfully wrote outputs/anchor_solver_ml_distance_completion_bigD_h176_l5_fast_best_loss.pt. Monitor: bigA/bigB step1620 epoch14, bigC step560 epoch5 loss0.227. GPU 97% util, 17938/24576 MiB, 64C, 193.24W.
- 03:29: monitor: bigA epoch14 mean worsened to 0.283 (best still0.259); bigB epoch14 mean=0.255 (best0.251); bigC epoch5 mean=0.267 (best0.267), step620; bigD epoch2 step160 loss0.308, only best_loss checkpoint from epoch1 so far. All stderr logs 0 bytes. GPU 98% util, 17950/24576 MiB, 65C, 207.94W.
- 03:30: monitor: bigA step1720 epoch15 loss0.230 best0.259; bigB step1720 epoch15 loss0.294 best0.251; bigC step660 epoch6 loss0.233 best0.267; bigD step200 epoch2 loss0.309 best0.388. GPU 96% util, 18017/24576 MiB, 63C, 205.02W.
- 03:32: bigD completed epoch2 mean=0.30854, wrote/updated both best_loss and latest checkpoints. Monitor: bigA step1780 epoch15 loss0.201 best0.259; bigB step1780 epoch15 loss0.281 best0.251; bigC step720 epoch6 done loss0.185, best_epoch_mean0.267; bigD best_epoch_mean0.309. GPU 96% util, 18119/24576 MiB, 64C, 219.14W.
- 03:35: monitor: bigA step1860 epoch16 loss0.267 last_epoch_mean0.260 best0.259; bigB step1860 epoch16 loss0.339 last_epoch_mean0.259 best0.251; bigC step820 epoch7 loss0.327 best0.267; bigD step300 epoch3 loss0.223 best0.309. GPU 96% util, 18145/24576 MiB, 63C, 198.23W.
- 03:39: user requested fold-kick and repel-anneal solver variants. Started standalone script work/anchor_solver_fold_rescue_experiment.py comparing distance-only, production-priors, fold-kick, repel-anneal, and fold-kick+repel on fair generated cases.
- 03:47: updated fold-kick experiment per user: pinned anchor is rng.choice per folded cluster; fold-kick now runs cycles*trials attempts and returns immediately when a solved candidate has zero folded clusters, otherwise keeps best candidate by close-pair/fold-aware score.
- 03:52: implemented fold-kick-frame-anneal in work/anchor_solver_fold_rescue_experiment.py: arbitrary fixed-frame parameterization, random survivor pinned, random peer mirrored across center and pinned, rest optimized; then all pins released and all anchors polished. Integrated into metrics/figure as sixth method.
- 03:54: fold-kick-frame-anneal smoke passed. On 1 case/bucket, frame-anneal unfolded the grid case (folded_rate 0 vs production folded_rate 1) and improved grid max offset from production 10.998m to 6.140m, but RMSE remained high at 0.906m with only 25 iterations. Random case solved to 0.142m like fold-kick; office not improved.
- 03:56: launched background PowerShell job 1 for fold-rescue mini comparison: work/anchor_solver_fold_rescue_experiment.py --cases-per-bucket 3 --seed-count 8 --iterations 60 --fold-threshold 1.65 --prefix anchor_solver_fold_rescue_frame_mini. Logs: outputs/anchor_solver_fold_rescue_frame_mini.run.log and .run.err.log.
- 03:58: background fold-rescue mini job did not produce output/persist; added --buckets option to fold rescue script and dynamic figure bucket rows so grid-only focused runs are possible.
- 04:00: grid3 fold-rescue comparison completed. Outputs prefix anchor_solver_fold_rescue_frame_grid3. Summary: production p95=13.475m folded_rate=1.0 RMSE=0.179m; fold-kick p95=15.982m folded_rate=0.667; frame-anneal p95=15.345m folded_rate=1.0 RMSE=0.504m. Frame helped individual case0 (11.05m -> 9.60m and fewer close pairs) but not enough. User requested graph-aware shortest-path scaffold springs next.
- 04:02: implemented graph-shortest-scaffold in fold rescue experiment: non-measured finite shortest-path graph springs up to 4 hops, sigma grows with path length and hop count; solve known+scaffold+spacing, then drop scaffold and polish on measured ranges only. Integrated as graph-shortest-scaffold method.
- 04:04: graph-shortest-scaffold smoke was weak (grid p95 10.863m, RMSE 1.035m, folded_rate 1.0). Adjusted method: max_hops=3, relative_sigma=0.30, hop_sigma=0.85, use production priors during scaffold stage, add classical MDS seed from known+scaffold, and shorter scaffold-stage iteration.
- 04:05: adjusted graph-shortest-scaffold smoke v2 was excellent on one grid case: production-priors max_offset=6.830m folded_rate=1.0 RMSE=0.0253m; graph-shortest-scaffold max_offset=0.204m folded_rate=0 RMSE=0.0196m. Launching grid3 comparison next.
- 04:09: graph-shortest-scaffold grid3 completed. Summary: graph scaffold median max_offset=0.284m, p95=5.028m, under1m=0.667, folded_rate=0, RMSE=0.0379m. Production-priors median=13.171m, p95=13.475m, under1m=0, folded_rate=1, RMSE=0.1793m. This is the strongest non-ML solver result so far but one of three grid cases remains a shape failure.
- 04:12: added work/anchor_solver_graph_scaffold_sweep.py to sweep max_hops, relative_sigma, hop_sigma on fixed generated grid cases, evaluating production and graph-shortest-scaffold only with detail/summary CSV and PNG.
- 04:20: graph scaffold sweep grid4 completed despite command timeout after writing outputs. Outputs prefix anchor_solver_graph_scaffold_sweep_grid4. All 12 tested settings achieved p95˜0.215m, median˜0.177m, under1m=1.0, folded_rate=0, RMSE˜0.036m. Production on same 4 cases: p95=14.290m, median=10.745m, under1m=0, folded_rate=1.0. Added lean production-vs-graph evaluator work/anchor_solver_graph_scaffold_eval.py.
- 04:36: ML monitor after eval: bigA step3720 epoch31 best_epoch_mean=0.247; bigB step3720 epoch31 best=0.243; bigC step2680 epoch23 best=0.250; bigD step1560 epoch13 best=0.228 with checkpoint outputs/anchor_solver_ml_distance_completion_bigD_h176_l5_fast_best_loss.pt. GPU healthy 97%, 17821/24576 MiB, 63C.
- 04:36: graph scaffold broad eval rgo4 completed despite timeout after writing outputs. Outputs prefix anchor_solver_graph_scaffold_eval_rgo4. Summary: Random graph p95=0.208m vs production 14.309m, under1m 1.0 vs 0.25. Grid graph p95=8.388m vs production 23.078m, median 0.455m vs 16.806m, under1m 0.75 vs 0.0. Office graph p95=6.356m vs production 10.055m, under1m 0.5 vs 0.25. Graph scaffold helps strongly but still has low-RMSE global ambiguity failures in office and one grid case.
- 04:37: evaluated BigD best_loss checkpoint on fixed 12/bucket heldout against baseline. BigD is not better: gate_checks=48 failed=42. Examples: grid p95 worse 0.721 vs 0.631 completed, office p95 worse 6.273 vs 5.430, missing MAE worse. Outputs prefix anchor_solver_fixed_holdout_bigD_best_e13_small.
- 04:38: added --init-checkpoint to work/anchor_solver_ml_distance_completion.py so a model can be initialized from an existing checkpoint before continuing training on freshly generated cases.
- 04:38: launched fifth CUDA run finetuneE_baseline_lr5e4 initialized from outputs/anchor_solver_ml_distance_completion_fair_diag_cuda.pt. Args: hidden=144 layers=5 batch=64 lr=0.0005 wd=0.00008 cap=4 checkpoint_every=2 epochs=80. GPU after launch: 98% util, 21003/24576 MiB, 63C.
- 04:41: monitor after 5-run training: bigA step3900 epoch33 best0.247; bigB step3900 epoch33 best0.243; bigC step2860 epoch24 best0.250; bigD step1660 epoch14 best0.228; finetuneE step40 epoch1 loss0.223. All stderr 0. GPU 97% util, 21003/24576 MiB, 62C.
- 04:44: monitor: bigA step4000 epoch34 best0.247; bigB step4000 epoch34 best0.243; bigC step2960 epoch25 best0.250; bigD step1740 epoch15 best0.228 with recent batch loss0.181; finetuneE step80 epoch1 loss0.222, cps14.2. GPU 98%, 21003/24576 MiB, 63C.
- 04:49: fixed p95 load_model torch.load adjusted to weights_only=False for local checkpoints because init_checkpoint Path in args breaks weights_only=True. BigE epoch1 fixed-heldout eval prefix anchor_solver_fixed_holdout_finetuneE_e1_small: not better, gate_checks=48 failed=43. Grid p95 worse 0.655/0.670 vs baseline 0.631/0.571, office p95 worse 6.427/6.114 vs 5.430/5.250, random about equal but missing MAE worse. Continue training, do not crown E yet.

## 2026-06-27 post-compaction monitor
- BigE finetune reached epoch 2 mean_loss=0.21771 with best checkpoint updated; live GPU ~21GB/24GB, util 96%.
- Graph shortest-path scaffold smoke/eval files are present; small 4-per-bucket eval: random p95 0.208m vs production 14.309m; grid p95 8.388m vs 23.078m; office p95 6.356m vs 10.055m. Next: broader graph eval and worst-case inspection.

## Graph scaffold wider eval launch
- Started PID 65636: anchor_solver_graph_scaffold_eval_rgo8_h2, 8 cases per bucket, seed_count=8, iterations=60, max_hops=2, relative_sigma=0.36, hop_sigma=0.65.

## BigE epoch 2 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e2_small on 8 cases/bucket baseline vs BigE best epoch2.
- Result: random essentially tied/slightly better missing MAE; office p95 improved (completed 12.094m vs baseline 14.538m; weak polish 12.524m vs 15.233m); grid p95 worsened (completed 1.012m vs 0.636m). Gate checks 48 failed 27, so not a clean replacement yet.

## Graph scaffold verify
- Foreground random2 verify completed: graph-shortest scaffold p95=0.197m median=0.163m under1m=1.0 folded=0; production p95=10.962m median=8.344m under1m=0 folded=0.5. Background rgo8_h2 still running/being monitored.

## Graph scaffold GO3 fast eval
- Ran anchor_solver_graph_scaffold_eval_go3_fast: grid+office, 3 cases each, seed_count=4, iterations=30, max_hops=2.
- Grid graph p95=0.223m median=0.215m under1m=1.0 folded=0 RMSE=0.0353m; production p95=22.485m folded=1.0.
- Office graph p95=0.211m median=0.180m under1m=1.0 folded=0 RMSE=0.0472m; production p95=18.545m folded=1.0.
- BigE reached epoch3 mean_loss=0.211 per monitor; needs fixed-heldout gate.

## BigE epoch 3 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e3_small on 8 cases/bucket.
- Training mean_loss improved to ~0.211, but geometry gate worsened vs epoch2 and baseline: gate checks 48 failed 38.
- Baseline grid p95 completed 0.636m; BigE e3 grid p95 0.731m. Baseline office completed p95 14.538m; BigE e3 13.963m but median office worsened 5.623m vs 2.426m. Random tied p95 but missing MAE worsened 0.250m vs 0.242m. Not accepted.

## BigD epoch 19 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_bigD_e19_small on 8 cases/bucket.
- Result gate checks 48 failed 36. Improvements: random missing MAE 0.233 vs baseline 0.242; office p95 completed 12.914m vs 14.538m and weak-polish 12.377m vs 15.233m; office missing MAE 0.997 vs 1.048.
- Regressions: grid p95 completed 0.683m vs baseline 0.636m, weak-polish 0.902m vs 0.731m; office median much worse. Not accepted.

## Graph eval harness max-nodes option
- Added --max-nodes to anchor_solver_graph_scaffold_eval.py to filter generated cases by anchor count. This avoids one 47-anchor case dominating quick graph-scaffold checks. The uncapped rgo8_h2 background eval was stopped and superseded after reaching grid case 3/8.

## Graph scaffold capped eval launch
- Started PID 36732: anchor_solver_graph_scaffold_eval_rgo8_cap32_h2, 8 cases per bucket, max_nodes=32, seed_count=6, iterations=45, max_hops=2, relative_sigma=0.36, hop_sigma=0.65.

## Graph scaffold capped rgo8 result
- Completed anchor_solver_graph_scaffold_eval_rgo8_cap32_h2: 8 cases each random/grid/office, max_nodes=32, seed_count=6, iterations=45, max_hops=2.
- Graph-shortest scaffold: random p95=0.199m median=0.157m under1m=1.0 folded=0 RMSE=0.0546m; grid p95=0.262m median=0.197m under1m=1.0 folded=0 RMSE=0.0363m; office p95=0.796m median=0.246m under1m=1.0 folded=0 RMSE=0.0414m.
- Production-priors same cases: random p95=11.852m folded=0.75; grid p95=14.375m folded=0.875; office p95=22.742m folded=0.75. This is the strongest classical solver evidence so far.

## BigD epoch 20 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_bigD_e20_small. Gate checks 48 failed 35.
- Improvements: office p95 completed 11.906m vs baseline 14.538m; weak-polish 12.077m vs 15.233m; office missing MAE 0.981 vs 1.048; grid missing MAE 0.846 vs 0.863.
- Regressions: grid p95 completed 0.986m vs 0.636m and under1m dropped to 0.875; random missing MAE 0.244 vs baseline 0.242. Not accepted.

## Graph scaffold selected-case figure
- Created outputs/anchor_solver_graph_scaffold_selected_cases.png and CSV. It renders selected cases 2 (random p95 graph), 12 (grid hard old-solver), and 23 (office hardest graph) with truth, production-priors, and graph-shortest scaffold side-by-side.
- The renderer re-generates cases from seed 2026062707 with max_nodes=32 and solves only selected cases using seed_count=6, iterations=45.

## Corrected graph selected-case figure
- Fixed selected-case renderer to consume graph_batch noise for every generated case before bucket transitions, matching the full capped eval RNG stream.
- Re-ran outputs/anchor_solver_graph_scaffold_selected_cases.png/CSV. Selected case counts now match capped eval detail: random case2 24 anchors, grid case12 27 anchors, office case23 24 anchors.

## BigE epoch 6 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e6_small. Gate checks 48 failed 33.
- Improvements: grid missing MAE 0.801 vs baseline 0.863; office p95 completed 13.563m vs 14.538m and weak-polish 14.669m vs 15.233m; office missing MAE 0.976 vs 1.048.
- Regressions: grid p95 completed 1.048m vs 0.636m and under1m dropped to 0.875; random missing MAE 0.249 vs 0.242. Not accepted.

## ML blend-sweep setup
- Because BigD/BigE improve office/missing but regress grid/random, next step is blend candidate predicted distance matrices with baseline predictions on the fixed heldout set.
- Snapshotting current BigD/BigE best checkpoints to avoid concurrent training overwrites while evaluating blends.

## Blend sweep launch
- Started PID 65716: anchor_solver_checkpoint_blend_sweep_small.
- Baseline checkpoint plus BigD snapshot and BigE snapshot; alphas 0.10,0.20,0.35,0.50,0.70,1.00; 8 fixed cases per bucket.

## Blend sweep result
- Completed anchor_solver_checkpoint_blend_sweep_small with baseline vs BigD/BigE snapshots, alphas 0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 8 fixed cases/bucket.
- No blend passed the all-metrics gate: 576 gate checks, 403 failures overall. Pattern: small alphas preserve random/grid but still worsen office medians; larger alphas improve missing MAE/office p95 somewhat but break grid p95/under1m.
- Blend approach not accepted; graph-shortest scaffold remains strongest improvement so far.

## BigE epoch 7 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e7_small. Gate checks 48 failed 34.
- Improvements: random missing MAE 0.239 vs baseline 0.242; grid missing MAE 0.816 vs 0.863; office p95 completed 13.542m vs 14.538m, weak-polish 14.599m vs 15.233m.
- Regressions: grid p95 completed 0.877m vs 0.636m and under1m dropped to 0.875; office median remains worse. Not accepted.

## BigE epoch 9 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e9_small. Gate checks 48 failed 24, best ML candidate so far but still not pass.
- Wins: random missing MAE 0.238 vs baseline 0.242; grid missing MAE 0.759 vs 0.863; office completed p95 13.665 vs 14.538, median 1.490 vs 2.426, under1m 0.500 vs 0.375, missing 0.919 vs 1.048; office weak-polish p95 14.124 vs 15.233, median 0.352 vs 1.458, under1m 0.625 vs 0.500.
- Fails: grid shape p95 completed 1.027 vs 0.636 and weak-polish 1.128 vs 0.731, under1m drops to 0.875. Next: blend e9 with baseline and test oracle bucket selector as sanity check.

## Oracle bucket selector sanity check
- Built selector_grid_baseline_else_e9 from e9 detail CSV: baseline rows for grid, e9 rows for random/office.
- It still fails strict gate: 48 checks, 28 failures. Reason: e9 improves office p95/median/under1m/missing but worsens office max_offset_m slightly (16.48m vs 16.14m completed), and strict gate also counts equal/tiny numerical metric changes. Not accepted.

## BigE e9 blend sweep launch
- Started PID 65456: anchor_solver_checkpoint_blend_sweep_e9. Candidate bigE_e9 snapshot vs baseline, alphas 0.05,0.10,0.15,0.20,0.30,0.40,0.55,0.75,1.00, 8 fixed cases/bucket.

## BigE e9 blend sweep result
- Completed anchor_solver_checkpoint_blend_sweep_e9. No blend passed: 432 gate checks, 275 failures.
- Best intuition: low alphas (0.05-0.20) slightly improve office p95/missing but immediately worsen weak-polish grid under1m/p95 relative to baseline; high alphas inherit e9 grid p95 >1m. Blend not accepted.

## BigE epoch 12 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e12_small. Gate checks 48 failed 35.
- Despite lower train mean_loss=0.20116, geometry worsened vs e9: grid p95 completed 1.114m vs baseline 0.636m; office completed p95 14.582m is slightly worse than baseline 14.538m; random missing MAE worsened to 0.253 vs baseline 0.242. Not accepted. Best ML geometry checkpoint remains BigE e9 snapshot, but still fails grid.

## BigE epoch 13 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e13_small. Gate checks 48 failed 32.
- E13 is better than E12 on grid p95 (0.702m completed, 0.751m weak) but still worse than baseline (0.636m, 0.731m) and weak-polish under1m drops to 0.875. Random missing MAE worsens to 0.246 vs baseline 0.242. Not accepted. Best ML geometry checkpoint remains E9, though it fails grid.

## BigA/BigB final fixed-heldout gate
- A/B finished and wrote outputs/anchor_solver_ml_distance_completion_bigA_h224_l8.pt and ...bigB_h192_l10.pt. Font fallback only in stderr.
- Gate anchor_solver_fixed_holdout_bigAB_final_small: 96 checks, 58 failures. Not accepted.
- BigA wins shape: grid p95 completed 0.504 vs baseline 0.636; office p95 completed 11.836 vs 14.538; random missing MAE 0.232 vs 0.242. But grid/office missing MAE worsens (0.939/1.123 vs 0.863/1.048).
- BigB similar/better shape: grid p95 completed 0.511, weak 0.497; office p95 completed 11.502, weak 11.926; random missing 0.230. But grid/office missing MAE worsens (0.974/1.238). Next: continue BigB with lower LR/checkpointing or blend with baseline.

## Continuation launch from BigA/BigB finals
- Launched two lower-LR continuation fine-tunes after A/B finals freed memory.
- BigB continue: prefix anchor_solver_ml_distance_completion_bigB_continue_lr6e4, PID 15944, hidden=192 layers=10 batch=32 lr=0.0006 edm=0.045 cap=5 init=bigB final checkpoint, checkpoint_every=1.
- BigA continue: prefix anchor_solver_ml_distance_completion_bigA_continue_lr6e4, PID 47132, hidden=224 layers=8 batch=32 lr=0.0006 edm=0.035 cap=5 init=bigA final checkpoint, checkpoint_every=1.
- Startup clean, GPU memory ~19.6GB/24GB. BigD also reached a new best around 0.207 and needs gating.

## BigD epoch 33 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_bigD_e33_small. Gate checks 48 failed 28.
- Strong office wins: completed p95 11.866 vs baseline 14.538, weak p95 11.549 vs 15.233, office missing MAE 0.902 vs 1.048, under1m 0.5/0.625 vs 0.375/0.5.
- Fails: grid p95 completed 0.953 vs 0.636 and weak 1.066 vs 0.731; random missing MAE 0.256 vs 0.242. Not accepted.

## Monitor after continuation warmup
- Continuations are running but first epochs are not yet better than parent final checkpoints. BigC is nearing final. BigD and BigE have lower best training losses (~0.198 and ~0.195); next step is fixed-heldout gating for new best checkpoints.

## BigD e35 + BigE current fixed-heldout gate
- Ran anchor_solver_fixed_holdout_bigD_e35_bigE_current_small. 96 checks, 58 failures total.
- BigD e35 is best balanced ML so far: random missing MAE 0.234 vs baseline 0.242; grid missing 0.794 vs 0.863; office p95 completed 13.001 vs 14.538, weak 12.252 vs 15.233, office medians and under1m improve. Fails grid shape: p95 completed 0.762 vs 0.636, weak 0.910 vs 0.731, under1m 0.875 vs 1.0.
- BigE current: missing MAE improves more but grid p95 ~1.02/1.13 and office median worse than BigD. Not accepted.

## Cap4 fixed-heldout gate
- Reran baseline vs BigD e35 and BigE current with closest_predicted_pairs_per_anchor=4.0. 96 checks, 57 failures total, not accepted.
- Baseline cap4 itself changes: grid p95 slightly better than cap5, office much worse median. BigE current cap4 improves office substantially (completed median 0.447m, p95 12.817m, under1m 0.625) but still breaks grid p95/under1m. BigD cap4 also fails grid. Keep graph scaffold as strongest.

## BigB continuation epoch 3 gate
- Ran anchor_solver_fixed_holdout_bigB_continue_e3_small. Gate checks 48 failed 32.
- Bad regression: grid p95 exploded to ~4.85m for both completed/weak despite random missing MAE improving to 0.237. Office missing worsened to 1.332. Not accepted; lower loss is not reliable for geometry here.

## Hybrid rank/value eval launch
- Created and launched anchor_solver_hybrid_rank_value_eval.py PID 17760.
- Tests baseline value predictions with rank/closest-pair ordering from bigA, bigB, bigC, and bigD e35, cap=5, 8 fixed cases/bucket. Goal: keep baseline missing MAE while borrowing geometry-improving pair rankings.

## Hybrid baseline-values/candidate-rank eval
- Completed anchor_solver_hybrid_rank_value_eval_small. No pass: 192 gate checks, 169 failures.
- Insight: rank hybrid can decouple missing MAE from shape. BigD rank + baseline values gives grid p95 0.493m completed / 0.655m weak while keeping baseline missing MAE, but office median worsens. A/B/C ranks improve office p95 (completed ~12.1-12.6, weak ~9.8) but break grid weak-polish under1m/p95. Useful idea but not accepted.

## BigA continuation best gate
- Ran anchor_solver_fixed_holdout_bigA_continue_best_small. Gate checks 48 failed 29.
- It preserves shape gains (grid p95 ~0.51m, office p95 ~12.8m, random missing 0.238) but still worsens grid/office missing MAE (0.969/1.216 vs baseline 0.863/1.048). Not accepted. Continuation does not fix the core shape-vs-missing tradeoff.

## BigE epoch 21 fixed-heldout gate
- Ran anchor_solver_fixed_holdout_finetuneE_e21_small. Gate checks 48 failed 30.
- Strong missing-MAE improvements: grid 0.693 vs 0.863, office 0.793 vs 1.048, random 0.231 vs 0.242. But shape still fails: grid p95 0.823/0.925 vs baseline 0.636/0.731 and under1m 0.875; office medians worse. Not accepted.
- Patched EDM eigvalsh in trainer with jitter/fallback after BigA continuation crashed on ill-conditioned matrix.

## BigD current cap6 gate
- Ran anchor_solver_fixed_holdout_bigD_current_cap6_small. Gate checks 48 failed 25 due strict tie handling + weak-polish grid regression.
- Important: ML completed at cap6 is very close/strong: BigD current beats baseline cap6 on grid p95 (0.623 vs 0.664), office p95 (13.523 vs 14.401), random/office/grid missing MAE (0.223/0.724/0.722 vs 0.242/1.048/0.863). Remaining completed failures appear mostly exact/effectively exact ties on medians/under1m/random p95. Weak-polish still fails grid under1m/p95.
- Candidate for final ML mode: BigD current with cap6 and ML completed only; need larger/tie-aware validation if claiming pass.

## BigD cap6 medium fixed-heldout validation
- Ran anchor_solver_fixed_holdout_bigD_current_cap6_med with 24 cases/bucket.
- Result: not robust. BigD cap6 improves missing MAE in all buckets (grid 0.668 vs 0.741, office 0.519 vs 0.589, random 0.281 vs 0.297) and weak grid p95 (1.307 vs 1.423), but completed grid p95 worsens (3.396 vs 1.998) and office p95 worsens (7.802/8.461 vs 6.862/6.912). Not accepted. Small 8-case apparent pass was not robust.
2026-06-27T08:35:06.7015404+02:00 latest gate bigD_e52/finetuneE_e30: lower missing MAE, but no all-metric pass; graph-shortest scaffold remains best practical solver. GPU 92%, BigD epoch54/72, BigB-cont epoch30/48, BigE epoch31/80.
2026-06-27T09:20:10.8467165+02:00 graph scaffold 32/bucket eval done: medians excellent but p95 has rare global failures (grid p95 6.65m, office 11.72m, random 3.15m), so graph scaffold needs robust parameter/candidate selection; training still active BigB e46/48 BigD e65/72 BigE e39/80.
2026-06-27T09:28:46.3147822+02:00 topology-selection graph probe: grid outliers fixed (5/5 under 0.322m); random hard cases still fail around 5-10m; office mostly fixed except ambiguous cases 64/72/93, with h8 helping case72 but not case93. Need more compute/diagnostics.
2026-06-27T09:30:05.7205398+02:00 gated BigB-cont final/best, BigD e67 best, BigE e41 best: no all-metric pass. BigD best: missing MAE grid 0.641/off 0.683/random 0.237 but geometry still fails (grid weak under1=0.875; office p95 11.8/12.2m). BigB final overtrained grid p95 3.16m.
2026-06-27T09:34:25.5161016+02:00 launched BigF from BigD best: hidden176/layers5, lr=3.5e-4, edm_weight=0.08, cap=5, epochs36, PID 63432. Init loaded and step1 running on CUDA. Cap sweep BigD best: cap helps grid weak but office p95 still ~7m best.
2026-06-27T09:38:29.3672322+02:00 BigF epoch1 done mean_loss=0.17753 best saved; BigD e70/72 best=0.186; BigE e43/80 best=0.176; patched graph eval at random 6/16.
2026-06-27T09:49:56.1303434+02:00 gated BigD final/best and BigF e2: no all-metric pass. BigD best improves missing MAE grid 0.595/off 0.694/random 0.222 and office p95 11.44/10.92, but grid completed p95 0.649 worse than baseline 0.636. BigF e2 improves grid p95 0.522/0.658 and missing 0.593/0.607/0.214, but office p95 worsens 13.04/14.15.
2026-06-27T09:52:25.0442970+02:00 launched BigG radio-lower hinge from BigD best; first step radio_lower_loss=0 because model forward already clamps missing predictions >= radio radius. Important: office failures are not due predicted missing distances below radio threshold; they are sparse/global ambiguity or selected-distance insufficiency.
2026-06-27T09:54:52.6651717+02:00 patched graph eval rgo16 h2 done: Random p95=0.220 under1=1.0; Grid median=0.220 but p95=3.484 under1=0.938; Office median=0.213 but p95=7.604 under1=0.813. Topology selector not sufficient for p95.
2026-06-27T09:58:00.6715076+02:00 gated BigF/BigG early: no pass. BigF best: grid p95 0.488/0.558 and missing MAE 0.584/0.581/0.211, but office p95 11.697/12.255 and under1 0.375/0.500. BigD still better office p95 but worse grid completed. Single checkpoint not enough.
2026-06-27T10:07:59.3885231+02:00 candidate selector eval small: selector improves office p95 to 8.64 vs individual ML ~10-11, grid/random fine, but oracle-best candidate office p95 still 6.33 => candidate pool lacks good solution for hard office cases; selector alone not enough.
2026-06-27T10:09:02.6460446+02:00 BigF e8 gate: no strict all-method pass, but ML-completed branch is promising: grid p95 0.619 < baseline 0.636, office p95 10.970 <14.538, missing MAE 0.615/0.643/0.216 improved; random p95 tied ~0.307. Weak-polish branch regresses grid under1=0.875 and p95 0.758.
2026-06-27T10:10:55.0045918+02:00 BigF e9/BigE best gate: no pass. BigF e9 lower loss and missing MAE grid/off/random 0.553/0.606/0.211 but office p95 worsened vs e8 (12.426/13.435 vs e8 10.970/12.831). Loss not aligned with geometry; e8 checkpoint overwritten by best_loss.
2026-06-27T10:14:53.9034149+02:00 graph h4 eval rgo16 done: Random p95=0.220, Grid p95=3.446 (same issue), Office p95=10.194 (worse than h2 7.604). Wider hops do not fix office p95.
2026-06-27T10:15:45.7568655+02:00 BigG e5 gate: no pass, office p95 poor (14.06/14.82). BigF remains best ML trajectory, especially ML-completed; BigG radio term inert due output clamp.
2026-06-27T10:32:49.7846399+02:00 BigE e53 gate: no pass. BigE e53 completed grid p95 0.519, office p95 11.239 under1=0.625, missing 0.605/0.702/0.211; weak office under1=0.750 but p95 12.273. BigF current best remains lower missing but office p95 ~11.7.
2026-06-27T10:34:46.9830639+02:00 medium 24-case validation BigF/BigE: no pass, worse p95 despite lower missing MAE. Baseline grid p95 1.579/1.303, office 7.542/7.073, random 0.199; BigF grid 6.694/6.494, office 9.311/9.175, random 0.199, missing MAE much lower 0.460/0.425/0.243. ML learns distances but can worsen solved sparse geometry.
2026-06-27T10:40:46.3459625+02:00 BigF e16/BigE e54 gate: no strict pass. BigF e16 small-set improved grid p95 0.626/0.709 vs baseline 0.636/0.731, office 11.152/11.271 vs 14.538/15.233 and under1 0.625/0.750, missing lower. But medium validation earlier showed p95 not robust.
2026-06-27T10:53:42.1360737+02:00 BigF e19 small gate looked best, but med24 failed: missing MAE lower (grid 0.452/off 0.412/random 0.251) but p95 worsens vs baseline: grid 6.431/6.493 vs 1.579/1.303, office 9.651/9.583 vs 7.542/7.073. Small-set geometry not robust.
2026-06-27T10:53:52.4441850+02:00 keep-prior graph h2 eval done: essentially same as previous h2 (random p95 0.220, grid 3.484, office 7.607). Choosing prior-preserved vs pure polish did not fix p95.
2026-06-27T11:33:19.3158872+02:00 stopped remaining training near 9h window. Final active bests: BigF best e19/e later overwritten? latest best_loss from BigF was e19 then continued; final small gates no strict pass, med24 no pass. BigE final no pass. Processes stopped.
