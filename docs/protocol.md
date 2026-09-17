# Evaluation protocol

The completed evaluation contains **900 episodes in 18 complete groups**. The final configuration, recovery, and episode audits passed. This repository is a compact result release with policy reference source and exported outcomes. It does not include the full simulator, demonstration assets, or a complete evaluation rerun harness.

## Evaluated system

`gpt-6-astra` generated a demonstration retrieval controller at `xhigh` reasoning effort. The generated controller was frozen without manual edits. During evaluation, this program used the current images, robot state, training demonstrations, and its own episode memory to select the next action sequence. Astra was not called at each control step. The reported result measures this Astra-authored system; it is not a zero-shot result.

Both the online and generated-program approaches began development at `high`. The controlled online comparison at `xhigh`, `max`, and `ultra` had 0 successes in 18 development episodes. The selected program configuration had 4 successes in a separate 18-episode development validation. Development also compared action-sequence lengths and deployment configurations. These small samples supported candidate selection and were excluded from the reported test scores; they do not establish an optimal configuration.

Each task and randomization condition used the first 100 official training demonstrations (`train0:100`) from [DexJoCo-Datasets-LeRobot, revision `5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3`](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot/tree/5a57c54e55dc5858dd9fb949c5f67c0c9716e6b3). Demonstration assets contain images, robot state, and actions. Policy source and training assets remained fixed during evaluation and recovery.

## Tasks and environment

The environment was pinned to [DexJoCo commit `8d23b0fab23b17a58c4b55f3942e17013aaf8267`](https://github.com/brave-eai/dexjoco/tree/8d23b0fab23b17a58c4b55f3942e17013aaf8267). Task configurations, physical control, randomization, and success checks follow this version.

| Task | Maximum steps | Official success condition |
| --- | ---: | --- |
| [Hammer Nail](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/sim/envs/panda_hammer_nail_env.py) | 1,000 | Nail depth reaches at least 0.04 m. |
| [Water Plant](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/sim/envs/panda_water_plant_env.py) | 1,000 | The sprayer reference point stays in the official plant region with the trigger pulled for 30 consecutive steps. Pulling the trigger outside this region terminates the episode as a failure. |
| [Bimanual Microwave Cook](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/sim/envs/panda_bimanual_microwave_cook_env.py) | 1,100 | Food is inside, the door is closed, and the start button is in contact. |

Microwave evaluation covers the full **open door, load food, close door, start** task. Its official terminal check requires button contact, without requiring a heating sequence. Scores use official terminal outcomes; video inspection does not replace those checks.

Each task was evaluated under two conditions:

- `rand_obj`: randomized objects and table height.
- `rand_full`: the same randomization plus the main camera, lighting, and table texture.

Neither condition additionally enabled dynamics randomization.

## Observations and actions

The runtime inputs were the official RGB images, robot state, and original task instruction. Images use the official aspect-preserving resize and padding to 224 × 224 RGB `uint8`. Under `rand_full`, the main image comes from the official randomized camera.

| Interface | Single arm: Hammer, Water | Two arms: Microwave |
| --- | --- | --- |
| Images | Main camera, wrist camera | Main camera, left wrist, right wrist |
| Robot state | 23 values: end-effector world position (3), quaternion (4), hand joints (16) | 46 values: right end-effector pose (7), left end-effector pose (7), right hand joints (16), left hand joints (16) |
| Action | 22 values: absolute world target position (3), rotation vector (3), hand-joint targets (16) | 44 values: the right-arm action (22), followed by the left-arm action (22) |

The official joint ordering for each hand was preserved. The controller reduced the received images to retrieval features internally. No depth, object ground-truth coordinates, camera extrinsics, contacts, rewards, success flags, seeds, or environment step counters were supplied to the policy. Each episode began with empty controller memory; no extra history images were supplied. See the pinned [observation adapter](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco/tasks/obs_adapters.py) and [evaluation environment interface](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py).

## Timing and deployment

The controller produced 30-step action sequences, with replanning horizon 30 and replanning ratio 0.8. The [official asynchronous evaluation loop](https://github.com/brave-eai/dexjoco/blob/8d23b0fab23b17a58c4b55f3942e17013aaf8267/dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py) continued stepping the environment while requests were pending. Expired actions were discarded, overlapping actions were blended under the original rules, and an empty action buffer held the current pose. Computation, network, and queue delays consumed the original step budget. The simulator was not paused to wait for the controller, and task budgets were not extended.

The program ran on an Apple M5 CPU, while simulation ran on an A100 server. There were six policy services, one worker per task/condition, at most three environment streams per service, and at most 15 active environments overall.

| Runtime | Versions |
| --- | --- |
| Simulator | Python 3.10.12, NumPy 1.26.4, MuJoCo 3.3.2, Gymnasium 0.29.1 |
| Controller | Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0 |

A renderer compatibility patch added a constructor fallback for unsupported width/height arguments while retaining the model's offscreen size. Its source and diff were recorded in the frozen manifest. It did not change physics or success checks.

Across the six scored conditions, median end-to-end request time ranged from 118 to 249 ms, including image encoding, transport, and response checks. There were 98 pose-hold steps and zero wholly expired action sequences. These steps remained within the original budgets.

## Sampling and statistics

Each task/condition used process seeds 0, 1, and 2, with 50 consecutive episode resets per process. A process was seeded once at startup; episodes were not individually reseeded. Thus the evaluation contains 3 tasks × 2 conditions × 3 seeds × 50 episodes = 900 episodes.

For each task/condition, the reported percentage is the mean of the three per-seed success rates. The uncertainty is their sample standard deviation (`ddof=1`, denominator 2). It describes variation across three seeded processes. Missing episodes and infrastructure errors were never filled in as task failures. A complete valid group with zero successes was retained as a valid result.

## Earlier attempts and fixed recovery

The evaluation had prior test-seed exposure and two interrupted attempts:

1. An early online `low`-effort pilot with a 600-step action horizon accessed official test seeds. Its 35 failed terminal episodes were excluded.
2. The first attempt with the selected frozen program was interrupted by laptop sleep lasting 358 seconds, exceeding the request timeout. All 15 terminal episodes from that attempt were excluded.
3. The next attempt lost its reverse SSH connection after 805 terminal episodes. Nine complete valid groups contributed 450 retained episodes. All 355 episodes from the nine interrupted groups were excluded.

Before any recovery run began, an immutable recovery manifest fixed the nine groups to retain and the nine groups to rerun. Each interrupted group restarted from its original process seed and ran all 50 resets once. No partial group was continued, and no episodes were selected for their success outcomes. Policy code, demonstration assets, evaluator, action deadlines, and HTTP timing rules remained fixed.

| Task / condition | Retained seeds | Seeds rerun in full |
| --- | --- | --- |
| Hammer / `rand_obj` | 0, 1, 2 | — |
| Hammer / `rand_full` | 0, 1, 2 | — |
| Water / `rand_obj` | 0, 1 | 2 |
| Water / `rand_full` | 0 | 1, 2 |
| Microwave / `rand_obj` | — | 0, 1, 2 |
| Microwave / `rand_full` | — | 0, 1, 2 |

The final result combines 450 retained episodes and 450 complete recovery episodes. The original aggregation and supplemental audit checked all 18 groups, frozen configuration and asset hashes, service/request provenance, software compatibility, and episode completeness. All 900 accepted episodes passed final acceptance. This is an exploratory evaluation with disclosed test-seed exposure and a fixed recovery procedure, rather than an untouched blind test.

## Published comparison and release scope

DP-T, DP-C, ACT, π0.5, and GR00T N1.5 results are references from the [DexJoCo paper, Table 2](https://arxiv.org/html/2605.16257v1#S4.T2). They were not rerun on this deployment. Public task rules are aligned, but hardware, execution speed, and logging overhead were not matched. The paper does not provide episode initialization receipts for pairing with these runs. Consequently, these comparisons do not establish a same-hardware reproduction or a paired significance result.

The public files support inspection of the controller and recomputation of the released outcome statistics. Full replay of the original audit or simulator evaluation also requires the original demonstrations, simulator setup, execution harness, and detailed runtime receipts, which are outside this compact release. Exported records are summaries of the completed local audit; they do not constitute a fresh independent rerun.
