# Comparison with the published DexJoCo results

Values are success-rate percentages, reported as mean and standard deviation. Our results use three seeds with 50 episodes each; the standard deviation is the sample standard deviation across the three seed-level rates. Reference scores are from [Table 2 of the DexJoCo paper](https://arxiv.org/html/2605.16257v1#S4.T2). They were not rerun on our hardware.

## rand_obj

| Policy | Hammer Nail | Water Plant | Microwave Cook (two arms) |
| --- | ---: | ---: | ---: |
| Controller written by GPT-6 Astra (this evaluation) | $21.3\pm5.8$ | $22.0\pm0.0$ | $2.7\pm1.2$ |
| DP-T | $81.3\pm3.1$ | $84.0\pm3.5$ | $73.3\pm11.6$ |
| DP-C | $58.7\pm4.2$ | $63.3\pm3.1$ | $54.0\pm12.5$ |
| ACT | $50.0\pm7.2$ | $47.3\pm4.6$ | $66.0\pm2.0$ |
| π0.5 | $84.7\pm5.0$ | $88.7\pm3.1$ | $70.0\pm3.5$ |
| GR00T N1.5 | $67.3\pm4.2$ | $72.7\pm1.2$ | $50.7\pm4.6$ |

## rand_full

| Policy | Hammer Nail | Water Plant | Microwave Cook (two arms) |
| --- | ---: | ---: | ---: |
| Controller written by GPT-6 Astra (this evaluation) | $6.7\pm4.6$ | $22.0\pm5.3$ | $1.3\pm1.2$ |
| DP-T | $18.7\pm1.2$ | $56.0\pm8.7$ | $21.3\pm4.6$ |
| DP-C | $19.3\pm3.1$ | $54.0\pm5.3$ | $62.7\pm6.4$ |
| ACT | $22.7\pm6.1$ | $52.7\pm8.1$ | $50.0\pm6.9$ |
| π0.5 | $17.3\pm5.0$ | $75.3\pm6.4$ | $54.7\pm6.1$ |
| GR00T N1.5 | $38.7\pm8.3$ | $66.0\pm5.3$ | $42.0\pm7.2$ |

The evaluated controller uses 100 official training demonstrations for each task and condition. All six reported means are below the five reference policies. This comparison does not establish paired statistical significance or measure the best performance achievable by GPT-6 Astra.

See [episode records](../results/episodes.csv), [seed-level counts](../results/per_seed.csv), [summaries](../results/summary.csv), and [reference scores](../results/published_baselines.csv). The [protocol](protocol.md) describes inputs, timing, interruptions, and prior test-seed exposure.
