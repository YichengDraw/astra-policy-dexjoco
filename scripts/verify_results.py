#!/usr/bin/env python3
"""Verify the portable results export using only the Python standard library.

Private originals are identified by SHA-256; they are not bundled. This script
checks the export, not simulator execution or the full private raw audit trail.
"""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("hammer_nail", "water_plant", "bimanual_microwave_cook")
REGIMES = ("rand_obj", "rand_full")
MODELS = ("DP-T", "DP-C", "ACT", "π0.5", "GR00T N1.5")
EXPECTED = set(itertools.product(TASKS, REGIMES, range(3)))
PAIRS = set(itertools.product(TASKS, REGIMES))
METRICS = ("steps", "policy_action_steps", "stay_steps", "expired_chunks", "wall_seconds")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BASELINE_URL = "https://arxiv.org/html/2605.16257v1#S4.T2"
SCHEMAS = {
    "episodes": "task regime seed role episode success steps policy_action_steps stay_steps expired_chunks wall_seconds initial_sha256",
    "per_seed": "task regime seed role episodes successes success_rate_pct steps policy_action_steps stay_steps expired_chunks wall_seconds episodes_source_id summary_source_id",
    "summary": "task regime seeds episodes successes mean_success_rate_pct sample_std_success_rate_pct ddof",
    "published_baselines": "model task regime mean_success_rate_pct published_std_success_rate_pct source_url evaluation_kind",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    require(re.fullmatch(r"0|[1-9][0-9]*", str(value)) is not None, f"Invalid nonnegative integer: {value!r}")
    return int(value)


def number(value):
    result = float(value)
    require(math.isfinite(result) and result >= 0, f"Invalid finite nonnegative number: {value!r}")
    return result


def same(actual, expected, label):
    require(math.isclose(number(actual), expected, rel_tol=1e-12, abs_tol=1e-9), f"Mismatch: {label}")


def identity(row):
    return row["task"], row["regime"], integer(row["seed"])


def expected_role(key):
    task, regime, seed = key
    retained = task == "hammer_nail" or (task == "water_plant" and (seed == 0 or (regime == "rand_obj" and seed == 1)))
    return "retained" if retained else "retry"


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    require(provenance["schema_version"] == 1 and provenance["export_is_derived"] is True, "Expected a derived schema-1 export")
    protocol = provenance["evaluation"]
    require(set(protocol["tasks"]) == set(TASKS) and set(protocol["regimes"]) == set(REGIMES), "Wrong task/regime scope")
    require(protocol["seeds"] == [0, 1, 2] and protocol["episodes_per_group"] == 50, "Wrong seed/episode protocol")
    require(protocol["episode_indices"] == [0, 49] and protocol["groups"] == 18 and protocol["episodes"] == 900 and protocol["sample_std_ddof"] == 1, "Wrong export protocol")
    digests = provenance["source_sha256"]
    require(all(re.fullmatch(r"[a-z0-9_./]+", key) and not key.startswith("/") and ".." not in key and SHA256.fullmatch(value) for key, value in digests.items()), "Invalid conceptual source IDs or SHA-256 digests")
    frozen = provenance["frozen"]
    for prefix in ("evaluation", "recovery"):
        require(frozen[f"{prefix}_manifest_sha256"] == digests[frozen[f"{prefix}_manifest_source_id"]], "Broken frozen source digest reference")
    require(SHA256.fullmatch(frozen["controller_sha256"]), "Invalid controller digest")
    require(hashlib.sha256((ROOT / "policy/controller.py").read_bytes()).hexdigest() == frozen["controller_sha256"], "Frozen controller file digest mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", frozen["official_commit"]), "Invalid official commit")
    require(re.fullmatch(r"[0-9a-f]{40}", provenance["training_demonstrations"]["revision"]), "Invalid dataset revision")

    accepted = provenance["accepted_groups"]
    require(len(accepted) == 18 and {identity(r) for r in accepted} == EXPECTED, "Accepted groups must be exactly 18 distinct identities")
    accepted = {identity(r): r for r in accepted}
    source_ids = {"frozen/evaluation_manifest.json", "recovery/recovery_manifest.json", "audit/final_audit.json", "aggregate/report.json"}
    for key, group in accepted.items():
        require(group["role"] == expected_role(key), f"Wrong accepted role: {key}")
        task, regime, seed = key
        for kind, filename in (("episodes", "episodes.jsonl"), ("summary", "summary.json")):
            source_id = f"accepted/{task}/{regime}/seed{seed}/{filename}"
            require(group[f"{kind}_source_id"] == source_id and source_id in digests, "Broken accepted source digest reference")
            source_ids.add(source_id)
    require(set(digests) == source_ids, "Unexpected or missing source digest IDs")

    tables = {}
    counts = {"episodes": 900, "per_seed": 18, "summary": 6, "published_baselines": 30}
    require(set(provenance["export_files"]) == {f"results/{name}.csv" for name in counts}, "Unexpected export file IDs")
    for name, count in counts.items():
        relative = f"results/{name}.csv"
        path = ROOT / relative
        receipt = provenance["export_files"][relative]
        require(SHA256.fullmatch(receipt["sha256"]) and hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"], f"Export digest mismatch: {relative}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames == SCHEMAS[name].split(), f"Unexpected CSV columns: {relative}")
            rows = list(reader)
        require(len(rows) == count == receipt["rows"] and all(None not in r and None not in r.values() for r in rows), f"CSV row count/shape mismatch: {relative}")
        tables[name] = rows

    grouped = {key: [] for key in EXPECTED}
    for row in tables["episodes"]:
        key = identity(row)
        require(key in EXPECTED and row["role"] == expected_role(key), "Unexpected episode identity/role")
        require(row["success"] in ("true", "false"), "Success must be a strict boolean encoded as true/false")
        require(SHA256.fullmatch(row["initial_sha256"]), "Invalid initial-state digest")
        integer(row["episode"])
        for metric in METRICS:
            (number if metric == "wall_seconds" else integer)(row[metric])
        limit = 1100 if key[0] == "bimanual_microwave_cook" else 1000
        require(0 < integer(row["steps"]) <= limit, "Episode exceeds its step budget")
        require(integer(row["policy_action_steps"]) + integer(row["stay_steps"]) == integer(row["steps"]), "Episode step accounting mismatch")
        grouped[key].append(row)
    require(sum(row["role"] == "retained" for row in tables["episodes"]) == 450, "Wrong retained/retry split")
    for key, rows in grouped.items():
        require(len(rows) == 50 and {integer(r["episode"]) for r in rows} == set(range(50)), f"Incomplete or duplicate episodes: {key}")

    seed_rows = tables["per_seed"]
    require(len(seed_rows) == 18 and {identity(r) for r in seed_rows} == EXPECTED, "Wrong seed table identities")
    rates = {}
    for row in seed_rows:
        key = identity(row)
        episodes = grouped[key]
        successes = sum(r["success"] == "true" for r in episodes)
        require(integer(row["episodes"]) == 50 and integer(row["successes"]) == successes, "Wrong seed counts")
        require(row["role"] == accepted[key]["role"], "Wrong seed role")
        for kind in ("episodes", "summary"):
            require(row[f"{kind}_source_id"] == accepted[key][f"{kind}_source_id"], "Broken seed source reference")
        same(row["success_rate_pct"], 100 * successes / 50, "seed success rate")
        for metric in METRICS:
            if metric != "wall_seconds":
                integer(row[metric])
            same(row[metric], sum(number(r[metric]) for r in episodes), f"seed {metric}")
        rates[key] = 100 * successes / 50

    summary = tables["summary"]
    require(len(summary) == 6 and {(r["task"], r["regime"]) for r in summary} == PAIRS, "Wrong summary identities")
    for row in summary:
        pair = row["task"], row["regime"]
        seed_rates = [rates[(*pair, seed)] for seed in range(3)]
        successes = sum(r["success"] == "true" for seed in range(3) for r in grouped[(*pair, seed)])
        require(integer(row["seeds"]) == 3 and integer(row["episodes"]) == 150 and integer(row["successes"]) == successes and integer(row["ddof"]) == 1, "Wrong summary counts/ddof")
        same(row["mean_success_rate_pct"], statistics.mean(seed_rates), "summary mean")
        same(row["sample_std_success_rate_pct"], statistics.stdev(seed_rates), "summary sample SD")

    baselines = tables["published_baselines"]
    require(len(baselines) == 30 and {(r["model"], r["task"], r["regime"]) for r in baselines} == set(itertools.product(MODELS, TASKS, REGIMES)), "Incomplete baseline identities")
    for row in baselines:
        require(number(row["mean_success_rate_pct"]) <= 100 and number(row["published_std_success_rate_pct"]) <= 100, "Baseline percentage out of range")
        require(row["source_url"] == BASELINE_URL and row["evaluation_kind"] == "published_reference_not_rerun", "Invalid baseline source/kind")
    require(provenance["published_baselines"]["source_url"] == BASELINE_URL and provenance["published_baselines"]["source_id"] in digests, "Broken baseline provenance")
    total_success = sum(r["success"] == "true" for r in tables["episodes"])
    print(f"PASS: 900 episodes, 18 complete groups, 6 summaries, 30 published baseline rows; {total_success} successes.")
    print("PASS: seeds 0–2, episodes 0–49, retained/retry split 450/450, exported hashes and 40 source digest references.")
    print("PASS: step budgets/accounting, finite latency fields, per-seed totals, mean and sample SD (ddof=1).")
    print("Scope: verifies exported results; does not reproduce simulator runs, private raw audit evidence, or published baselines.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
