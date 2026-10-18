"""Minimal sequential sweep runner: one `tinylm.train` subprocess per (config, seed).

Honesty bar: one registry row per run is written, `ok` or `failed` - failed runs
are never silently dropped. There is no retry/append mode; an existing registry
refuses to run (re-runs take a fresh --out).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tinylm.utils import config_hash, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a sweep: one tinylm.train subprocess per (config, seed)."
    )
    p.add_argument("--sweep", required=True, help="sweep descriptor YAML")
    p.add_argument(
        "--out", required=True, help="output dir; its basename namespaces run-dirs"
    )
    p.add_argument(
        "--filter",
        default=None,
        help="substring on the config stem, or key=value on the loaded config",
    )

    return p.parse_args(argv)


def load_sweep(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("entries") if isinstance(raw, dict) else None

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"sweep needs a non-empty 'entries' list: {path}")

    for entry in entries:
        if (
            not isinstance(entry, dict)
            or "config" not in entry
            or not isinstance(entry.get("seeds"), list)
        ):
            raise ValueError(f"each entry needs 'config' and a 'seeds' list: {entry!r}")

    return raw


def git_commit() -> tuple[str | None, str]:
    """Return (commit sha, ""), or (None, reason) when git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return proc.stdout.strip(), ""

    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"git unavailable: {exc}"


def matches_filter(spec: str, cfg_path: str, cfg) -> bool:
    if "=" not in spec:
        return spec in Path(cfg_path).stem

    key, _, value = spec.partition("=")

    return (
        key in {f.name for f in dataclasses.fields(cfg)}
        and str(getattr(cfg, key)) == value
    )


def last_metrics_row(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.jsonl"

    if not path.exists():
        return None

    last = None

    with path.open() as f:
        for line in f:
            if line.strip():
                last = json.loads(line)

    return last


def run_one(
    sweep_name: str,
    cfg_path: str,
    seed: int,
    steps_override: int | None,
    device_override: str | None,
    timeout_s: int,
    commit: str | None,
    commit_reason: str,
) -> dict:
    run_name = f"{sweep_name}-{Path(cfg_path).stem}-s{seed}"
    run_dir_rel = f"logs/{run_name}"

    # config_hash recomputed from the descriptor + overrides; None if it never loads
    recomputed = None
    loaded = None

    try:
        loaded = load_config(REPO_ROOT / cfg_path)

        if device_override is not None:
            loaded.device = device_override

        if steps_override is not None:
            loaded.steps = steps_override

        loaded.seed = seed
        recomputed = config_hash(loaded)

    except (OSError, ValueError, TypeError, yaml.YAMLError):
        loaded = None

    cmd = [
        sys.executable,
        "-m",
        "tinylm.train",
        "--config",
        cfg_path,
        "--seed",
        str(seed),
        "--run-name",
        run_name,
    ]

    if steps_override is not None:
        cmd += ["--steps", str(steps_override)]

    if device_override is not None:
        cmd += ["--device", device_override]

    t0 = time.perf_counter()
    status = "failed"
    stderr_tail = ""
    final_train_acc = final_val_acc = torch_version = logged_hash = None

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )

        if proc.returncode == 0:
            status = "ok"
            row = last_metrics_row(REPO_ROOT / run_dir_rel)

            if row is not None:
                final_train_acc = row["train_acc"]
                final_val_acc = row["val_acc"]
                torch_version = row["torch_version"]
                logged_hash = row["config_hash"]

                if recomputed is not None and logged_hash != recomputed:
                    print(
                        f"WARNING: {run_name}: logged config_hash {logged_hash} != "
                        f"recomputed {recomputed}",
                        file=sys.stderr,
                    )
        else:
            stderr_tail = (proc.stderr or proc.stdout or "")[-500:]

    except subprocess.TimeoutExpired as exc:
        stderr_tail = f"timeout after {timeout_s}s; {str(exc)[-400:]}"

    wall_s = time.perf_counter() - t0

    if commit is None:
        stderr_tail = f"{commit_reason}; {stderr_tail}"[-500:]

    if status != "ok":
        hash_match = None
    elif logged_hash is None or recomputed is None:
        hash_match = False
    else:
        hash_match = logged_hash == recomputed

    return {
        "sweep": sweep_name,
        "config": cfg_path,
        "seed": seed,
        "config_hash": recomputed if recomputed is not None else logged_hash,
        "hash_match": hash_match,
        "git_commit": commit,
        "run_dir": run_dir_rel,
        "status": status,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "backend": device_override or (loaded.device if loaded is not None else None),
        "wall_s": round(wall_s, 3),
        "torch_version": torch_version,
        "stderr_tail": stderr_tail,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    registry = out / "registry.jsonl"

    if registry.exists():
        print(
            f"refusing to run: {registry} exists; re-run with a fresh --out "
            f"(or rm -rf {out})",
            file=sys.stderr,
        )
        sys.exit(1)

    sweep = load_sweep(Path(args.sweep))
    sweep_name = out.name
    commit, commit_reason = git_commit()
    timeout_s = sweep.get("timeout_s", 300)
    steps_override = sweep.get("steps_override")
    device_override = sweep.get("device_override")

    entries = sweep["entries"]

    if args.filter is not None:
        kept = []

        for entry in entries:
            try:
                cfg = load_config(REPO_ROOT / entry["config"])
            except (OSError, ValueError, TypeError, yaml.YAMLError):
                continue

            if matches_filter(args.filter, entry["config"], cfg):
                kept.append(entry)

        entries = kept

    out.mkdir(parents=True, exist_ok=True)

    with registry.open("w") as reg:
        for entry in entries:
            for seed in entry["seeds"]:
                row = run_one(
                    sweep_name,
                    entry["config"],
                    seed,
                    steps_override,
                    device_override,
                    timeout_s,
                    commit,
                    commit_reason,
                )
                reg.write(json.dumps(row) + "\n")
                reg.flush()
                print(
                    f"{row['status']:>6}  {row['run_dir']}  "
                    f"{row['config_hash']}  {row['wall_s']:.1f}s"
                )


if __name__ == "__main__":
    main()
