"""
Batch test runner.

YAML schema (examples):

tests:
  - users_file: userslist/test1.txt
    sites:
      - site: head
        checkpoint: checkpoints/head.ckpt
      - site: neck
        checkpoint: checkpoints/neck.ckpt

You can also omit the top-level key and provide a list directly.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

RESULTS_ROOT = Path("results/single_site_debug")
COMBINED_XLSX = RESULTS_ROOT / "combined.xlsx"


def load_tests(config_path: Path) -> List[dict]:
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty config: {config_path}")
    if isinstance(data, dict) and "tests" in data:
        tests = data["tests"]
    elif isinstance(data, list):
        tests = data
    else:
        raise ValueError("Config must be a list or contain a 'tests' list.")
    if not isinstance(tests, list):
        raise ValueError("'tests' must be a list.")
    return tests


def read_users(users_file: Path) -> List[str]:
    users = []
    with users_file.open("r") as f:
        for line in f:
            username = line.strip()
            if not username or username.startswith("#"):
                continue
            users.append(username)
    return users


def run_single_test(site: str, checkpoint: Path, user: str, debug: bool) -> int:
    cmd = [
        sys.executable,
        "src/test.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        site,
        "--leave_out_users",
        user,
    ]
    if debug:
        cmd.append("--debug")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def safe_sheet_name(name: str) -> str:
    invalid_chars = '[]:*?/\\'
    cleaned = "".join("_" if c in invalid_chars else c for c in name)
    if len(cleaned) > 31:
        cleaned = cleaned[:28] + "..."
    return cleaned or "unknown"


def infer_user_from_path(csv_path: Path, site: str) -> str:
    stem = csv_path.stem
    suffix = f"_{site}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)] or stem
    return stem


def aggregate_results(results_root: Path = RESULTS_ROOT, output_path: Path = COMBINED_XLSX) -> None:
    if not results_root.exists():
        logging.warning("Results directory not found: %s", results_root)
        return
    user_sites: Dict[str, List[pd.DataFrame]] = {}
    for site_dir in results_root.iterdir():
        if not site_dir.is_dir():
            continue
        site = site_dir.name
        for csv_file in site_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:  # pragma: no cover - defensive
                logging.error("Failed to read %s: %s", csv_file, exc)
                continue
            user = infer_user_from_path(csv_file, site)
            # rename metrics with site prefix; keep fname as join key
            renamed = df.rename(
                columns={col: f"{site}_{col}" for col in df.columns if col != "fname"}
            )
            user_sites.setdefault(user, []).append(renamed)
    if not user_sites:
        logging.warning("No CSV files found under %s", results_root)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for user, frames in sorted(user_sites.items()):
            combined = None
            for frame in frames:
                combined = frame if combined is None else combined.merge(frame, on="fname", how="outer")
            combined.to_excel(writer, sheet_name=safe_sheet_name(user), index=False)
    logging.info("Wrote combined Excel to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Batch test runner using YAML config.")
    parser.add_argument("--config", required=True, help="Path to batch YAML config.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip running tests; only aggregate existing CSVs into Excel.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if args.aggregate_only:
        aggregate_results()
        return

    tests = load_tests(config_path)

    for entry in tests:
        if not isinstance(entry, dict):
            raise ValueError("Each test entry must be a mapping.")
        users_file = entry.get("users_file")
        sites = entry.get("sites")
        if not users_file or not sites:
            raise ValueError("Each entry needs 'users_file' and 'sites'.")
        users_path = Path(users_file)
        if not users_path.is_file():
            raise FileNotFoundError(f"Users file not found: {users_path}")
        users = read_users(users_path)
        if not users:
            logging.warning("No users found in %s", users_path)
            continue
        for site_entry in sites:
            site = site_entry.get("site")
            checkpoint = site_entry.get("checkpoint")
            if not site or not checkpoint:
                raise ValueError("Each site entry needs 'site' and 'checkpoint'.")
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            for user in users:
                logging.info("Testing user=%s site=%s checkpoint=%s", user, site, checkpoint_path)
                code = run_single_test(site, checkpoint_path, user, debug=True)
                if code != 0:
                    logging.error("Test failed for user=%s site=%s (exit=%s)", user, site, code)

    aggregate_results()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()

