#!/usr/bin/env python3
"""Aggregate mean and std for key "10000" across KS, Wasserstein, and Shapiro-Wilk outputs."""
from pathlib import Path
import json
from statistics import mean, pstdev

BASE_DIR = Path(__file__).resolve().parent
KS_DIR = BASE_DIR / "output" / "KS_dist"
WASSERSTEIN_DIR = BASE_DIR / "output" / "Wasserstein_dist"
SW_DIR = BASE_DIR / "output" / "SW_test"
TARGET_KEY = "10000"


def load_values(directory: Path):
    values = []
    issues = []
    for path in sorted(directory.glob("*_10000.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            issues.append((path.name, f"json error: {exc}"))
            continue
        if TARGET_KEY not in data:
            issues.append((path.name, f"missing key {TARGET_KEY}"))
            continue
        value = data[TARGET_KEY]
        if not isinstance(value, (int, float)):
            issues.append((path.name, f"non-numeric value {value!r}"))
            continue
        values.append(value)
    return values, issues


def load_sw_values(directory: Path):
    """Extract the p-value (index 1) from Shapiro-Wilk outputs at key TARGET_KEY."""
    values = []
    issues = []
    for path in sorted(directory.glob("*_10000.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            issues.append((path.name, f"json error: {exc}"))
            continue
        if TARGET_KEY not in data:
            issues.append((path.name, f"missing key {TARGET_KEY}"))
            continue
        entry = data[TARGET_KEY]
        if not (isinstance(entry, list) and len(entry) > 1):
            issues.append((path.name, f"expected list with at least 2 items, got {entry!r}"))
            continue
        value = entry[0]
        if not isinstance(value, (int, float)):
            issues.append((path.name, f"non-numeric statistic {value!r}"))
            continue
        values.append(value)
    return values, issues


def summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: no usable values found"
    avg = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return f"{label}: n={len(values)}, mean={avg:.2e}, std={std:.2e}"


def main() -> None:
    ks_values, ks_issues = load_values(KS_DIR)
    wass_values, wass_issues = load_values(WASSERSTEIN_DIR)
    sw_values, sw_issues = load_sw_values(SW_DIR)

    print(summarize("KS_dist", ks_values))
    print(summarize("Wasserstein_dist", wass_values))
    print(summarize("Shapiro_Wilk_pvalue", sw_values))

    for label, issues in (
        ("KS_dist issues", ks_issues),
        ("Wasserstein_dist issues", wass_issues),
        ("Shapiro_Wilk issues", sw_issues),
    ):
        if issues:
            print(f"\n{label}:")
            for name, reason in issues:
                print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
