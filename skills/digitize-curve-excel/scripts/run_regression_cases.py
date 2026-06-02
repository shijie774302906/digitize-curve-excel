from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "accuracy_runner.py"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def strip_confirmation(target: dict[str, Any]) -> None:
    for field in (
        "confirmed_by_user",
        "confirmation_source",
        "user_confirmation_text",
        "target_selection_note",
        "target_curves",
        "independent_axis",
        "dependent_axis",
        "line_style",
        "data_form",
        "continuity_required",
        "curve_form",
        "target_colors",
        "exclude_same_color_text",
        "single_value_axis",
    ):
        target.pop(field, None)


def confirm_depth_target(target: dict[str, Any], *, set_profile: bool, remove_unsafe_thresholds: bool) -> None:
    script_args = target.setdefault("script_args", {})
    preset = str(script_args.get("curve_preset", target.get("key", "target")))
    target.update(
        {
            "confirmed_by_user": True,
            "confirmation_source": "explicit_user_response",
            "user_confirmation_text": f"Regression fixture: user confirmed {target.get('name', target.get('key', 'target'))} as a solid continuous depth profile with one x per depth y and same-color text excluded.",
            "target_selection_note": f"Regression: digitize {target.get('name', target.get('key', 'target'))} as one x per depth y.",
            "target_curves": [{"name": str(target.get("name", target.get("key", "target"))), "color": preset}],
            "independent_axis": "y",
            "dependent_axis": "x",
            "line_style": "solid",
            "data_form": "continuous_curve",
            "continuity_required": True,
            "curve_form": "depth_profile_y_to_x",
            "target_colors": [preset],
            "exclude_same_color_text": True,
            "single_value_axis": "y_to_x",
        }
    )
    if set_profile:
        script_args["trace_mode"] = "profile"
    if remove_unsafe_thresholds:
        audit = target.setdefault("audit", {})
        for key in ("max_duplicate_depth_rows", "max_profile_gap_rows", "min_profile_row_coverage"):
            audit.pop(key, None)


def confirm_xy_target(target: dict[str, Any]) -> None:
    target.update(
        {
            "confirmed_by_user": True,
            "confirmation_source": "explicit_user_response",
            "user_confirmation_text": f"Regression fixture: user confirmed {target.get('name', target.get('key', 'target'))} as a solid continuous normal XY curve.",
            "target_selection_note": f"Regression: digitize {target.get('name', target.get('key', 'target'))} as a normal XY curve.",
            "target_curves": [{"name": str(target.get("name", target.get("key", "target"))), "color": str(target.get("key", "target"))}],
            "independent_axis": "x",
            "dependent_axis": "y",
            "line_style": "solid",
            "data_form": "continuous_curve",
            "continuity_required": True,
            "curve_form": "normal_xy_x_to_y",
            "target_colors": [target.get("key", "target")],
            "exclude_same_color_text": False,
            "single_value_axis": "x_to_y",
        }
    )


def case_config_path(benchmark_root: Path, case_id: str) -> Path:
    return benchmark_root / "benchmark_runs" / case_id / "case_config.json"


def base_config(benchmark_root: Path, source_case: str, case_name: str, work_dir: Path) -> dict[str, Any]:
    cfg = read_json(case_config_path(benchmark_root, source_case))
    cfg["case_id"] = case_name
    cfg["out_dir"] = str((work_dir / case_name).resolve())
    return cfg


def prepare_image_copy_3_bad(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_3", "image_copy_3_bad_old_config", work_dir)
    for target in cfg["targets"]:
        confirm_depth_target(target, set_profile=False, remove_unsafe_thresholds=False)
        target.setdefault("script_args", {})["trace_mode"] = "trend-profile"
        audit = target.setdefault("audit", {})
        audit["max_duplicate_depth_rows"] = 999
        audit["max_profile_gap_rows"] = 999
        audit["min_profile_row_coverage"] = 0.1
    return cfg


def prepare_image_copy_3_depth_profile_baseline(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_3", "image_copy_3_depth_profile_baseline", work_dir)
    for target in cfg["targets"]:
        confirm_depth_target(target, set_profile=True, remove_unsafe_thresholds=True)
    return cfg


def prepare_image_copy_5_current(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_5", "image_copy_5_current", work_dir)
    for target in cfg["targets"]:
        confirm_depth_target(target, set_profile=True, remove_unsafe_thresholds=True)
    return cfg


def prepare_image_2_simple_depth(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_2", "image_2_simple_depth", work_dir)
    for target in cfg["targets"]:
        confirm_depth_target(target, set_profile=True, remove_unsafe_thresholds=True)
    return cfg


def prepare_same_color_unexcluded(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_3", "same_color_label_unexcluded", work_dir)
    cfg["targets"] = [cfg["targets"][0]]
    target = cfg["targets"][0]
    confirm_depth_target(target, set_profile=True, remove_unsafe_thresholds=True)
    target["script_args"].pop("exclude_rects", None)
    target["forbidden_regions"] = []
    return cfg


def prepare_ambiguous_no_confirmation(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_3", "ambiguous_target_no_confirmation", work_dir)
    for target in cfg["targets"]:
        strip_confirmation(target)
    return cfg


def prepare_image_copy_2_multi_xy(benchmark_root: Path, work_dir: Path) -> dict[str, Any]:
    cfg = base_config(benchmark_root, "image_copy_2", "image_copy_2_multi_xy", work_dir)
    for target in cfg["targets"]:
        strip_confirmation(target)
    return cfg


CASES: dict[str, dict[str, Any]] = {
    "image_copy_3_bad_old_config": {
        "prepare": prepare_image_copy_3_bad,
        "expected": "PASS",
        "next_action": "ready_to_publish",
        "codes": set(),
        "requires_auto_repair": True,
    },
    "image_copy_3_depth_profile_baseline": {
        "prepare": prepare_image_copy_3_depth_profile_baseline,
        "expected": "PASS",
        "next_action": "ready_to_publish",
        "codes": set(),
    },
    "image_copy_5_current": {
        "prepare": prepare_image_copy_5_current,
        "expected": "PASS",
        "next_action": "ready_to_publish",
        "codes": set(),
    },
    "image_2_simple_depth": {
        "prepare": prepare_image_2_simple_depth,
        "expected": "PASS",
        "next_action": "ready_to_publish",
        "codes": set(),
    },
    "same_color_label_unexcluded": {
        "prepare": prepare_same_color_unexcluded,
        "expected": "FAIL",
        "next_action": "ask_user",
        "codes": {"strict_depth_extra_runs", "strict_unclassified_same_color_component", "profile_report_check"},
    },
    "ambiguous_target_no_confirmation": {
        "prepare": prepare_ambiguous_no_confirmation,
        "expected": "FAIL",
        "next_action": "ask_user",
        "codes": {"missing_confirmation"},
    },
    "image_copy_2_multi_xy": {
        "prepare": prepare_image_copy_2_multi_xy,
        "expected": "FAIL",
        "next_action": "ask_user",
        "codes": {"missing_confirmation"},
    },
}


def failure_codes(audit: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for target in audit.get("targets", []):
        for item in target.get("failures", []):
            codes.add(str(item.get("code", "")))
    return codes


def assert_final_artifacts(audit: dict[str, Any], expected: str) -> list[str]:
    errors: list[str] = []
    out_dir = Path(str(audit["out_dir"]))
    final_dir = out_dir / "final"
    if expected == "FAIL":
        if final_dir.exists() and any(final_dir.rglob("*")):
            errors.append(f"FAIL case left final artifacts under {final_dir}")
        return errors
    required = {"result.xlsx", "overlay.png", "redrawn.png", "audit.json", "strict_qa.json", "strict_qa_overlay.png"}
    for target in audit.get("targets", []):
        target_dir = final_dir / str(target["target_key"])
        missing = sorted(name for name in required if not (target_dir / name).exists())
        if missing:
            errors.append(f"{target['target_key']} missing final artifacts: {', '.join(missing)}")
    return errors


def assert_next_steps(audit: dict[str, Any], expected_action: str) -> list[str]:
    errors: list[str] = []
    actual_action = str(audit.get("next_action", ""))
    if actual_action != expected_action:
        errors.append(f"expected next_action={expected_action}, got {actual_action}")
    next_steps = Path(str(audit.get("next_steps", "")))
    if not next_steps.exists():
        errors.append(f"missing next_steps.md at {next_steps}")
        return errors
    text = next_steps.read_text(encoding="utf-8", errors="replace")
    if f"Workflow next action: {expected_action}" not in text:
        errors.append(f"next_steps.md does not contain expected action {expected_action}")
    if expected_action == "ask_user" and "User-facing response:" not in text:
        errors.append("ask_user case did not produce a user-facing response")
    if expected_action == "ready_to_publish" and "Final artifacts:" not in text:
        errors.append("ready_to_publish case did not list final artifacts")
    return errors


def assert_auto_repair(out_dir: Path) -> list[str]:
    if any(out_dir.glob("attempts/*/auto_repair_1")):
        return []
    return [f"expected an auto_repair_1 attempt under {out_dir / 'attempts'}"]


def run_case(name: str, benchmark_root: Path, work_dir: Path, keep_outputs: bool) -> tuple[bool, str]:
    if name not in CASES:
        return False, f"unknown case {name!r}"
    spec = CASES[name]
    out_dir = work_dir / name
    if out_dir.exists() and not keep_outputs:
        shutil.rmtree(out_dir)
    cfg = spec["prepare"](benchmark_root, work_dir)
    config_path = work_dir / f"{name}.json"
    write_json(config_path, cfg)

    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config_path), "--strict-exit"],
        cwd=str(SCRIPT_DIR.parent),
        text=True,
        capture_output=True,
    )
    audit_path = out_dir / "audit.json"
    if not audit_path.exists():
        return False, f"{name}: missing audit.json; returncode={proc.returncode}; stderr={proc.stderr.strip()}"
    audit = read_json(audit_path)
    expected = str(spec["expected"])
    errors: list[str] = []
    if audit.get("status") != expected:
        errors.append(f"expected {expected}, got {audit.get('status')}")
    if expected == "PASS" and proc.returncode != 0:
        errors.append(f"PASS case returned {proc.returncode}")
    if expected == "FAIL" and proc.returncode == 0:
        errors.append("FAIL case returned 0 with --strict-exit")
    codes = failure_codes(audit)
    expected_codes: set[str] = set(spec["codes"])
    if expected_codes and not (codes & expected_codes):
        errors.append(f"expected one of failure codes {sorted(expected_codes)}, got {sorted(codes)}")
    if bool(spec.get("requires_auto_repair")):
        errors.extend(assert_auto_repair(out_dir))
    errors.extend(assert_final_artifacts(audit, expected))
    errors.extend(assert_next_steps(audit, str(spec["next_action"])))
    if errors:
        return False, f"{name}: " + "; ".join(errors)
    return True, f"{name}: {expected} ({', '.join(sorted(codes)) or 'no failures'})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run digitize-curve-excel accuracy regression cases.")
    parser.add_argument("--benchmark-root", default=".", help="Workspace containing benchmark_runs/ fixtures.")
    parser.add_argument("--work-dir", default="benchmark_runs/skill_regression_tmp", help="Temporary output directory.")
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="Case to run. Repeatable. Defaults to all cases.")
    parser.add_argument("--keep-outputs", action="store_true", help="Do not delete an existing temporary output directory first.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    names = args.case or list(CASES)
    failed = False
    for name in names:
        ok, message = run_case(name, benchmark_root, work_dir, args.keep_outputs)
        print(("PASS " if ok else "FAIL ") + message)
        failed = failed or not ok
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
