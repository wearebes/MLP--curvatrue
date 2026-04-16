from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_CONFIG = REPO_ROOT / "test" / "fixtures" / "phase2_smoke" / "train_config.yaml"
PHASE3_CONFIG = REPO_ROOT / "test" / "fixtures" / "phase3_schedule" / "train_config.yaml"
PHASE3_SCHEDULE_DIR = REPO_ROOT / "test" / "fixtures" / "phase3_schedule" / "model" / "_schedules"


def run_command(command: list[str], *, expect_exit: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != expect_exit:
        message = [
            f"Command failed with exit={completed.returncode}, expected={expect_exit}",
            "Command: " + " ".join(command),
            "--- stdout ---",
            completed.stdout,
            "--- stderr ---",
            completed.stderr,
        ]
        raise RuntimeError("\n".join(message))
    return completed


def latest_schedule_summary() -> Path | None:
    candidates = sorted(PHASE3_SCHEDULE_DIR.glob("schedule_summary_*.json"))
    return candidates[-1] if candidates else None


def main() -> None:
    run_id = f"smoke_cpu_run_{int(time.time())}"

    print("[smoke] step 1/7 env-check")
    env_result = run_command([sys.executable, "-m", "train", "--env-check"])
    env_payload = json.loads(env_result.stdout)
    print(
        "[smoke] env-check ok | "
        f"python={env_payload['python']['version']} "
        f"cuda_available={env_payload['cuda']['cuda_available']}"
    )

    print("[smoke] step 2/7 invalid-config preflight path")
    invalid_config = run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--config",
            "does/not/exist.yaml",
            "--datasets",
            "tiny_ds",
            "--batch-sizes",
            "8",
            "--rho",
            "256",
            "--preflight-schedule",
        ],
        expect_exit=1,
    )
    invalid_config_payload = json.loads(invalid_config.stdout)
    if invalid_config_payload["status"] != "failed" or "config_error" not in invalid_config_payload:
        raise RuntimeError("Expected structured config failure report for invalid config path.")
    print("[smoke] invalid-config preflight path ok")

    print("[smoke] step 3/7 validate-data success path")
    validate_ok = run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--dataset",
            "tiny_ds",
            "--config",
            str(PHASE2_CONFIG),
            "--rho",
            "256",
            "266",
            "--validate-data",
        ]
    )
    validate_ok_payload = json.loads(validate_ok.stdout)
    if int(validate_ok_payload["failure_count"]) != 0:
        raise RuntimeError("Expected zero validation failures for tiny_ds.")
    print("[smoke] validate-data success path ok")

    print("[smoke] step 4/7 validate-data failure path")
    validate_fail = run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--dataset",
            "tiny_fail",
            "--config",
            str(PHASE3_CONFIG),
            "--rho",
            "256",
            "266",
            "--validate-data",
        ],
        expect_exit=1,
    )
    validate_fail_payload = json.loads(validate_fail.stdout)
    if int(validate_fail_payload["failure_count"]) == 0:
        raise RuntimeError("Expected validation failure for tiny_fail.")
    print("[smoke] validate-data failure path ok")

    print("[smoke] step 5/7 dataset-run cpu smoke")
    run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--dataset",
            "tiny_ds",
            "--config",
            str(PHASE2_CONFIG),
            "--run-id",
            run_id,
            "--rho",
            "256",
            "266",
            "--dataset-run",
            "--rho-max-concurrent",
            "2",
            "--allow-cpu",
            "--max-epochs",
            "1",
            "--patience",
            "1",
        ]
    )
    print(f"[smoke] dataset-run ok | run_id={run_id}")

    print("[smoke] step 6/7 verify-run")
    verify_run = run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--config",
            str(PHASE2_CONFIG),
            "--verify-run",
            "--run-id",
            run_id,
        ]
    )
    verify_run_payload = json.loads(verify_run.stdout)
    if verify_run_payload["status"] != "ok":
        raise RuntimeError("verify-run did not return ok.")
    print("[smoke] verify-run ok")

    print("[smoke] step 7/7 schedule-run downgrade smoke")
    before_summary = latest_schedule_summary()
    run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--config",
            str(PHASE3_CONFIG),
            "--datasets",
            "tiny_fail",
            "tiny_ok",
            "--batch-sizes",
            "8",
            "--rho",
            "256",
            "266",
            "--schedule-run",
            "--rho-max-concurrent",
            "2",
            "--allow-cpu",
        ],
        expect_exit=1,
    )
    after_summary = latest_schedule_summary()
    if after_summary is None or after_summary == before_summary:
        raise RuntimeError("Expected a new schedule summary after schedule-run smoke.")

    verify_schedule = run_command(
        [
            sys.executable,
            "-m",
            "train",
            "--verify-schedule-path",
            str(after_summary),
        ]
    )
    verify_schedule_payload = json.loads(verify_schedule.stdout)
    if verify_schedule_payload["status"] != "ok":
        raise RuntimeError("verify-schedule did not return ok.")
    print(f"[smoke] schedule-run downgrade ok | summary={after_summary.name}")

    print("[smoke] all CPU validation steps passed")


if __name__ == "__main__":
    main()
