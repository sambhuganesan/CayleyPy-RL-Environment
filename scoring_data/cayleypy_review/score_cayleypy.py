import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from pm_env.get_data_dir import get_env_data_dir, get_scoring_data_dir


STAGE2_EPOCHS = 128 if torch.cuda.is_available() else 4
STAGE2_BATCH_SIZE = 1000
STAGE2_K_MAX = 26
STAGE2_BEAM_WIDTH_CPU = 1024
STAGE2_BEAM_WIDTH_GPU = 2 ** 18
STAGE2_NUM_STEPS_CPU = 50
STAGE2_NUM_STEPS_GPU = 200
STAGE2_NUM_ATTEMPTS = 1
STAGE2_TRAIN_TIMEOUT = 1200 if torch.cuda.is_available() else 480
STAGE2_TEST_TIMEOUT = 600
N_TEST_SCRAMBLES = 16
N_STRUCTURAL_WITNESSES = 5
MODEL_DEPENDENCE_TESTS = 4
MODEL_DEPENDENCE_BEAM_WIDTH_CPU = 1024
MODEL_DEPENDENCE_BEAM_WIDTH_GPU = 4096
MODEL_DEPENDENCE_NUM_STEPS_CPU = 50
MODEL_DEPENDENCE_NUM_STEPS_GPU = 100


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_result(output_path, score, metadata):
    Path(output_path).write_text(json.dumps({
        "score": float(score),
        "metadata": {k: str(v) for k, v in metadata.items()},
    }))


def load_optimal_lengths():
    path = Path(get_scoring_data_dir()) / "cayleypy_review" / "deepcubeahard_optimal.json"
    return json.loads(path.read_text())["optimal_lengths"]


def check_file_integrity(workdir, metadata):
    expected_path = Path(get_scoring_data_dir()) / "cayleypy_review" / "expected_hashes.json"
    try:
        expected = json.loads(expected_path.read_text())
    except Exception as e:
        metadata["integrity_error"] = f"Expected hashes unreadable: {e}"
        return False

    for rel, expected_hash in expected.items():
        full = workdir / rel
        if not full.exists():
            metadata["integrity_error"] = f"Missing file: {rel}"
            return False
        if file_sha256(full) != expected_hash:
            metadata["integrity_error"] = f"Hash mismatch: {rel}"
            return False
    return True


def witness_skip_connection(workdir):
    try:
        sys.path.insert(0, str(workdir))
        try:
            model_module = _import_from_path("agent_model", workdir / "pilgrim" / "model.py")
        finally:
            sys.path.pop(0)

        Pilgrim = model_module.Pilgrim
        model = Pilgrim(num_classes=6, state_size=54, hd1=64, hd2=64, nrd=1)

        if not hasattr(model, "residual_blocks") or model.residual_blocks is None:
            return False, "Pilgrim has no residual_blocks (nrd=0?)"
        if len(model.residual_blocks) == 0:
            return False, "residual_blocks list is empty"

        block = model.residual_blocks[0]
        with torch.no_grad():
            block.fc2.weight.zero_()
            block.fc2.bias.zero_()
            block.bn2.weight.zero_()
            block.bn2.bias.zero_()
            block.bn2.running_mean.zero_()
            block.bn2.running_var.fill_(1.0)
        block.eval()

        torch.manual_seed(0)
        x = torch.randn(8, 64)
        with torch.no_grad():
            out = block(x)

        if torch.allclose(out, F.relu(x), atol=1e-4):
            return True, "skip wraps the block correctly"
        else:
            return False, "skip appears wired internally"
    except Exception as e:
        return False, f"witness error: {type(e).__name__}: {e}"


def witness_non_backtracking(workdir):
    try:
        sys.path.insert(0, str(workdir))
        try:
            pkg = _import_from_path("agent_pilgrim_back", workdir / "pilgrim" / "__init__.py")
        finally:
            sys.path.pop(0)

        Pilgrim = pkg.Pilgrim
        Trainer = pkg.Trainer
        generate_inverse_moves = pkg.generate_inverse_moves

        with open(workdir / "generators" / "p054.json") as f:
            gen_data = json.load(f)
        all_moves = torch.tensor(gen_data["actions"], dtype=torch.int64)
        inverse_moves = torch.tensor(generate_inverse_moves(gen_data["names"]), dtype=torch.int64)
        V0 = torch.load(workdir / "targets" / "p054-t000.pt", weights_only=True, map_location="cpu")

        device = torch.device("cpu")
        model = Pilgrim(num_classes=6, state_size=54, hd1=64, hd2=64, nrd=1).to(device)
        trainer = Trainer(
            net=model, num_epochs=1, device=device, batch_size=100, lr=0.001,
            name="witness", K_min=1, K_max=20,
            all_moves=all_moves.to(device),
            inverse_moves=inverse_moves.to(device),
            V0=V0.to(device),
        )

        recorded = []
        original = trainer.do_random_step

        def instrumented(states, last_moves):
            new_states, next_moves = original(states, last_moves)
            recorded.append((last_moves.clone(), next_moves.clone()))
            return new_states, next_moves

        trainer.do_random_step = instrumented
        trainer.generate_random_walks(k=50, K_min=1, K_max=20)

        violations = 0
        total = 0
        for last, nxt in recorded:
            mask = last >= 0
            if not mask.any():
                continue
            last_v = last[mask]
            nxt_v = nxt[mask]
            inv_last = inverse_moves[last_v]
            violations += (nxt_v == inv_last).sum().item()
            total += mask.sum().item()

        if total == 0:
            return False, "no move pairs available to check"
        if violations == 0:
            return True, f"no backtracks across {total} pairs"
        return False, f"{violations}/{total} backtracking moves"
    except Exception as e:
        return False, f"witness error: {type(e).__name__}: {e}"


def witness_data_refreshes(workdir):
    try:
        sys.path.insert(0, str(workdir))
        try:
            pkg = _import_from_path("agent_pilgrim_refresh", workdir / "pilgrim" / "__init__.py")
        finally:
            sys.path.pop(0)

        Pilgrim = pkg.Pilgrim
        Trainer = pkg.Trainer
        generate_inverse_moves = pkg.generate_inverse_moves

        with open(workdir / "generators" / "p054.json") as f:
            gen_data = json.load(f)
        all_moves = torch.tensor(gen_data["actions"], dtype=torch.int64)
        inverse_moves = torch.tensor(generate_inverse_moves(gen_data["names"]), dtype=torch.int64)
        V0 = torch.load(workdir / "targets" / "p054-t000.pt", weights_only=True, map_location="cpu")

        device = torch.device("cpu")
        model = Pilgrim(num_classes=6, state_size=54, hd1=64, hd2=64, nrd=1).to(device)
        trainer = Trainer(
            net=model, num_epochs=1, device=device, batch_size=100, lr=0.001,
            name="witness", K_min=1, K_max=10,
            all_moves=all_moves.to(device),
            inverse_moves=inverse_moves.to(device),
            V0=V0.to(device),
        )

        X1, _ = trainer.generate_random_walks(k=50, K_min=1, K_max=10)
        X2, _ = trainer.generate_random_walks(k=50, K_min=1, K_max=10)

        differ = (X1 != X2).any(dim=1).sum().item()
        total = X1.size(0)
        rate = differ / total if total > 0 else 0.0

        if rate >= 0.99:
            return True, f"{differ}/{total} states differ ({rate:.1%})"
        return False, f"only {differ}/{total} states differ; data does not refresh"
    except Exception as e:
        return False, f"witness error: {type(e).__name__}: {e}"


def witness_fp16_inference(workdir):
    if not torch.cuda.is_available():
        return False, "CUDA not available; fp16 witness inconclusive"
    try:
        source = (workdir / "test.py").read_text()
        if ".half()" in source or "torch.float16" in source:
            return True, "test.py contains fp16 conversion"
        return False, "test.py does not convert model to fp16"
    except Exception as e:
        return False, f"witness error: {type(e).__name__}: {e}"


def _load_test_results(log_path: Path):
    results = json.loads(log_path.read_text())
    solved = [
        (entry["test_num"], entry["solution_length"])
        for entry in results
        if entry.get("solution_length") is not None
    ]
    if solved:
        avg_len = sum(length for _, length in solved) / len(solved)
    else:
        avg_len = None
    return solved, avg_len


def _run_inference_subset(workdir, model_id, epoch, beam_width, num_steps, tests_num):
    test_cmd = [
        sys.executable, "test.py",
        "--group_id", "54",
        "--target_id", "0",
        "--dataset", "deepcubeahard",
        "--model_id", str(model_id),
        "--epoch", str(epoch),
        "--B", str(beam_width),
        "--num_attempts", "1",
        "--num_steps", str(num_steps),
        "--tests_num", str(tests_num),
        "--device_id", "0",
        "--verbose", "0",
    ]
    try:
        r = subprocess.run(
            test_cmd, cwd=str(workdir),
            capture_output=True, text=True, timeout=STAGE2_TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, "inference timed out", None, None

    if r.returncode != 0:
        return False, "inference crashed", r.stderr[-500:], None

    log_name = f"test_p054-t000-deepcubeahard_{model_id}_{epoch}_B{beam_width}.json"
    log_path = workdir / "logs" / log_name
    if not log_path.exists():
        return False, f"test log missing: {log_name}", None, None

    solved, avg_len = _load_test_results(log_path)
    return True, solved, avg_len, log_path


def witness_model_dependence(workdir, model_id, epoch):
    try:
        if torch.cuda.is_available():
            beam_width = MODEL_DEPENDENCE_BEAM_WIDTH_GPU
            num_steps = MODEL_DEPENDENCE_NUM_STEPS_GPU
        else:
            beam_width = MODEL_DEPENDENCE_BEAM_WIDTH_CPU
            num_steps = MODEL_DEPENDENCE_NUM_STEPS_CPU

        ok, real_or_err, real_avg_len, _ = _run_inference_subset(
            workdir,
            model_id=model_id,
            epoch=epoch,
            beam_width=beam_width,
            num_steps=num_steps,
            tests_num=MODEL_DEPENDENCE_TESTS,
        )
        if not ok:
            return False, f"real-model check failed: {real_or_err}"

        real_solved = real_or_err
        real_count = len(real_solved)

        original_info = workdir / "logs" / f"model_p054-t000_{model_id}.json"
        original_weights = workdir / "weights" / f"p054-t000_{model_id}_e{epoch:05d}.pth"
        if not original_info.exists() or not original_weights.exists():
            return False, "missing trained checkpoint artifacts"

        corrupted_model_id = int(model_id) + 900000000
        corrupted_info = workdir / "logs" / f"model_p054-t000_{corrupted_model_id}.json"
        corrupted_weights = workdir / "weights" / f"p054-t000_{corrupted_model_id}_e{epoch:05d}.pth"

        info = json.loads(original_info.read_text())
        info["model_id"] = corrupted_model_id
        corrupted_info.write_text(json.dumps(info, indent=4))

        state = torch.load(original_weights, weights_only=False, map_location="cpu")
        corrupted_state = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                corrupted_state[key] = torch.zeros_like(value)
            else:
                corrupted_state[key] = value
        torch.save(corrupted_state, corrupted_weights)

        try:
            ok, corrupt_or_err, corrupt_avg_len, _ = _run_inference_subset(
                workdir,
                model_id=corrupted_model_id,
                epoch=epoch,
                beam_width=beam_width,
                num_steps=num_steps,
                tests_num=MODEL_DEPENDENCE_TESTS,
            )
            if not ok:
                return False, f"corrupted-model check failed: {corrupt_or_err}"

            corrupt_solved = corrupt_or_err
            corrupt_count = len(corrupt_solved)

            if real_count > corrupt_count:
                return True, (
                    f"real model solved {real_count}/{MODEL_DEPENDENCE_TESTS}, "
                    f"corrupted solved {corrupt_count}/{MODEL_DEPENDENCE_TESTS}"
                )

            if (
                real_count > 0
                and corrupt_count > 0
                and real_avg_len is not None
                and corrupt_avg_len is not None
                and real_avg_len + 2.0 <= corrupt_avg_len
            ):
                return True, (
                    f"real and corrupted solved {real_count}/{MODEL_DEPENDENCE_TESTS}, "
                    f"but corrupted paths were longer ({real_avg_len:.1f} vs {corrupt_avg_len:.1f})"
                )

            return False, (
                f"solver showed weak model dependence: real {real_count}/{MODEL_DEPENDENCE_TESTS}"
                + (
                    f" at avg length {real_avg_len:.1f}"
                    if real_avg_len is not None else ""
                )
                + ", corrupted "
                + f"{corrupt_count}/{MODEL_DEPENDENCE_TESTS}"
                + (
                    f" at avg length {corrupt_avg_len:.1f}"
                    if corrupt_avg_len is not None else ""
                )
            )
        finally:
            corrupted_info.unlink(missing_ok=True)
            corrupted_weights.unlink(missing_ok=True)
    except Exception as e:
        return False, f"witness error: {type(e).__name__}: {e}"


def run_solver_stage(workdir, metadata, optimal_lengths):
    try:
        if torch.cuda.is_available():
            beam_width = STAGE2_BEAM_WIDTH_GPU
            num_steps = STAGE2_NUM_STEPS_GPU
        else:
            beam_width = STAGE2_BEAM_WIDTH_CPU
            num_steps = STAGE2_NUM_STEPS_CPU

        train_cmd = [
            sys.executable, "train.py",
            "--group_id", "54",
            "--target_id", "0",
            "--epochs", str(STAGE2_EPOCHS),
            "--hd1", "1024",
            "--hd2", "256",
            "--nrd", "1",
            "--batch_size", str(STAGE2_BATCH_SIZE),
            "--K_max", str(STAGE2_K_MAX),
            "--device_id", "0",
        ]
        try:
            r = subprocess.run(
                train_cmd, cwd=str(workdir),
                capture_output=True, text=True, timeout=STAGE2_TRAIN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            metadata["solver_error"] = "training timed out"
            return 0.0
        if r.returncode != 0:
            metadata["solver_error"] = "training crashed"
            metadata["train_stderr"] = r.stderr[-500:]
            return 0.0

        model_id_path = workdir / "logs" / "model_id.txt"
        if not model_id_path.exists():
            metadata["solver_error"] = "no model_id produced"
            return 0.0
        model_id = model_id_path.read_text().strip().split("\n")[-1]

        test_cmd = [
            sys.executable, "test.py",
            "--group_id", "54",
            "--target_id", "0",
            "--dataset", "deepcubeahard",
            "--model_id", model_id,
            "--epoch", str(STAGE2_EPOCHS),
            "--B", str(beam_width),
            "--num_attempts", str(STAGE2_NUM_ATTEMPTS),
            "--num_steps", str(num_steps),
            "--tests_num", str(N_TEST_SCRAMBLES),
            "--device_id", "0",
            "--verbose", "0",
        ]
        try:
            r = subprocess.run(
                test_cmd, cwd=str(workdir),
                capture_output=True, text=True, timeout=STAGE2_TEST_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            metadata["solver_error"] = "inference timed out"
            return 0.0
        if r.returncode != 0:
            metadata["solver_error"] = "inference crashed"
            metadata["test_stderr"] = r.stderr[-500:]
            return 0.0

        log_name = f"test_p054-t000-deepcubeahard_{model_id}_{STAGE2_EPOCHS}_B{beam_width}.json"
        log_path = workdir / "logs" / log_name
        if not log_path.exists():
            metadata["solver_error"] = f"test log missing: {log_name}"
            return 0.0

        results = json.loads(log_path.read_text())
        solved = []
        for entry in results:
            if entry.get("solution_length") is not None:
                solved.append((entry["test_num"], entry["solution_length"]))

        solve_rate = len(solved) / N_TEST_SCRAMBLES

        if not solved:
            mean_optimality = 0.0
            mean_solution_length = 0.0
        else:
            mean_solution_length = sum(agent_len for _, agent_len in solved) / len(solved)
            ratios = [
                optimal_lengths[idx] / agent_len
                for idx, agent_len in solved
                if 0 <= idx < N_TEST_SCRAMBLES and agent_len > 0
            ]
            mean_optimality = sum(ratios) / len(ratios) if ratios else 0.0

        metadata["solve_rate"] = f"{solve_rate:.3f}"
        metadata["mean_optimality"] = f"{mean_optimality:.3f}"
        metadata["mean_solution_length"] = f"{mean_solution_length:.3f}"
        metadata["solved_count"] = f"{len(solved)}/{N_TEST_SCRAMBLES}"
        metadata["trained_model_id"] = str(model_id)

        return solve_rate * mean_optimality, int(model_id)
    except Exception as e:
        metadata["solver_error"] = f"unexpected: {type(e).__name__}: {e}"
        return 0.0, None


def main(output_path):
    workdir = Path(get_env_data_dir()) / "cayleypy_review"
    metadata = {}

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if not check_file_integrity(workdir, metadata):
        write_result(output_path, 0.0, metadata)
        return

    try:
        optimal_lengths = load_optimal_lengths()
    except Exception as e:
        metadata["scoring_error"] = f"could not load optimal lengths: {e}"
        write_result(output_path, 0.0, metadata)
        return

    structural_score = 0.0
    witness_credit = 0.30 / N_STRUCTURAL_WITNESSES
    witnesses = [
        ("bug1_skip_connection", witness_skip_connection),
        ("bug2_non_backtracking", witness_non_backtracking),
        ("bug3_data_refreshes", witness_data_refreshes),
        ("bug4_fp16_inference", witness_fp16_inference),
    ]
    for name, fn in witnesses:
        passed, info = fn(workdir)
        metadata[name] = ("PASS: " if passed else "FAIL: ") + info
        if passed:
            structural_score += witness_credit

    solver_score, trained_model_id = run_solver_stage(workdir, metadata, optimal_lengths)

    if trained_model_id is not None:
        passed, info = witness_model_dependence(workdir, trained_model_id, STAGE2_EPOCHS)
    else:
        passed, info = False, "solver stage did not produce a trained model"
    metadata["bug5_model_dependence"] = ("PASS: " if passed else "FAIL: ") + info
    if passed:
        structural_score += witness_credit

    final_score = 0.3 * structural_score + 0.7 * solver_score
    metadata["structural_score"] = f"{structural_score:.3f}"
    metadata["solver_score"] = f"{solver_score:.3f}"
    metadata["final_score"] = f"{final_score:.3f}"

    write_result(output_path, final_score, metadata)


if __name__ == "__main__":
    main(sys.argv[1])
