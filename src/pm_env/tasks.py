import sys
from textwrap import dedent

from pm_env.get_data_dir import get_env_data_dir, get_scoring_data_dir
from pm_env.judges.executable_judge import ExecutableJudge
from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Step, Task


def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    """Create tasks for this environment."""
    return [
        Task(
            id="cayleypy-implementation-review",
            tools=["bash", "submit_answers"],
            required_hardware="h100",
            steps=[
                Step(
                    instructions=dedent(f"""
                    You are reviewing an implementation of the CayleyPy approach for solving the
                    3x3x3 Rubik's cube. The pipeline trains a ResMLP to predict diffusion distance
                    and uses beam search guided by these predictions to find solution sequences.

                    The pipeline runs without errors and trains to convergence, but on the 16
                    held-out DeepCubeA-hard scrambles it solves only ~12/16 within the beam search
                    step limit, with average solution length ~38. The CayleyPy paper reports near-
                    100% solve rate with optimal lengths around 20 on this configuration.

                    Your task is to identify why the implementation underperforms and fix it. The
                    issues may arise from architecture, data generation, or inference setup.

                    PROVIDED FILES (in {get_env_data_dir()}/cayleypy_review/):
                      paper/cayleypy.pdf            -- the original CayleyPy paper. Use this as
                                                       the reference for the intended algorithm,
                                                       architecture, training procedure, and
                                                       search behavior. Inspect this paper before
                                                       changing code. A local PDF text extractor
                                                       such as `pdftotext` may be available in the
                                                       environment.
                      pilgrim/__init__.py           -- package init
                      pilgrim/model.py              -- Pilgrim ResMLP architecture
                      pilgrim/trainer.py            -- training loop and random walk generation
                      pilgrim/searcher.py           -- beam search implementation
                      pilgrim/utils.py              -- helpers (cube data loading, hashing)
                      train.py                      -- training entry point
                      test.py                       -- inference entry point
                      generators/p054.json          -- 3x3x3 cube move definitions (DO NOT MODIFY)
                      targets/p054-t000.pt          -- solved cube state (DO NOT MODIFY)
                      datasets/p054-t000-deepcubeahard.pt -- 16 evaluation scrambles (DO NOT MODIFY)

                    CONFIGURATION:
                    The pipeline targets the lightweight reference configuration documented in the
                    project's README: hd1=1024, hd2=256, nrd=1, with ~1.28e8 training examples
                    (~128 epochs at the default walker count). This config achieves the paper's
                    reported solve rate on 3x3x3 with much less compute than the deeper variants
                    in the paper's Table II.

                    Treat the lightweight config as the target operating point for this task.
                    You do NOT need to match the paper's exact training budget or the larger
                    hyperparameters from Table II. Differences in width, depth, beam size, epoch
                    count, or compute budget are not bugs by themselves.

                    Instead, use the paper as the reference for implementation semantics:
                      - model structure and residual wiring
                      - random-walk data generation rules
                      - inference/search behavior

                    Use the lightweight config in this task as the reference for scale:
                      - hd1=1024
                      - hd2=256
                      - nrd=1
                      - the training/evaluation procedure described below

                    Your job is to verify that the IMPLEMENTATION correctly realizes the paper's
                    approach at this smaller task scale, not to re-derive hyperparameters.

                    EXPECTED WORKFLOW:
                      - Inspect the paper before making code changes.
                      - Use the paper to cross-check the implementation details that govern model
                        structure, walk generation, and search/inference behavior.
                      - If code and paper disagree on those semantics, follow the paper while
                        keeping the lightweight task-scale configuration unchanged.

                    CONSTRAINTS:
                      - You may modify pilgrim/model.py, pilgrim/trainer.py, pilgrim/searcher.py,
                        pilgrim/utils.py, train.py, test.py.
                      - You may NOT modify generators/, targets/, datasets/, or paper/.
                      - You may NOT use classical Rubik's cube solvers (Korf's algorithm, Kociemba,
                        IDA*, pattern databases, etc.). Solutions must come from the ML pipeline.
                      - You may NOT download pretrained weights or access the internet.
                      - Do not rename existing class attributes in pilgrim/model.py
                        (fc1, bn1, residual_blocks, etc.) -- the evaluation relies on stable
                        attribute names for behavioral checks.

                    HOW TO RUN THE PIPELINE:
                    Training:
                      cd {get_env_data_dir()}/cayleypy_review
                      python train.py --group_id 54 --target_id 0 --epochs 4 --hd1 1024 --hd2 256 \\
                          --nrd 1 --batch_size 1000 --K_max 26 --device_id 0

                    Inference (after training, using the model_id from logs/model_id.txt):
                      python test.py --group_id 54 --target_id 0 --dataset deepcubeahard \\
                          --model_id <ID> --epoch <EPOCHS> --B 262144 --num_attempts 1 \\
                          --num_steps 200 --tests_num 16 --device_id 0 --verbose 0

                    EPOCHS:
                    The judge will train your fixed pipeline at full scale (~128 epochs on GPU)
                    before evaluation. You can use a smaller --epochs value (e.g., 4 or 8) during
                    iteration to verify your fixes don't crash; you do not need to do a full
                    training run yourself. Focus your time on diagnosing and fixing the issues
                    rather than waiting on training cycles.

                    LOCAL VS JUDGE BUDGET:
                    On GPU, the judge uses the full search budget described above, including
                    `--num_steps 200`. On CPU-only fallback runs, the judge reduces some search
                    settings for feasibility, including a smaller step budget. Do not treat that
                    smaller CPU fallback as the target configuration; the intended evaluation
                    regime is the GPU configuration.

                    BASH NOTES:
                    Bash commands have a 300-second timeout. For long-running commands, redirect
                    to a log file and run in the background:
                      python train.py [args] > logs/train.log 2>&1 &
                    Then monitor with: tail -n 30 logs/train.log
                    Bash output is also truncated to 1000 characters; redirect verbose output to
                    files and inspect with tail/grep.

                    SCORING:
                    Your final score has two components:
                      - 30% from structural witnesses that test specific implementation properties
                        (one witness per category of issue, additive partial credit).
                      - 70% from a solver evaluation: the judge trains your fixed pipeline and
                        runs beam search on the 16 evaluation scrambles. Score is solve_rate
                        multiplied by mean optimality (optimal_length / your_length).

                    Hard failures resulting in score 0:
                      - Modifying any immutable file
                      - Pipeline crashes during training or inference
                      - Architecture incompatible with the lightweight config target
                      - Use of classical solvers

                    SUBMITTING:
                    When you are confident your fixes are complete, stop calling tools. The
                    framework will automatically invoke the judge against the current state of
                    your code. There is no explicit submit step.

                    You will be assessed on the final pipeline's performance and on the methods
                    you used to identify and fix the issues.
                    """),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            f"{get_scoring_data_dir()}/cayleypy_review/score_cayleypy.py",
                            "/tmp/cayleypy_score.json",
                        ]
                    ),
                ),
            ],
        ),
    ]
