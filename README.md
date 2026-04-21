# pm_env

Execute RL task(s) for LLM in a containerized environment.

## Download UV to manage python packages
https://docs.astral.sh/uv/getting-started/installation/

## Install Podman or Docker
- [Podman](https://podman.io/)
    - podman cli installer
    - and podman desktop app
- [Docker](https://docs.docker.com/desktop/?_gl=1*1bqni31*_gcl_au*MTIzNTY1MDQxNS4xNzY5NzI5MTYz*_ga*MTkzMDg1NDk1Ny4xNzY5NzI5MTYz*_ga_XJWPQMJYHQ*czE3NzEzMTcxNDMkbzIkZzEkdDE3NzEzMTcxNDMkajYwJGwwJGgw) Docker Desktop > Setup > Install

## Create an Anthropic API key
- head to https://platform.claude.com to set this up and create a key
- add funds to the account so that your key will work ($5 should be good)

## Usage

### Prepare virtual environment

First, run

```bash
uv sync
```

You can add files to the venv using `uv add`.

### Create a run configuration

You'll need an anthropic API key for testing. (make a key from the anthropic developer console)

```bash
uv run pm_env create-run-config --model claude-haiku-4-5-20251001 --model-api-key $ANTHROPIC_API_KEY
```

This creates a `run_config.json` with default settings. You'll need to edit `run_config.json` to change the task ID. 

### start the podman machine (if using podman)
- after installing the podman cli, run "podman machine init", then "podman machine start"

### open docker desktop app (if using docker)
- make sure to open and run the docker desktop app

### Run a task

```bash
uv run pm_env run --config run_config.json
```

Transcripts get saved to `out/`.

By default, tasks run containerized using podman. Use `--runtime docker` if you want to use docker.

### Run multiple tasks in parallel

```bash
uv run pm_env run --config run_config.json --n-parallel 3
```

## Files to edit

### pyproject.toml

This contains the list of packages in the virtual environment. You can add packages as needed using `uv add`. 

### tasks.py

Modify [tasks.py](src/pm_env/tasks.py) to change the instructions, judge, task ID, etc.

### scoring_script.py

Modify [scoring_script.py](src/pm_env/scoring_script.py) to change the scoring script run by `ExecutableJudge`. 

### setup_data.py

Files greater than 5MB cannot be pushed to GitHub. Use [setup_data.py](setup_data.py) to download and pre-process large datasets and place the processed data in one of the following locations:

- `env_data`: Put data the model should be able to see into [env_data](env_data/). You can access this directory from inside `tasks.py` or inside the scoring script with `get_env_data_dir()`. 

- `scoring_data`: Put data the model should not be able to see (e.g., scoring data and scripts) into [scoring_data](scoring_data/). You can access this directory from inside the scoring script with `get_scoring_data_dir()`. 

### env_requirements.txt

Add any Python dependencies the model needs for solving tasks in `env_requirements.txt`. 

### ContainerFile

In some cases, you may have to add dependencies to the `ContainerFile`. 

### Judges

Find available judges in [src/pm_env/judges](src/pm_env/judges/). Currently `RegexJudge` and `ExecutableJudge` are used in `tasks.py`, but `RubricJudge` (i.e., LLM-as-a-judge) is also available.

### README.md

Please edit this README to document and explain your environment to Preference Model. Please explain what your environment does, what it's testing, and how it is being graded. Also let us know if the environment requires additional hardware (GPU/TPU).

# AI Usage Policy

AI agents are good at coding, but bad at creating RL environments. You are allowed to use AI agents if you would like, but if you rely heavily on AI agents then your environment will likely contain major mistakes. We encourage you to use tab-autocomplete, but discourage you from using AI coding agents except for minor implementation details.

We encourage you to tell us how you used AI tools in your README.

# Next steps

Read `MNIST_TUTORIAL.md`. 

# After creating your Env
- be sure to push all changes to the remote repo! This is so that we may see your work!
