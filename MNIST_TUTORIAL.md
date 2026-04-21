In this tutorial, we’ll create a simple RL environment in which a model is tasked with training MNIST using PyTorch.

Training an MNIST classifier is, of course, a fairly trivial task for current state-of-the-art LLMs. The goal of this tutorial is to understand **how** to build an RL environment inside `pm_env_slim`. 

## Installing environment dependencies

We’ll use `torchvision` to download the MNIST dataset. The judge will also need to use `torch` to load and evaluate the trained model.

Since `torch` and `torchvision` aren’t in the `pyproject.toml` dependencies, we’ll need to add them to `pyproject.toml`.

Run the command

```bash
uv sync
```

This creates a virtual environment based on the dependencies in the `pyproject.toml` file. If you want to use the packages in `pyproject.toml` for local testing, you can activate the venv with

```bash
source .venv/bin/activate
```

**Installing torch[gpu]**

Add `torch` and `torchvision` to the `pyproject.toml` dependencies manually or using the command

```bash
uv add torch torchvision
```

## Installing model dependencies

The packages in `pyproject.toml` have not yet been made available to the model, because the model will be running in a virtual environment inside the container. 

To make python packages available to the model, we want to change `env_requirements.txt` as shown below:

```python
# Add any Python dependencies the model needs for solving tasks
torch
torchvision
```

## Downloading MNIST train and test datasets

We will need to put the MNIST train dataset in the `env_data` folder, and the full MNIST dataset in the `scoring_data` folder. This makes sure that the model doesn’t have access to the test dataset in the `env_data` folder.

To do this, edit the file `setup_data.py` . This file will download the data into the appropriate folders and delete the testing data from `env_data`. `setup_data.py` should look like:

```python
# This script gets executed on the build machine that builds your environment,
# but before the environment is created. That means you cannot depend on system
# packages that are not installed on the build machine, so keep it simple.
# Add any dependencies required for data setup in the section below.

# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch", "torchvision"]
# ///

import glob
import os

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def main():
    train_ds = MNIST("env_data", train=True, transform=ToTensor(), download=True)
    test_ds = MNIST("scoring_data", train=False, transform=ToTensor(), download=True)

    # delete test files from env_data
    for file in glob.glob("env_data/MNIST/raw/t10k*"):
        os.remove(file)

if __name__ == "__main__":
    main()
```

We can then run `setup_data.py` locally to download the MNIST data into the appropriate folders. This should be done with the bash command

```bash
uv run setup_data.py
```

These files should not be committed to GitHub. 

## Implement task

Now we need to edit the `get_tasks` function in `tasks.py`. This is where we explain to the model what it needs to do, and create the `Judge` we will use to evaluate it.In this task, there is only one `Step`. We modify the `instructions` and create a `Judge`.

```python
def get_tasks(config: EvaluationRunConfig) -> list[Task]:
    """Create tasks for this environment."""
    module_name = "classifier.py"
    checkpoint_name = "mnist-classifier.pt"

    return [
        Task(
            id="train-mnist",
            # Check out tool implementations in the `tools` directory
            tools=["bash, submit_answers"],
            steps=[
                Step(
                    instructions=dedent(f"""
                    Your task is to train a machine learning model that classifies
                    MNIST digits. Your goal is to achieve an accuracy above 95%.
                    Your cumulative time limit to complete the task is 5 minutes.
                    
                    You have access to the MNIST training data in {get_env_data_dir()}.
                    However, you do not have access to the MNIST testing data. This means
                    that torchvision.datasets.MNIST will not work. You will need to write
                    your own custom Dataset.
                    
                    You should save two files. The first file should be named
                    {module_name} and contain a torch class MNISTClassifier.
                    The second file {checkpoint_name} should contain the state_dict.
                    """),
                    judge=ExecutableJudge(
                        [
                            sys.executable,
                            f"{get_scoring_data_dir()}/score_mnist_classifier.py",
                            f"{get_env_data_dir()}/{module_name}",
                            f"{get_env_data_dir()}/{checkpoint_name}",
                            "/tmp/mnist_classifier_results.txt",
                        ]
                    ),
                ),
            ],
        ),
    ]

```

The instructions tell the model to train a PyTorch model and save it as `mnist-classifier.pt` in the `env_data` dir. To ensure the model can be loaded by the scoring script, it also saves the subclass of `nn.Module` in a file `classifier.py`.

We use an `ExecutableJudge`. This judge is intended to allow easy evaluation of model-written code or other files that the model saves into its workdir. For more information, see the docstring in `executable_judge.py`.

## Write scoring script

Our `ExecutableJudge` allows us to run any scoring script to evaluate the model, so long as the script outputs a txt file with a numerical score.

Our scoring function, as mentioned above, is called `score_mnist_classifier.py`. We put this in the `/scoring_data` folder.

```python
import importlib.util
import json
import sys

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from pm_env.get_data_dir import get_scoring_data_dir

def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_eval_accuracy(model):
    test_ds = MNIST(get_scoring_data_dir(), train=False, transform=ToTensor(), download=False)
    test_dl = DataLoader(test_ds, batch_size=256)

    num_accurate = 0

    for X, y in test_dl:
        y_hat = model(X)
        num_accurate += (torch.argmax(y_hat, dim=-1) == y).sum()
    
    return (num_accurate / len(test_ds)).item()

if __name__ == '__main__':
    try:
        classifier_module = _import_from_path('MNISTClassifier', sys.argv[1])
    except Exception as e:
        score = 0
        metadata = {
            'error': 'Cannot import module',
            'exception': str(e),
        }
    else:
        try:
            model = classifier_module.MNISTClassifier()
        except Exception as e:
            score = 0
            metadata = {
                'error': 'Cannot create model from module',
                'exception': str(e),
                'module': classifier_module.__name__,
            }
        else:
            try:
                model.load_state_dict(torch.load(sys.argv[2], weights_only=True))
            except Exception as e:
                score = 0
                metadata = {
                    'error': 'Failed to load model',
                    'exception': str(e),
                }
            else:
                try:
                    score = get_eval_accuracy(model)
                    metadata = {'error': 'None'}
                except Exception as e:
                    score = 0
                    metadata = {
                    'error': 'Failed to evaluate dataset',
                    'dataset dir': str(get_scoring_data_dir()),
                    'exception': str(e),
                }

    Path(sys.argv[-1]).write_text(json.dumps({"score": score, "metadata": {}}))

```

Since `/scoring_data` is ignored by default in `.gitignore`, you will need to add this script to `.gitignore` to ensure it can be added to GitHub. This can be done as follows:

```
scoring_data/**
!scoring_data/.gitkeep
+ !scoring_data/score_d2l_task_1.py
+ !scoring_data/torch_d2l.py
```

## Prepare run config file

You will use a configuration file `run_config.json` for evaluating the RL environment. This file can be created using the command

```bash
uv run pm_env create-run-config --model claude-haiku-4-5-20251001 --model-api-key $ANTHROPIC_API_KEY
```

You should now see that `run_config.json` has been created. You should change `task-id` to the current task ID, in this case `"train-mnist"`.

## Running environment

We can run our environment using 

```python
uv run pm_env run --config run_config.json --n-parallel 3
```

This will launch the environment. If everything was configured successfully, the models should run for a few minutes then output a successful answer.