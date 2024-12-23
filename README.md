# Exploring machine learning engineering datasets, tools, and approaches

The purpose of this repo is to create a machine learning pipeline to analyze multi-modal biomedical data, implementing MLOps best practices and deploying in a cloud environment.

Project components:
1. Data preparation
    - Use of a dataset from [The Cancer Genome Atlas Program (TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga). Specifically, I use data from the following paper of uterine carcinosarcoma https://www.cell.com/cancer-cell/fulltext/S1535-6108(17)30053-3
    - Preprocess and clean the data using Python libraries.
2. Model development
    - Implement a simple neural network model using PyTorch to predict patient remission [(notebook)](https://github.com/benslack19/wandb-explore/blob/main/project/tcga.ipynb)).
    - Start with clinical features before integrating genomic data.
3. MLOps Pipeline:
    - Set up a version control system using Git for code management (this repo).
    - Implement experiment tracking and model versioning using [Weights and Biases](https://wandb.ai/site)
    - Set up a [virtual environment](https://github.com/benslack19/wandb-explore/blob/main/environments/wandb.yml) which can also be used in combination with a [Dockerfile](https://github.com/benslack19/wandb-explore/blob/main/Dockerfile) for reproducibility.
4. Cloud Deployment:
    - Deploy the model on AWS SageMaker.
    - Implement an API endpoint for model inference.
    - **Note: I am currently troubleshooting this step.**


# Setup

I preferred to install packages with [mamba](https://mamba.readthedocs.io/en/latest/index.html). I installed with miniforge release `Release 24.9.0-0`.

1. I cloned [`data-science-base-repo`](https://github.com/benslack19/data-science-base-repo), renamed to `wandb-explore`, then updated the environment yaml file to be specific to this project.

```sh
git clone https://github.com/benslack19/data-science-base-repo.git wandb-explore
cd wandb-explore
```

2. Create mamba environment:

`mamba env create --file=environments/wandb.yml`

3. Install pre-commit hooks:

```sh
pre-commit install
```


## Usage

- Activate the base environment for data science work:

`mamba activate wandb`

- Code!

- Commit with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#specification)

- Run `pre-commit` on files:
```sh
# all files
pre-commit run --all-files
# or a specific file
pre-commit run --files ./project/utils.py
```

- Or run specific tools:
```sh
# ruff only
ruff check .
ruff format . 

# mypy only
mypy .
```



## Formatting notebooks

In the settings.json within jupyter lab code formatter:

```json
{
  "preferences": {
    "default_formatter": {
      "python": ["ruff", "black"]
    }
  },
  "formatOnSave": true
}
```

Note that ruff [does not autofix long line lengths](https://stackoverflow.com/questions/76771858/ruff-does-not-autofix-line-too-long-violation). Therefore, it helps to use ruff in combination with `black`.

## Formatting scripts using VSCode
- Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
- Install the [Black formatter extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter).
- Ensure there are no potential conflicts for extensions (e.g. isort is disabled).

```
# Other configurations
If using iterm2, it might be helpful to allow for this option for moving the cursor by going to `Settings -> Profiles -> Keys -> Key Mappings` as explained [here](https://stackoverflow.com/questions/81272/how-to-move-the-cursor-word-by-word-in-the-os-x-terminal) and [here](https://coderwall.com/p/h6yfda/use-and-to-jump-forwards-backwards-words-in-iterm-2-on-os-x).


## Examples to verify formatting

Paste these examples as a way to verify formatting in your notebook or script is working.

```python
# formatting should alphabetize this list of packages
import seaborn as sns
import pandas as pd
import numpy as np

# formatting should change this long list so that each element is on its own line
test_list = [ "apple", "banana", "orange", "apple", "banana", "orange", "apple", "banana", "orange"]
```

## Using docker image with [colima](https://github.com/abiosoft/colima)

[Colima](https://github.com/abiosoft/colima) allows use of running containers while obviating the use of Docker Desktop.

Some settings were necessary for testing of local deployment. 

```python
import os
import docker

# Set the Docker host to Colima's socket
os.environ["DOCKER_HOST"] = (
    f"unix:///Users/{os.getlogin()}/.colima/default/docker.sock"
)

client = docker.from_env(
    environment={
        "DOCKER_HOST": f"unix:///Users/{os.getlogin()}/.colima/default/docker.sock"
    }
)

predictor_local = pt_local_builder.deploy(
    docker_settings={
        "DOCKER_HOST": f"unix:///Users/{os.getlogin()}/.colima/default/docker.sock"
    }
)
```

# To do
- Add tests?
