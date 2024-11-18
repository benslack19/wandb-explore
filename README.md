# A Base Data Science Repo for Environment Configurations

It's common to make different virtual environments depending on project needs. But often a common set of packages or linting and formatting programs will be used across these different environments. The purpose of this repository is to streamline the setup of data science environments. An example use case is to clone 

Note that I preferred to install packages with [mamba](https://mamba.readthedocs.io/en/latest/index.html). I installed with minforge release `Release 24.9.0-0`.

## Setup

1. Clone the repository. (If you already know what packages you'd like to add, you can edit the `environments/base.yml` file. It might be helpful to rename both the repo and the yaml file itself.)

```sh
git clone https://github.com/benslack19/data-science-base-repo.git
cd data-science-base-repo
```

2. Create conda environment:

`mamba env create -f environments/base.yml`

3. Install pre-commit hooks:

```sh
pre-commit install
```


## Usage

- Activate the base environment for data science work:

`mamba activate ds-base`

- Code!

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

This was done with the following steps.

If `ruff`, `black` and `jupyterlab-code-formatter` were not already included in the environment yml file
```sh
pip install ruff black jupyterlab-code-formatter
```

Then in the settings.json within jupyter lab code formatter:

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



## To do
- Create a generic data science template notebook and validate code linting
- Create a utils script and validate code linting
- Re-build environment with python 3.11
- Re-visit ruff settings
- Create project-specific environments
    - NLP environment
    - Bayesian statistics