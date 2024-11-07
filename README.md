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

- Run Ruff manually: 

```sh
ruff check .
ruff format . 
```

- Run `pre-commit` on all files:

```sh
pre-commit run --all-files
```

## Formatting notebooks on save

This was done with the following:

```sh
pip install ruff jupyterlab-code-formatter
```

# Other configurations

If using iterm2, it might be helpful to allow for this option for moving the cursor by going to `Settings -> Profiles -> Keys -> Key Mappings` as explained [here](https://stackoverflow.com/questions/81272/how-to-move-the-cursor-word-by-word-in-the-os-x-terminal) and [here](https://coderwall.com/p/h6yfda/use-and-to-jump-forwards-backwards-words-in-iterm-2-on-os-x).

## To do
- Create a generic data science template notebook and validate code linting
- Create a utils script and validate code linting
- Re-build environment with python 3.11
- Re-visit ruff settings
- Create project-specific environments
    - NLP environment
    - Bayesian statistics