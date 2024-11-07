# A Base Data Science Repo for Environment Configurations

It's common to make different virtual environments depending on project needs. But often a common set of packages or linting programs will be used across these different environments. The purpose of this repository is to streamline the setup of data science environments. An example use case is to clone 

 contains conda environment configurations and linting setup for data science projects. 

## Setup

1. Clone the repository. (If you already know what packages you'd like to add, you can edit the `environments/base.yml` file. It might be helpful to rename both the repo and the yaml file itself.)

```bash
git clone https://github.com/benslack19/data-science-base-repo.git
cd data-science-base-repo
```

2. Create conda environment:

`conda env create -f environments/base.yml`

3. Install pre-commit hooks:

```bash
pre-commit install
pre-commit install
```


## Usage

- Activate the base environment for data science work:

`conda activate ds-base`

- Code!

- Run Ruff manually: 

```bash
ruff check .
ruff format . 
```

- Run `pre-commit` on all files:

```bash
pre-commit run --all-files
```
