## 📲 Install 📲

### Clone the repository

Clone the repository of this project by running the following command.

```commandline
git clone https://github.com/lodovicoazzini/unboxer.git
```

### Create the conda environment

Create and activate the conda environment from the provided `env.yml` file executing the following commands.

```commandline
conda env create -f env.yml --name unboxer
conda activate unboxer
```

### Install pypotrace

Move in the `feature_map/pypotrace/` directory and install the required tool executing the following commands.

```commandline
brew install libagg pkg-config potrace
cd feature_map/pypotrace/
pip install .
cd ../../
```

### Install PyCairo

Install the dependency executing the following commands.

```commandline
brew install pygobject3 gtk+3
brew install cairo
```

### Install the dependencies

Install the dependencies executing the following command.

```commandline
pip install -r requirements.txt 
```