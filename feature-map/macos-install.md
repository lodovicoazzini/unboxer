# MacOS installation

## Python version
The following steps refers to `python@3.8`. To ensure compatibility please use the same version.

To install `python@3.8` make sure you have `brew`, or follow the [official installation guide](https://docs.brew.sh/Installation). Then run the command
```
brew install python@3.8
```

## Install git
Run the command
```
git --version
```
If `git` is not already present on your machine, it will prompt you to install it.

## Clone the project
Open the terminal in the directory where you want to clone the repository and run the command
```
git clone https://github.com/zohdit/feature-map
cd feature-map
```

## Create a python virtual environment

Move in the cloned project
```
cd <path/to/feature-map>
```

Make sure the python version in use is `python@3.8`
```
python -V
    Python 3.8.13
```

Create and load the virtual environment
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## Install Potrace
Install system dependencies
```
brew install libagg pkg-config potrace
```

Install pypotrace
```
git clone https://github.com/flupke/pypotrace.git
cd pypotrace
git checkout 76c76be2458eb2b56fcbd3bec79b1b4077e35d9e
pip install numpy
pip install .
cd ..
rm -rf pypotrace
```

## Install Cairo
```
brew install pygobject3 gtk+3
brew install cairo
```

## Install the other dependencies
```
cd MNIST
pip install -r requirements.txt
```