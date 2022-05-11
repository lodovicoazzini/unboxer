# Docker installation

## Configuring Ubuntu
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt update && apt-get update
apt-get install -y software-properties-common
```

## Installing git
Use the following command to install git:

``` 
apt install -y git
```


## Copy the project into the docker container

To copy FEATURE-MAP inside the docker container, open another console and run:

``` 
cd <FEATURE-MAP>
docker cp FEATURE-MAP/ <DOCKER_ID>:/
```

Where `<FEATURE-MAP>` is the location in which you downloaded the artifact and `<DOCKER_ID>` is the ID of the ubuntu docker image just started.

You can find the id of the docker image using the following command:

```
docker ps -a

CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS          PORTS     NAMES
13e590d65e60   ubuntu:bionic   "/bin/bash"   2 minutes ago   Up 2 minutes             recursing_bhabha
```

## Installing Python 3.6
Install Python 3.6:

``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.6
```

And check if it is correctly installed, by typing the following command:

``` 
python3 -V

Python 3.6.9
```

Check that the version of python matches `3.6.*`.

## Installing pip

Use the following commands to install pip and upgrade it to the latest version:

``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version

pip 21.1.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
```
## Creating a Python virtual environment

Install the `venv` module in the docker container:

``` 
apt install -y python3-venv
```

Create the python virtual environment:

```
cd /FEATURE-MAP
python3 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```

## Installing Python Binding to the Potrace library
Install Python Binding to the Potrace library.

``` 
apt install -y build-essential python-dev libagg-dev libpotrace-dev pkg-config
``` 

Install `pypotrace` (commit `76c76be2458eb2b56fcbd3bec79b1b4077e35d9e`):

``` 
cd /
git clone https://github.com/flupke/pypotrace.git
cd pypotrace
git checkout 76c76be2458eb2b56fcbd3bec79b1b4077e35d9e
pip install numpy
pip install .
``` 

To install PyCairo and PyGObject, we follow the instructions provided by [https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started](https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started).

``` 
apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
apt install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev
``` 

## Installing Other Dependencies

This tool has other dependencies, including `tensorflow` and `deap`, that can be installed via `pip`:

```
cd /DeepHyperion-MNIST
pip install -r requirements.txt
```