# Simulation of Twin-field QKD and its reconciliation

this repo is part of my master thesis at Uni würzburg if i am finished i will also upload it.
For questions write felix.brandner@gmx.de
I know that this is not really neat code, i am sorry for anyone reading this.

*This repo does not include any of the data generated, as it is quite large.*

## Installation
```
git clone https://github.com/Felix1250/Twinfield-reconciliation.git
cd Twinfield-reconciliation
python -m venv twin
source twin/bin/activate
pip install requirements.txt
```
Tensorflow and strawberry fields might have some dependencies that do not get installed outomatically, they may need to be installed manually. 
strawberry fields relies on pkg_resources, which is deprecated, so this might become a future problem, but it works in ubuntu 20 and 24 at least.


## Usage

enter the virtual environment
```
source twin/bin/activate
```

Then start a simulation. adjust the simulation parameters or use one of my examples by changing the main. T
```
python3 src/server_graphs.py
```

This file shows best how the simulation can function. 
```
python3 src/general_graphs.py
```
+ create a twinfield object, set the parameters you want
+ use `tf_communicate` to simulate the hardware side and save it in a file
+ use `tf_communicat_load_settigns` if you want to load the parameters from the save file
+ otherwise use `tf_communicat_load` to load the saved file

because i was a bit stupid in how to structure everything, the imports only work from the Twinfield-reconciliation folder.

## Sources:

see masters thesis



### LDPC: 
the implementation for the LDPC code is taken from ```https://github.com/marcoavesani/QKD_LDPC_python/blob/master/README.md```

### AOPP:
the description for the AOPP protocol can be found here ```https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.042330```

### AI usage:
Ai was used to optimze code and to tell me how to solve certain mathematial problems i.e. chernoff bounds and to read documentation. i can not be bothered to figure out how to do matplotlib dual axis stuff on my own.