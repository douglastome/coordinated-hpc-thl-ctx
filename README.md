# Coordinated hippocampal-thalamic-cortical communication crucial for engram dynamics underneath systems consolidation

This repository contains the code used to perform the simulations reported in our manuscript:

```
Tomé, D.F., Sadeh, S. & Clopath, C. Coordinated hippocampal-thalamic-cortical
communication crucial for engram dynamics underneath systems consolidation. 
Nat Commun 13, 840 (2022). 
https://doi.org/10.1038/s41467-022-28339-z
```

This file provides specific instructions to reproduce our key simulation results (Fig. 2C and Fig. 3A). All the remaining simulations and data analyses in the manuscript can also be reproduced with this source code by modifying simulation and data analysis parameters (see below).

This code has been tested in machines running Ubuntu 18.04.5 LTS with at least 16 cores. If running on a machine with less than 16 cores, reset the number of MPI ranks in `globalvars.sh` accordingly.

## Installing prerequisites

Download the spiking network simulator Auryn version `6928b97` available at https://github.com/fzenke/auryn/tree/6928b97de024b47f696091c1d3de18ff535ffc91 and place it in your home directory. 

Unzip the Auryn source code zip and install Auryn:

```
cd ~
unzip auryn-6928b97de024b47f696091c1d3de18ff535ffc91
mv auryn-6928b97de024b47f696091c1d3de18ff535ffc91 auryn
sudo apt-get install cmake git build-essential libboost-all-dev
cd auryn/build/release
./bootstrap.sh && make
```

If you experience issues when installing Auryn, you may refer to its official documentation at https://fzenke.net/auryn/doku.php?id=quick_start.

Now create a Python 3 (version 3.8 or later) virtual environment `venv_sim` in your home directory and install the necessary packages:

```
cd ~
sudo apt-get install python3-venv
python3 -m venv venv_sim
source venv_sim/bin/activate
pip install numpy
pip install pandas
pip install cython
pip install sklearn
pip install scipy
pip install matplotlib
pip install seaborn
deactivate
```

Lastly, place the source code directory `src` in your home directory, create the simulation directory in the preset path, and move the source code directory `src` there:

```
cd ~
mkdir -p projects/systems-consolidation/simulations/sim_rc_p11/run-001
mv src/ projects/systems-consolidation/simulations/sim_rc_p11/run-001/
```


## Running a simulation

Run the simulation in Fig. 2B:

```
cd ~/projects/systems-consolidation/simulations/sim_rc_p11/run-001/src/
make clean
make
./run.sh 5
```

This will produce memory recall curves in the individual directories of trials 0, 1, 2, 3, and 4 in `~/projects/systems-consolidation/simulations/sim_rc_p11/run-001/`.

## Combining trial results

Take the steps below:

1) Open the file `run.sh` using a text editor and set the variable `HAS_SIMULATION` to `false` and the variable `HAS_MERGE` to `true`. Save `run.sh`.
2) Open the file `analyze_run.sh` using a text editor and set the variable `HAS_MERGE` to `true`. Save `analyze_run.sh`.
3) Merge trials:

```
cd ~/projects/systems-consolidation/simulations/sim_rc_p11/run-001/src/
./run.sh 5
```

This will produce average memory recall curves with 90% confidence intervals in `~/projects/systems-consolidation/simulations/sim_rc_p11/run-001/`.

## Further data analysis and simulations

Further analysis of the simulation results above (Fig. 3, Fig. S9, Fig. S10, Fig. S11, Fig. S14, Fig. S19, Fig. S20, Fig. S21, and Fig. S22) can be peformed by modifying parameters in `run.sh` and `analyze_run.sh` accordingly and running `./run.sh 5` from the source code location similarly to the procedure to combine trial results outlined above.

Simulations with blocked excitatory engram cells (Fig. 2D-E), blocked inhibitory neurons (Fig. 4), and blocked inhibitory engram cells (Fig. S25) during consolidation can be performed by adding the appropriate command-line arguments to `run-consolidation.sh` (see `sim_rc_p11.cpp`).

Simulations with hippocampus ablation in the testing phase (Fig. 5A) can be performed by adding the appropriate command-line arguments to `run-test_cue.sh` (see `sim_rc_p11.cpp`). Simulations with hippocampus ablation in the consolidation phase (Fig. S26) can be performed by adding the appropriate command-line arguments to both `run-consolidation.sh` and `run-test_cue.sh` (see `sim_rc_p11.cpp`).

Simulations with the same network configuration but different parameters (Fig. 5C-F, Fig. 5G-J, Fig. S12) can be performed by changing parameter values in `globalvars.sh`.

Simulations with different network configurations (i.e., Fig. 1, Fig. S6, Fig. S7, Fig. S8, Fig. S15, Fig. S16, Fig. S17) can be performed by changing the appropriate preprocessor directives in `sim_rc_p11.cpp` and changing simulation parameter values in `globalvars.sh`.

Statistical analysis of memory recall in different regions can be performed by running `datatester.py` with the data filters set for the desired recall data (see `datatester.py`).
