# 3D Maxwell FDFD solvers
Chenkai Mao, chenkaim@stanford.edu, 10/11/2025

##  Introduction:
This folder configures 3d Maxwell's equation FDFD simulation setups (with eps, source, pml), and calls either the neural network backend or the spins backend for simulation.

Currently it is a temporary setup, especially for the spins backend it relies on specific configurations on the FanLab server (it won't work on your own machine).

- NN github page (currently private): https://github.com/ChenkaiMao97/Waveynet3d
- spins-b github page: https://github.com/stanfordnqp/spins-b 

## Folder structure:
- src: this folder stores the API (calling NN and spins), as well as scripts for simulation setup
- notebooks: jupyter notebook scripts for interactive evaluation
- configs: configuration files for selecting which model to use, data logging etc.

## Instructions: 
1. Clone this repo to your own working directory
```
git clone https://github.com/ChenkaiMao97/3D_Maxwell_FDFD_solvers.git
```

2. Environment setup: (currently you can use Chenkai's prebuilt conda env, we will pack it into a python libary in the future). Please don't install or remove packages for now.
```
source /media/ps2/chenkaim/software/miniconda3/etc/profile.d/conda.sh # Chenkai's prebuilt environment
conda activate waveynet3d
```

3. (noninteractive mode) You can run your simulation by calling static python scripts. 

 - Simulate predefined random meta atom from the training dataset:
```
python random_sim.py configs/random_meta_atom.yaml
```

 - Simulate predefined large aperiodic device from the training dataset: (note that for large simulations if iterations are large, GPUs may run out of memory, if that happens use restarted GMRES can help)
```
python random_sim.py configs/random_large_aperiodic.yaml
```

 - Simulate custom defined meta atom (if parameter distribution is different from training, NN will be worse)
```
python custom_sim.py configs/meta_atom.yaml
```

 - Simulate custom defined waveguide bend (if parameter distribution is different from training, NN will be worse)
```
python custom_sim.py configs/waveguide_bend.yaml
```

4. (interacrive mode) You can also run commands interactively in a jupyter notebook. Note that in order to make jupyter notebook work on the server, you need to ssh with -X and port forwarding, (e.g. ssh -X -L 8888:localhost:8888 username@server_ip)

 - start the jupyter notebook as below, and open localhost:8888 in your local browser 
```
cd notebook
jupyter notebook --port 8888 --no-browsers
```

5. For spins, it works by first starting a local maxwells-b docker server, which accepts simulations. To check if the maxwells-b docker is running, run "docker container ls". If it is not running, you can start it by running this bash script: "/home/chenkaim/scripts/spins/maxwell-b/run_docker"


