# Experiments
This folder includes the codebase for Where2Act simulator and experiments.

## Before start
To train the models, please first go to the `../data` folder and download the pre-processed SAPIEN dataset for Where2Act. 
To test over the pretrained models, please go to the `logs` folder and download the pretrained checkpoints.

Please fill in [this form]() to download all the resources.

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

First, install SAPIEN

    pip install http://storage1.ucsd.edu/wheels/sapien-dev/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl


Then, if you want to run the 3D experiment, this depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.
    
    pip install -r requirements.txt

to install the other dependencies.

## Simulator
You can run the following command to test and visualize a random interation in the simulation environment.

    python collect_data.py 40147 StorageFurniture 0 pushing

Change the shape id to other ids for testing other shapes, 
and modify the primitive action type to any of the six supported types: *pushing, pushing-up, pushing-left, pulling, pulling-up, pulling-left*. 
Run `python collection_data.py --help` to understand the full input arguments. 

After you ran the code, you will find a record for this interaction trial under `./results/40147_StorageFurniture_0_pushing_0`, from where you can see the full log, 2D image, 3D depth and interaction outcome.
You can run the following command to replay the interaction.

    python replay_data.py results/40147_StorageFurniture_0_pushing_0/result.json

If you want to run on a headless server, simple put `xvfb-run -a ` before any code command that runs the SAPIEN simulator.
Install the `xvfb` tool on your server if not installed.

## Generate Offline Training Data
Before training the network, we need to collect a large set of interaction trials via random exploration.

    bash scripts/run_gen_offline_data.sh

This file generates data for a single cabinet example under the *pushing* primitive action. 
You can modify the content of the above file to generate data for different settings.
Also, please modify the `num_epochs` for generating different data amount and `num_processes` for the number of CPU cores to use.
Check the other parameters for more information.

    python gen_offline_data.py --help

## 3D Experiment
To train the network from scratch, run

    bash scripts/

To test the model, run

    bash scripts/



