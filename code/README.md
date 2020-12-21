# Experiments
This folder includes the codebase for Where2Act simulator and experiments.

## Before start
To train the models, please first go to the `../data` folder and download the pre-processed SAPIEN dataset for Where2Act. 
To test over the pretrained models, please go to the `logs` folder and download the pretrained checkpoints.

Please fill in [this form]() to download all the resources.

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

Please run
    
    pip3 install -r requirements.txt

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

## Quick Test
You can download the pre-trained model and use Jupyter Notebook to quickly visualize the result predictions.

## Generate Offline Training Data
Before training the network, we need to collect a large set of interaction trials via random exploration.


## 3D Experiment
To train the network from scratch, run

    bash scripts/

To test the model, run

    bash scripts/



