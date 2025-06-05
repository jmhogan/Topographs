# Topographs

Public repository for a minimum working example of the topographs project.
* Associated paper: [arxiv](https://arxiv.org/abs/2303.13937)

## Installation
1. Setup the environment.
    * You can setup your own environment using the requirements.txt file. The project was tested with python3.8
    * Alternatively you can build a docker image with the provided Dockerfile
2. Download the data from zenodo [data](https://zenodo.org/record/7737248)
    * Make sure to adjust the paths in the configuration files to point to the data.

## Usage
To train a model, simply run ```python train.py configs/config_full.yaml log_dir```. This will train a model on complete events only saving all outputs in the directory ```log_dir``` which will be created.
Alternatively you can use ```configs/config_partial.yaml``` to train including partial events.

## Sequence of Usage
1. Prior to using Topograph, one must obtain and place the three .h5 files (train, test, validate) into the Topograph folder. Using the BBTo2b4tau.py script, produce the necessary root files that will be converted to .h5 files. For example, in TIMBER, use ```python3 BBTo2b4tau.py full_SIGNAL_2022.txt 0 64 2022```, where the first argument is the index of the list of root files in the full signal text file at which the analysis will start, the second argument is the final index at which the analysis will end, and the third argument is the year of the text file. Run the script, and obtain the root files.
   * With the root files now obtained, use ```python3 fromRootToh5.py``` (in TIMBER) to change the root files to h5 files. In the ```fromRootToh5.py``` script, one must edit and write the name of the root files in the "List of filenames." Run the script, and obtain the three h5 files.
2. Move the three h5 files into the Topograph folder. Set up the environment of Topograph with ```source /cvmfs/sft.cern.ch/lcg/views/LCG_103_LHCB_7/x86_64-centos7-clang12-opt/setup.sh```. This enironment uses the 2.8 version of tensor flow.
   * At this point, it will also be helpful to move the root files into Topograph, as they will be needed for step five.
3. To run the Topograph, use ```train.py``` to produce the outputs. Use ```python3 train.py configs/config_partial.yaml Outputs --test_file b_test.h5 --all``` to run the script and move the output files into the ```Outputs``` folder.
4. Next, run ```python3 plotModelPrints.py``` (in Topograph) to convert the .csv files to .png. The script produces four plots: Loss by Epoch, Initialisation by Epoch, Regression Losses by Epoch, and Classification Losses by Epoch.
5. Using the root files from step two, edit and write the names of the root files in ```visualizeTopograph.py``` and run ```python3 visualizeTopograph.py```. The script produces a plot of the momentum.


