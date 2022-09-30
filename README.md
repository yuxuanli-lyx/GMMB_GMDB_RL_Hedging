# GMMB_GMDB_RL_Hedging
This is the accompanying codebase for the paper *Pseudo-Model-Free Hedging for Variable Annuities via Deep Reinforcement Learning*.

## Getting Started
### Conda Environment Setup
The conda environment can be created by
```sh
conda env create -f environment.yml
```
and be activated by
```sh
conda activate GMMB_GMDB_RL_Hedging
```
### Installation
Users can download the repo by running
```sh
git clone https://github.com/yuxuanli-lyx/GMMB_GMDB_RL_Hedging.git
```

## Basic Usage
### Training Phase
#### Train a RL agent in the training environment
Users can simply run the following code to train a RL agent in the training environment
```sh
python3 train.py
```
A training log will be automatically created under the directory below, with *trial_1* being the default trial number.
```sh
./pilot_tb/trial_1
```
To monitor the training process, users can utilize the tensorboard feature.
```sh
tensorboard --logdir ./pilot_tb/trial_1
```
Once the training is finished, the model of RL agent will be saved as
```sh
pilot_model_trial_1.zip
```
#### Train a DH agent
To train a DH agent, users can navigate to the deep hedging folder by
```sh
cd deep_hedging_method
```
and then run the following:
```sh
python3 deep_hedging_training.py
```
Similar to the case of RL agent training, a training log will also be automatically created under the directory below, with *trial_1* being the default trial number.
```sh
./dh_tb/trial_1
```
Users can use the following command to monitor the training progress of the DH agent
```sh
tensorboard --logdir ./dh_tb/trial_1
```
and the trained model will be saved as
```sh
model_trial_1.pt
```
#### Baseline Evaluation
##### Data Collection
To simulate the scenarios from the training environment and collect the realized terminal P&Ls of the trained RL agent and classical Deltas, users can follow the **Baseline Evaluation Section** in the notebook 
```sh
test.ipynb
```
Note that the realized terminal P&Ls by each hedging strategy will be saved in a *.pkl* file. In particular, the realized terminal P&Ls by Delta from Black-Scholes (BS) financial and constant force of mortality (CFM) actuarial parts are saved in
```sh
data_delta_bs_cfm.pkl
```
the realized terminal P&Ls by Delta from BS financial and increasing force of mortality (IFM) actuarial parts are saved in
```sh
data_delta_bs_ifm.pkl
```
the realized terminal P&Ls by Delta from Heston financial and CFM actuarial parts are saved in
```sh
data_delta_heston_cfm.pkl
```
the realized terminal P&Ls by Delta from Heston financial and IFM actuarial parts are saved in
```sh
data_delta_heston_ifm.pkl
```
and the realized terminal P&Ls by RL agent are saved in
```sh
data_rl.pkl
```
To do the same thing for the trained DH agent, users should navigate to the deep hedging folder by
```sh
cd deep_hedging_method
```
and use the code in the notebook
```sh
DH_evaluation.ipynb
```
Note that the realized terminal P&L by DH agent will also be saved in a *.pkl* file
```sh
data_dh.pkl
```
##### Data Analysis
Once the data collection is finished, all the figures and tables in Section 4.4 of the paper can be reproduced via **Training Phase Figures and Tables** in the notebook
```sh
data_analysis_file.ipynb
```
### Online Learning Phase
#### Online Learning and Rolling-basis Data Collection
To start the (trained) RL agent's online learning and data collection for the rolling-basis evaluation, users can simply run the following code
```sh
python3 online_eval.py
```
Note that the realized terminal P&Ls during the online learning phase will be saved in
```sh
data_ol.pkl
```
which also includes the realized terminal P&Ls by RL agent without online learning, correct Delta, and incorrect Delta for the purpose of comparison.
#### Data Analysis
Once the data collection is done, all the figures and tables in Section 6 of the paper can be reproduced via **Online Learning Phase Figures and Tables** in the notebook
```sh
data_analysis_file.ipynb
```
## Structure Overview
### Directories
#### Deep_Hedging_Method
*DH_evaluation.ipynb*: The notebook file used to simulate senarios from the training environment, implement the trained DH agent, and collect realized terminal P&Ls by the DH agent for data analysis.

*deep_hedging_training.py*: The pytorch implementation of the deep hedging algorithm.

*network.py*: The module of feedforward network architecture; can be extended to include different network architectures, for example RNN, CNN, etc.
#### utils
*agent.py*: Modules of different Delta strategies based on different models.

*envs.py*: The module of the MDP training environment, which has BS financial and CFM actuarial parts, for the RL agent.

*formulas.py*: Functions used for integral evaluation, net liability computation, fair rate determination, and delta calculation under different models.

*plots_generator.py*: Functions used for generating figures and tables in the paper.

*processes.py*: Functions used for simulating the risky asset price process. Only brownian motion and geometric brownian motion simulators are included in the current version.
#### plot_outputs
All the figures in the paper are included in this file.
### Files
*environment.yml*: The YAML file for the conda environment configuration.

*config.json*: The JSON file that specifies the values of parameters and hyperparameters used in this codebase.

*data_analysis_file.ipynb*: The notebook used for generating the figures and tables in the paper; functions from *plots_generator.py* are called in this notebook.

*test.ipynb*: The notebook used for testing the functions in *formulas.py* as well as collecting data for the baseline evaluation.

*train.py*: The script for training the RL agent in the MDP training environment.

*online_eval.py*: The script for the RL agent's online learning and roll-basis evaluation.
