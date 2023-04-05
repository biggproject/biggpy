rl-training
==============================

Training scripts for FQI-based agents. 

For training, run the file ``train.py``

Training configurations can be found in [configuration file](./configuration_files/offline_rl_config.yaml).

Command line arguments for different training scenarios are:

| Tag      | Description     | Type | Default |
|----------|-----------------|------|---------|
| ``-hid`` | House ID        | int  | 9       |
| ``-df``  | Data Frequency  | int  | 30      |
| ``-th``  | Horizon         | int  | 24      |
| ``-d``   | Depth           | int  | 8       |
| ``-es``  | Ensemble Size   | int  | 3       |
| ``-nb``  | Batch Size      | int  | 10000   |
| ``-wb``  | W&B Logger Name | str  | k8s     |



## Problem Description
The problem is defined as evaluating the business-as-usual control policy employed by boiler aggregator in terms of the amount of gas
consumed during a day.

### Data
The data obtained from boiler aggregator has a base frequency of 1 data point per minute. This data is resampled to lower frequencies
for training. The default data frequency is set to 1 datapoint every 5 minutes, however this can be changed to any 
_divisors_ of 60 (minutes).

The data obtained includes two types of measurements:
- Boiler Side

    Obtained from the boiler. Includes measurement such as boiler water temperatures, boiler setpoints, modulations etc.

- Boiler aggregator side (DomX)

    Obtained from the boiler aggregator sensor. Includes house level measurements such as room temperature, room setpoints, outside
    air temperature etc.

Note that neither of the sensors measure the "true" gas consumption, however the boiler modulation is considered as a 
good proxy for it.

### States
The states are defined to based on the assumption that a single houshold can be modelled as a partially observable 
Markov Decision process. To provide the agent with sufficient, _causal_ information, we use the following features:

    - time (t)
    - Outside Air Temperature (t_out)
    - Room Temperature Setpoint (t_r_set)
    - Past Room Temperatures  ([t_r,t-k, ..., t_r,t-1])
    - Past Boiler Modulations ([b_m,t-k, ..., b_r,t-1])
    - Current Room Temperature (t_r)

The value of k (referred to as _depth_) is dependent on the data frequency and is set such that the state comprises information from the past 4 
hours. (for frequency of 5 mins, k=48)

time is int between 0 and 1440, multiples of 5.

### Actions
The actions are assumed to be boiler setpoint values that are transformed into boolean ON/ OFF values.  The threshold 
for this is set to 20°C, implying any value above this leads to action ON (1) and setpoint value below this is considered 
as action OFF (0).

### Rewards
The instantaneous rewards are modelled as the boiler modulation at the current instant (b_m,t). Any Q-function trained 
using this reward leads to a value corresponding to the cumulative gas consumption of that house. 

### Algorithm
To train the agents, we follow a [backward FQI approach](https://arxiv.org/abs/2211.11830). The number of iterations are
chosen based on the data frequency to obtain a Q-function corresponding to the cumulative gas consumption for a day. For
eg., for a data frequency of 5mins, the horizon is set to **144**, leading to **288** total iterations (and function approximators).

To provide additional stability, we follow an ensemble-based approach called meanQ, which uses a set of N function approximators
per iteration (can be set using _-es_) and computes the mean value of their prediction for calculating the Q-value. 
This function has been implemented in the [Agent](./src/rl_agents/offline_fqi.py) as meanQ.

Additionally, because we use neural networks as functional approximators, the obtained data must be scaled before the 
regression step. To enable proper scaling for different houses, we use the [transformation_class](./src/rl_agents/utils/data_transformations.py)
This takes as input the house_id and produces a transformation object that can be used by the agent for all scaling and
re-scaling operations. 

### Outputs
This repository is used for training the RL agents. Once training is done, the models are saved in the agent directory 
as ``models.pkl``. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile
    ├── README.md
    │
    ├── configuration_files
    │   ├── offline_rl_config.yaml  <- YAML file containing values of required hyperparameters
    │   ├── house_id_mapping.yaml   <- YAML file mapping house_IDs to IDs used by boiler aggregator
    │   └── data_transformation_constants.yaml
    │
    ├── data
    │   ├── House_<ID>_<day>_<month>_<year>   <- All results are stored here
    │   └── wandb
    │
    ├── deploy
    │   └── <file_name>.yaml        <- YAML files containing deployment configs
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    |
    ├── notebooks                   <- Jupyter notebooks. Can be removed for deployment version
    │
    ├── references                  <- Data dictionaries, manuals, and all other explanatory materials.           
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── rltraining              <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                    <- Scripts to process data into requried format
    │   │   └── create_data.py
    │   │
    │   └── rl_agents               <- Scripts to train and save RL agents
    │   │   ├── utils               <- Support scripts, classes, etc.
    │   │   ├── modular_net.py      <- Script defining neural network architecture
    │   │   └── offline_fqi.py      <- Main class for Offline RL agent
    │
    ├── train.py                    <- Main training script
    │
    └── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
