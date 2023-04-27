"""
Main file for creating processed data and training an FQI agent

"""

import os
import yaml
import wandb
import multiprocessing as mp
import pickle
import datetime
# Data
from rltraining.data.create_data import create_data
# RL
from rltraining.rl_agents.offline_fqi import OfflineAgent as Agent
from rltraining.rl_agents.utils.data_transformations import DataTransformation
from rltraining.rl_agents.utils.cli_interface import create_cli


if __name__ == '__main__':
    mp.set_start_method(method="spawn")
    # Get arguments from CLI
    parser = create_cli()
    args = parser.parse_args()

    # Load config file
    configuration_dir = './configuration_files'

    house_id_mapping_path = f"{configuration_dir}/house_id_mapping.yaml"
    with open(house_id_mapping_path, 'r') as f_open:
        house_id_mapping = yaml.safe_load(f_open)

    config_file_path = f"{configuration_dir}/offline_rl_config.yaml"
    with open(config_file_path, 'r') as f_open:
        config_data = yaml.safe_load(f_open)

    # Modify config based on CLI arguments
    config_data['HOME_ID'] = args.home_id
    config_data['data_frequency'] = args.data_frequency
    config_data['logger']['name'] = args.wandb_name
    config_data['agent_params']['horizon'] = args.horizon
    config_data['agent_params']['ensemble_size'] = args.ensemble_size
    config_data['network_params']['depth'] = args.depth
    config_data['network_params']['batch_size'] = args.batch_size

    # Setup paths
    agent_dir = './data'
    wandb_dir = './data/wandb'
    agent_name = f"House_{config_data['HOME_ID']}_{datetime.date.today().day}_{datetime.date.today().month}_{datetime.date.today().year}"

    if os.path.isdir(wandb_dir) is False:
        os.makedirs(wandb_dir)

    # Initialise WandB logging
    wandb.login(key=config_data['wandb_key'])
    wandb_logger = wandb.init(project="fqi_policy_eval_boiler_agg", entity="physq",
                              group=f"nightly_{config_data['logger']['name']}",
                              tags="k8s",
                              name=f"{agent_name}_df_{config_data['data_frequency']}mins",
                              dir=wandb_dir,
                              config=config_data,
                              )

    # Initialize agent
    config_data['network_params']['boiler_network']['input_size'] = 1 * config_data['network_params']['depth']
    config_data['network_params']['house_network']['input_size'] = 1 * config_data['network_params']['depth']

    transformation_obj = DataTransformation(home_id=config_data['HOME_ID'],
                                            config_path=f"{configuration_dir}/data_transformation_constants.yaml")

    offline_agent = Agent(
        save_dir=f"{agent_dir}/{agent_name}",
        agent_params=config_data['agent_params'],
        network_params=config_data['network_params'],
        mode=config_data['mode'],
        data_frequency=config_data['data_frequency'],
        transformation_obj=transformation_obj,
        monitoring=True,
        wandb_logger=wandb_logger)

    # Create data
    training_df, val_df = create_data(home_id=config_data['HOME_ID'],
                                      house_id_mapping=house_id_mapping,
                                      data_frequency=config_data['data_frequency'],
                                      required_columns=config_data['data_columns'])

    data_dir = f'./{offline_agent.save_dir}/Data'

    if os.path.isdir(data_dir) is False:
        os.makedirs(data_dir)

    # Dump data into data_dir
    training_df.to_csv(path_or_buf=f'{data_dir}/training_data.csv')
    val_df.to_csv(path_or_buf=f'{data_dir}/val_data.csv')

    # Start Training
    print(f"Training HOME {config_data['HOME_ID']} with {config_data['data_frequency']} minutes data frequency")
    offline_agent.batch_train(train_batch_df=training_df, val_batch_df=val_df)

    model_path = f'./{offline_agent.save_dir}/models.pkl'
    with open(model_path, 'wb') as f_open:
        pickle.dump(obj=offline_agent, file=f_open)

    print(f"End")
