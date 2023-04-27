"""
Function for creating and saving processed data

Change date_duration, val_start_date

"""
import requests  # Dependency for wandb/ others
import json
import pandas as pd
from datetime import datetime, timedelta


def create_data(home_id: int = 2, house_id_mapping: dict = None,
                data_frequency: int = 5, required_columns: list = None):
    if house_id_mapping is None:
        print("No mapping found---> Using data from local system")
        raw_df = pd.read_csv(filepath_or_buffer=f'./data/raw/boiler_aggregator/HOME{home_id}.csv', header=0)
        raw_df['time'] = pd.to_datetime(raw_df['time'])
    else:
        house_id_key = house_id_mapping[home_id]
        raw_df = fetch_data(house_id_key=house_id_key)

    training_df, val_df = get_data(dataframe=raw_df, resampling_frequency=data_frequency)

    training_df = select_columns(training_df, cols_to_use=required_columns)
    val_df = select_columns(val_df, cols_to_use=required_columns)

    return training_df, val_df


# -------------------------------------------------------------------------------------------------------------------- #
# Support functions


def get_data(dataframe=None, resampling_frequency=5):
    df = dataframe
    df['time'] = pd.to_datetime(df['time'])

    # Cleaning
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')      # Can be removed in newer versions

    mask = df['blr_mod_lvl'] < 0
    df.loc[mask, "blr_mod_lvl"] = 0.0

    # Resample to 5 minutes using mean function
    resample_frequency = resampling_frequency  # 5 minutes
    df = df.resample(f'{resample_frequency}T', on='time').mean().reset_index()
    df = df.fillna(method='ffill')
    # Create day as a feature
    df['day'] = df['time'].dt.dayofweek

    # Select 5 days for creating validation set.
    val_start_day = pd.to_datetime(df['time'][0].date() + timedelta(12))
    val_end_day = pd.to_datetime(val_start_day + timedelta(5))

    val_df_mask = ((df['time'] > val_start_day) & (df['time'] < val_end_day))

    training_df = df.loc[~val_df_mask].reset_index()
    training_df['time'] = training_df['time'].dt.hour * 60 + training_df['time'].dt.minute
    training_df.drop(columns='index', inplace=True)

    val_df = df.loc[val_df_mask].reset_index()
    val_df['time'] = val_df['time'].dt.hour * 60 + val_df['time'].dt.minute
    val_df.drop(columns='index', inplace=True)

    return training_df, val_df


def select_columns(dataframe=None, cols_to_use=None):
    df = dataframe
    if cols_to_use is None:
        pass
    else:
        df = df.loc[:, cols_to_use]

    return df


def fetch_data(house_id_key: str = None):
    data_duration = 30  # days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=data_duration)

    # URL for Obelisk, change with gg_ingest token
    url = "http://10.2.33.34:5000/datagrouped"
    payload = json.dumps({
        "metrics_boiler_aggregator": [
            "t_r",
            "t_set",
            "t_r_set",
            "t_out",
        ],
        "metrics_boiler": [
            "blr_mod_lvl",
            "heat"
        ],
        "start": start_date.timestamp() * 1000,
        "end": end_date.timestamp() * 1000,
        "house_id": f"{house_id_key}",
        "user_id_obelisk": "NjNlMjU5NjZjNDdhZmU3MTFkODkwZTYwOmQ0N2NmZDJiLWNmYWMtNDU4OC05ZTczLWQ5ZmZmNmU5MWI5ZQ=="
    })
    headers = {
        'Content-Type': 'application/json'
    }
    print(f"Fetching Data for {house_id_key}")
    response = requests.request("GET", url, headers=headers, data=payload)
    response_json = response.json()
    response_dict = dict(response_json)

    house_data_dict = response_dict

    df = pd.DataFrame.from_dict(house_data_dict, orient='index')
    df['time'] = pd.to_datetime(df.index)
    df.reset_index(inplace=True, drop=True)

    return df
