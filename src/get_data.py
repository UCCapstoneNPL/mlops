# read params
# process
# return dataframe

import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path_im = config["data_source"]["gd_source_im"]
    data_path_ad = config["data_source"]["gd_source_ad"]
    im_df = pd.read_csv(data_path_im, sep=",", encoding="utf-8")
    ad_df = pd.read_csv(data_path_ad, sep=",", encoding="utf-8")
    return im_df, ad_df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    item_mvmt_df = get_data(config_path=parsed_args.config)
    adj_dtls_df = get_data(config_path=parsed_args.config)
