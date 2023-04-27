"""
Basic command line arguments

"""

import argparse


def create_cli():
    """
    Creates the argparser object
    Returns: parser

    """
    parser = argparse.ArgumentParser(prog="RL Training")

    parser.add_argument('-wb', '--wandb_name', default='k8s', type=str)
    parser.add_argument('-th', '--horizon', default=48, type=int)
    parser.add_argument('-nb', '--batch_size', default=10000, type=int)
    parser.add_argument('-hid', '--home_id', default=9, type=int)
    parser.add_argument('-d', '--depth', default=48, type=int)
    parser.add_argument('-df', '--data_frequency', default=15, type=int)
    parser.add_argument('-es', '--ensemble_size', default=3, type=int)

    return parser
