""" Train a ML model """

import os
import random
import string

import gym_super_mario_bros.actions

import lib.model_ml
import lib.utils


def train_ml():
    """Train a ML model on human data"""

    config = lib.utils.load_config("config_ml.yaml")
    data_path = config["data_path"]
    save_path = config["save_path"]

    actions, states = lib.utils.load_human_data(data_path)
    train_loader, test_loader = lib.utils.transform_data(actions, states, config)

    model = lib.model_ml.CNNModel(
        nclasses=len(gym_super_mario_bros.actions.SIMPLE_MOVEMENT),
        in_channels=states.shape[3],
    )

    model, metrics = lib.model_ml.learn(model, train_loader, test_loader, config)

    dir_hash = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    save_path = os.path.join(save_path, dir_hash)
    os.makedirs(save_path, exist_ok=False)

    lib.utils.dump_config(config, os.path.join(save_path, "config.yaml"))
    model.save(os.path.join(save_path, "model_cnn.pyt"))
    metrics.to_csv(os.path.join(save_path, "metrics.csv"), index=False)


if __name__ == "__main__":
    train_ml()
