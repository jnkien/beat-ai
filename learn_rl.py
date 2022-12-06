""" Train a RL model """

import lib.env
import lib.model


def main():
    """Main function"""
    env = lib.env.create_stacked_env()
    lib.model.learn("PPO", env)


if __name__ == "__main__":
    main()
