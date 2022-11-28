import lib.env
import lib.model
from stable_baselines3 import PPO

def main():
    env = lib.env.create_stacked_env()
    model=PPO.load('ppo_mario.zip')
    lib.model.run(env, model)

if __name__ == '__main__':
    main()