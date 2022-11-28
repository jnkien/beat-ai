import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainCallback(BaseCallback):
    def __init__(self, freq_to_save, save_path, verbose=1):
        super().__init__(verbose)
        self.freq_to_save = freq_to_save
        self.save_path = save_path
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.freq_to_save == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
        return True

def run(env, model):
    state = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()

def learn(model_name, env):
    model = model_factory(model_name, env)
    model.learn(total_timesteps=4_000_000, callback=TrainCallback(20_000, './data/models/'))

def model_factory(model_name, env):
    factory = {
        'PPO': PPO("CnnPolicy", env, verbose=1, tensorboard_log='./logs/', learning_rate=0.000001, n_steps=512)
    }
    return factory[model_name]