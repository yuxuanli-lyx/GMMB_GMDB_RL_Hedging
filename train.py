# main file to train the RL agent in a general market setting
import os
import json
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import VecCheckNan, SubprocVecEnv
from stable_baselines.common.schedules import LinearSchedule
from utils.envs import TradingEnvUnderBSCFM


# import the config file
config = open("config.json")
config_data = json.load(config)

# log path
trial_num = config_data["RL_training"]["trial_num"]
tensorboard_log_path = "./pilot_tb/trial_{}".format(trial_num)
model_saving_path = "./pilot_model_trial_{}".format(trial_num)


# RL architecture
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            net_arch=config_data["RL_training"]["net_arch"],
            feature_extraction="mlp"
        )


# market parameters
financial_market_params = {
    "S0": config_data["market_params"]["S0"],
    "r": config_data["market_params"]["r"],
    "mu": config_data["market_params"]["mu"],
    "sigma": config_data["market_params"]["sigma"],
    "num_steps": config_data["market_params"]["num_steps"],
}
actuarial_market_params = {
    "N0": config_data["market_params"]["N0"],
    "Gm": config_data["market_params"]["Gm"],
    "Gd": config_data["market_params"]["Gd"],
    "rho": config_data["market_params"]["rho"],
    "fom": config_data["market_params"]["fom"],
    "T": config_data["market_params"]["T"],
}
reward_type = config_data["RL_training"]["reward_type"]

# RL training parameters
n_env = config_data["RL_training"]["n_env"]
gamma = config_data["RL_training"]["gamma"]
n_steps = config_data["RL_training"]["n_steps"]
ent_coef = config_data["RL_training"]["ent_coef"]
vf_coef = config_data["RL_training"]["vf_coef"]
start_lr = config_data["RL_training"]["start_lr"]
terminal_lr = config_data["RL_training"]["terminal_lr"]
max_grad_norm = config_data["RL_training"]["max_grad_norm"]
lam = config_data["RL_training"]["lam"]
nminibatches = config_data["RL_training"]["nminibatches"]
noptepochs = config_data["RL_training"]["noptepochs"]
cliprange = config_data["RL_training"]["cliprange"]
cliprange_vf = None
verbose = config_data["RL_training"]["verbose"]
total_timesteps = int(config_data["RL_training"]["total_timesteps"])


def main():

    # Create the vectorized environment
    env = TradingEnvUnderBSCFM(
        financial_market_params, actuarial_market_params, reward_type=reward_type
    )
    env_to_use = SubprocVecEnv([lambda: env for _ in range(n_env)])

    # schedule of the learning rate (linear by default)
    scheduled_lr = LinearSchedule(total_timesteps, start_lr, terminal_lr)

    # RL model setup
    model = PPO2(
        CustomPolicy,
        VecCheckNan(env_to_use, raise_exception=True),
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        learning_rate=scheduled_lr.value,
        max_grad_norm=max_grad_norm,
        lam=lam,
        nminibatches=nminibatches,
        noptepochs=noptepochs,
        cliprange=cliprange,
        cliprange_vf=cliprange_vf,
        verbose=verbose,
        tensorboard_log=tensorboard_log_path,
    )

    # Training
    model.learn(total_timesteps=total_timesteps)

    # Saving
    model.save(model_saving_path)


if __name__ == "__main__":
    main()
