# script for online learning training and rolling basis evaluation
import numpy as np
import json
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecCheckNan, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from utils.envs import TradingEnvUnderBSCFM
from utils.formulas import delta_bs_cfm
import pickle
from tqdm import tqdm

# import the config file
config = open("config.json")
config_data = json.load(config)

# online market environment
S0 = config_data["market_params"]["S0"]
mu = config_data["online_learning"]["mu"]
sigma = config_data["online_learning"]["sigma"]
num_steps = config_data["market_params"]["num_steps"]
r = config_data["market_params"]["r"]
T = config_data["market_params"]["T"]
N0 = config_data["online_learning"]["N0"]
Gm = config_data["market_params"]["Gm"]
Gd = config_data["market_params"]["Gd"]
rho = config_data["market_params"]["rho"]
fom = config_data["online_learning"]["fom"]

# some hyperparameters
online_trial = config_data["online_learning"]["online_trial"]
eval_episodes = config_data["online_learning"]["eval_episodes"]
online_steps = config_data["online_learning"]["online_steps"]
update_freq = config_data["online_learning"]["update_freq"]


def main():
    financial_market_params = {
        "S0": S0,
        "r": r,
        "mu": mu,
        "sigma": sigma,
        "num_steps": num_steps,
    }
    actuarial_market_params = {
        "N0": N0,
        "Gm": Gm,
        "Gd": Gd,
        "rho": rho,
        "fom": fom,
        "T": T,
    }

    # placeholder for the data
    df = np.zeros([online_trial, 4, eval_episodes, int(online_steps / update_freq) + 1])

    for _ in tqdm(range(online_trial)):
        # Create the vectorized environment for online learning
        env = TradingEnvUnderBSCFM(
            financial_market_params, actuarial_market_params, reward_type="terminal"
        )
        env.seed(_)
        env.online_mode()
        env_for_train = SubprocVecEnv([lambda: env for _ in range(1)])

        # load the model
        model_id = "./pilot_model_trial_1_cont_1.zip"
        model = PPO2.load(
            model_id,
            env=VecCheckNan(env_for_train, raise_exception=True),
            learning_rate=config_data["online_learning"]["lr"],
            n_steps=config_data["online_learning"]["n_steps"],
            nminibatches=config_data["online_learning"]["nminibatches"],
            noptepochs=config_data["online_learning"]["noptepochs"],
            tensorboard_log=None,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=update_freq,
            save_path="./online_temp_model/",
            name_prefix="online_model",
        )

        # train the RL agent in the online environment and save the intermediate model every 'n_steps' steps
        model.learn(online_steps, callback=checkpoint_callback)

        # generate the starting state vectors for rolling basis evaluation
        rl_wol_starting_pts = []
        rl_wool_starting_pts = []
        delta_mis_starting_pts = []
        delta_true_starting_pts = []
        env_rl_wol_starting_pts = TradingEnvUnderBSCFM(
            financial_market_params, actuarial_market_params, reward_type="terminal"
        )
        env_rl_wool_starting_pts = TradingEnvUnderBSCFM(
            financial_market_params, actuarial_market_params, reward_type="terminal"
        )
        env_delta_mis_starting_pts = TradingEnvUnderBSCFM(
            financial_market_params, actuarial_market_params, reward_type="terminal"
        )
        env_delta_true_starting_pts = TradingEnvUnderBSCFM(
            financial_market_params, actuarial_market_params, reward_type="terminal"
        )
        env_rl_wol_starting_pts.online_mode()
        env_rl_wool_starting_pts.online_mode()
        env_delta_mis_starting_pts.online_mode()
        env_delta_true_starting_pts.online_mode()
        env_rl_wol_starting_pts.seed(_)
        env_rl_wool_starting_pts.seed(_)
        env_delta_mis_starting_pts.seed(_)
        env_delta_true_starting_pts.seed(_)
        obs_rl_wol_starting_pts = env_rl_wol_starting_pts.reset()
        obs_rl_wool_starting_pts = env_rl_wool_starting_pts.reset()
        obs_delta_mis_starting_pts = env_delta_mis_starting_pts.reset()
        obs_delta_true_starting_pts = env_delta_true_starting_pts.reset()

        # load the rl model without online learning
        rl_wool_id = f"./online_temp_model/online_model_0_steps.zip"
        model_rl_wool = PPO2.load(rl_wool_id)

        # rollout the main trajectory
        for starting_pts_step in range(online_steps + 1):
            # if we need to update the ol model and record the starting points
            if starting_pts_step in range(0, online_steps + 1, update_freq):
                rl_wol_id = (
                    f"./online_temp_model/online_model_{starting_pts_step}_steps.zip"
                )
                model_rl_wol = PPO2.load(rl_wol_id)

                # for RL with online learning
                S0_rl_wol_starting_pts = env_rl_wol_starting_pts._St_traj[
                    env_rl_wol_starting_pts._step
                ]
                curr_step_rl_wol_starting_pts = env_rl_wol_starting_pts._step
                remain_t_rl_wol_starting_pts = (
                    num_steps - curr_step_rl_wol_starting_pts
                ) / num_steps
                curr_Pt_rl_wol_starting_pts = env_rl_wol_starting_pts._Pt
                data_dict_rl_wol = {
                    "current_S0": S0_rl_wol_starting_pts,
                    "T": remain_t_rl_wol_starting_pts,
                    "current_Pt": curr_Pt_rl_wol_starting_pts,
                }
                rl_wol_starting_pts.append(data_dict_rl_wol)

                # for RL without online learning
                S0_rl_wool_starting_pts = env_rl_wool_starting_pts._St_traj[
                    env_rl_wool_starting_pts._step
                ]
                curr_step_rl_wool_starting_pts = env_rl_wool_starting_pts._step
                remain_t_rl_wool_starting_pts = (
                    num_steps - curr_step_rl_wool_starting_pts
                ) / num_steps
                curr_Pt_rl_wool_starting_pts = env_rl_wool_starting_pts._Pt
                data_dict_rl_wool = {
                    "current_S0": S0_rl_wool_starting_pts,
                    "T": remain_t_rl_wool_starting_pts,
                    "current_Pt": curr_Pt_rl_wool_starting_pts,
                }
                rl_wool_starting_pts.append(data_dict_rl_wool)

                # for misspecified Delta
                S0_delta_mis_starting_pts = env_delta_mis_starting_pts._St_traj[
                    env_delta_mis_starting_pts._step
                ]
                curr_step_delta_mis_starting_pts = env_delta_mis_starting_pts._step
                remain_t_delta_mis_starting_pts = (
                    num_steps - curr_step_delta_mis_starting_pts
                ) / num_steps
                curr_Pt_delta_mis_starting_pts = env_delta_mis_starting_pts._Pt
                data_dict_delta_mis = {
                    "current_S0": S0_delta_mis_starting_pts,
                    "T": remain_t_delta_mis_starting_pts,
                    "current_Pt": curr_Pt_delta_mis_starting_pts,
                }
                delta_mis_starting_pts.append(data_dict_delta_mis)

                # for true Delta
                S0_delta_true_starting_pts = env_delta_true_starting_pts._St_traj[
                    env_delta_true_starting_pts._step
                ]
                curr_step_delta_true_starting_pts = env_delta_true_starting_pts._step
                remain_t_delta_true_starting_pts = (
                    num_steps - curr_step_delta_true_starting_pts
                ) / num_steps
                curr_Pt_delta_true_starting_pts = env_delta_true_starting_pts._Pt
                data_dict_delta_true = {
                    "current_S0": S0_delta_true_starting_pts,
                    "T": remain_t_delta_true_starting_pts,
                    "current_Pt": curr_Pt_delta_true_starting_pts,
                }
                delta_true_starting_pts.append(data_dict_delta_true)

            # rollout RL with online learning
            action_rl_wol_starting_pts, _states_info = model_rl_wol.predict(
                obs_rl_wol_starting_pts
            )
            (
                obs_rl_wol_starting_pts,
                reward_rl_wol_starting_pts,
                done_rl_wol_starting_pts,
                info,
            ) = env_rl_wol_starting_pts.step(action_rl_wol_starting_pts)
            if done_rl_wol_starting_pts:
                obs_rl_wol_starting_pts = env_rl_wol_starting_pts.reset()

            # rollout RL without online learning
            action_rl_wool_starting_pts, _states_info = model_rl_wool.predict(
                obs_rl_wool_starting_pts
            )
            (
                obs_rl_wool_starting_pts,
                reward_rl_wool_starting_pts,
                done_rl_wool_starting_pts,
                info,
            ) = env_rl_wool_starting_pts.step(action_rl_wool_starting_pts)
            if done_rl_wool_starting_pts:
                obs_rl_wool_starting_pts = env_rl_wool_starting_pts.reset()

            # rollout misspecified Delta
            m_delta_mis = env_delta_mis_starting_pts._m
            Ft_delta_mis = np.exp(obs_delta_mis_starting_pts[0])
            t_delta_mis = T - obs_delta_mis_starting_pts[-1]
            action_delta_mis_starting_pts = delta_bs_cfm(
                Ft=Ft_delta_mis,
                t=t_delta_mis,
                G_m=Gm,
                G_d=Gd,
                mu=config_data["market_params"]["fom"],
                m=m_delta_mis,
                me=m_delta_mis * 0.95,
                T=T,
                r=r,
                sigma=config_data["market_params"]["sigma"],
                rho=rho,
            )
            (
                obs_delta_mis_starting_pts,
                reward_delta_mis_starting_pts,
                done_delta_mis_starting_pts,
                info,
            ) = env_delta_mis_starting_pts.step(float(action_delta_mis_starting_pts))
            if done_delta_mis_starting_pts:
                obs_delta_mis_starting_pts = env_delta_mis_starting_pts.reset()

            # rollout correct Delta
            m_delta_true = env_delta_true_starting_pts._m
            Ft_delta_true = np.exp(obs_delta_true_starting_pts[0])
            t_delta_true = T - obs_delta_true_starting_pts[-1]
            action_delta_true_starting_pts = delta_bs_cfm(
                Ft=Ft_delta_true,
                t=t_delta_true,
                G_m=Gm,
                G_d=Gd,
                mu=fom,
                m=m_delta_true,
                me=m_delta_true * 0.95,
                T=T,
                r=r,
                sigma=sigma,
                rho=rho,
            )
            (
                obs_delta_true_starting_pts,
                reward_delta_true_starting_pts,
                done_delta_true_starting_pts,
                info,
            ) = env_delta_true_starting_pts.step(float(action_delta_true_starting_pts))
            if done_delta_true_starting_pts:
                obs_delta_true_starting_pts = env_delta_true_starting_pts.reset()

        # evaluation
        for eval_step in range(int(online_steps / update_freq) + 1):

            # load the model
            rl_wol_model_id = (
                f"./online_temp_model/online_model_{eval_step * update_freq}_steps.zip"
            )
            rl_wool_model_id = f"./online_temp_model/online_model_0_steps.zip"
            model_rl_wol_eval = PPO2.load(rl_wol_model_id)
            model_rl_wool_eval = PPO2.load(rl_wool_model_id)

            # load the environment
            # for RL with OL
            fin_param_rl_wol = {
                "S0": rl_wol_starting_pts[eval_step]["current_S0"],
                "r": r,
                "mu": mu,
                "sigma": sigma,
                "num_steps": num_steps,
            }
            act_param_rl_wol = {
                "N0": N0,
                "Gm": Gm,
                "Gd": Gd,
                "rho": rho,
                "fom": fom,
                "T": rl_wol_starting_pts[eval_step]["T"],
            }
            env_rl_wol_eval = TradingEnvUnderBSCFM(
                fin_param_rl_wol, act_param_rl_wol, reward_type="terminal"
            )

            # for RL without OL
            fin_param_rl_wool = {
                "S0": rl_wool_starting_pts[eval_step]["current_S0"],
                "r": r,
                "mu": mu,
                "sigma": sigma,
                "num_steps": num_steps,
            }
            act_param_rl_wool = {
                "N0": N0,
                "Gm": Gm,
                "Gd": Gd,
                "rho": rho,
                "fom": fom,
                "T": rl_wool_starting_pts[eval_step]["T"],
            }
            env_rl_wool_eval = TradingEnvUnderBSCFM(
                fin_param_rl_wool, act_param_rl_wool, reward_type="terminal"
            )

            # for misspecified Delta
            fin_param_delta_mis = {
                "S0": delta_mis_starting_pts[eval_step]["current_S0"],
                "r": r,
                "mu": mu,
                "sigma": sigma,
                "num_steps": num_steps,
            }
            act_param_delta_mis = {
                "N0": N0,
                "Gm": Gm,
                "Gd": Gd,
                "rho": rho,
                "fom": fom,
                "T": delta_mis_starting_pts[eval_step]["T"],
            }
            env_delta_mis_eval = TradingEnvUnderBSCFM(
                fin_param_delta_mis, act_param_delta_mis, reward_type="terminal"
            )

            # for true Delta
            fin_param_delta_true = {
                "S0": delta_true_starting_pts[eval_step]["current_S0"],
                "r": r,
                "mu": mu,
                "sigma": sigma,
                "num_steps": num_steps,
            }
            act_param_delta_true = {
                "N0": N0,
                "Gm": Gm,
                "Gd": Gd,
                "rho": rho,
                "fom": fom,
                "T": delta_true_starting_pts[eval_step]["T"],
            }
            env_delta_true_eval = TradingEnvUnderBSCFM(
                fin_param_delta_true, act_param_delta_true, reward_type="terminal"
            )
            for simulate_traj in range(eval_episodes):
                env_rl_wol_eval.seed(simulate_traj)
                env_rl_wool_eval.seed(simulate_traj)
                env_delta_mis_eval.seed(simulate_traj)
                env_delta_true_eval.seed(simulate_traj)
                env_rl_wol_eval.init_pt(rl_wol_starting_pts[eval_step]["current_Pt"])
                env_rl_wool_eval.init_pt(rl_wool_starting_pts[eval_step]["current_Pt"])
                env_delta_mis_eval.init_pt(
                    delta_mis_starting_pts[eval_step]["current_Pt"]
                )
                env_delta_true_eval.init_pt(
                    delta_true_starting_pts[eval_step]["current_Pt"]
                )

                # rollout evaluation trajectory
                obs_rl_wol_eval = env_rl_wol_eval.reset()
                obs_rl_wool_eval = env_rl_wool_eval.reset()
                env_delta_mis_eval.reset()
                env_delta_true_eval.reset()
                Ft_traj_delta_mis_eval = env_delta_mis_eval._Ft_traj
                Ft_traj_delta_true_eval = env_delta_true_eval._Ft_traj
                timeline_delta_mis_eval = (
                    np.arange(env_delta_mis_eval._total_steps + 1)
                    * env_delta_mis_eval._dt
                )
                timeline_delta_true_eval = (
                    np.arange(env_delta_true_eval._total_steps + 1)
                    * env_delta_true_eval._dt
                )
                m_delta_mis_eval = env_delta_mis_eval._m
                m_delta_true_eval = env_delta_true_eval._m
                delta_traj_delta_mis_eval = delta_bs_cfm(
                    Ft_traj_delta_mis_eval,
                    timeline_delta_mis_eval,
                    Gm,
                    Gd,
                    config_data["market_params"]["fom"],
                    m_delta_mis_eval,
                    m_delta_mis_eval * 0.95,
                    T,
                    r,
                    config_data["market_params"]["sigma"],
                    rho,
                )
                delta_traj_delta_true_eval = delta_bs_cfm(
                    Ft_traj_delta_true_eval,
                    timeline_delta_true_eval,
                    Gm,
                    Gd,
                    fom,
                    m_delta_true_eval,
                    m_delta_true_eval * 0.95,
                    T,
                    r,
                    sigma,
                    rho,
                )
                done_rl_wol_eval = False
                done_rl_wool_eval = False
                done_delta_mis_eval = False
                done_delta_true_eval = False
                t_idx_delta_mis = 0
                t_idx_delta_true = 0

                # RL with OL
                while not done_rl_wol_eval:
                    action_rl_wol_eval, _states_rl_wol = model_rl_wol_eval.predict(
                        obs_rl_wol_eval
                    )
                    (
                        obs_rl_wol_eval,
                        reward_rl_wol_eval,
                        done_rl_wol_eval,
                        info,
                    ) = env_rl_wol_eval.step(action_rl_wol_eval)
                df[_, 0, simulate_traj, eval_step] = reward_rl_wol_eval

                # RL without RL
                while not done_rl_wool_eval:
                    action_rl_wool_eval, _states_rl_wool = model_rl_wool_eval.predict(
                        obs_rl_wool_eval
                    )
                    (
                        obs_rl_wool_eval,
                        reward_rl_wool_eval,
                        done_rl_wool_eval,
                        info,
                    ) = env_rl_wool_eval.step(action_rl_wool_eval)
                df[_, 1, simulate_traj, eval_step] = reward_rl_wool_eval

                # True delta
                while not done_delta_true_eval:
                    action_delta_true_eval = delta_traj_delta_true_eval[
                        t_idx_delta_true
                    ]
                    (
                        obs_delta_true_eval,
                        reward_delta_true_eval,
                        done_delta_true_eval,
                        info,
                    ) = env_delta_true_eval.step(action_delta_true_eval)
                    t_idx_delta_true += 1
                df[_, 2, simulate_traj, eval_step] = reward_delta_true_eval

                # Misspecified delta
                while not done_delta_mis_eval:
                    action_delta_mis_eval = delta_traj_delta_mis_eval[t_idx_delta_mis]
                    (
                        obs_delta_mis_eval,
                        reward_delta_mis_eval,
                        done_delta_mis_eval,
                        info,
                    ) = env_delta_mis_eval.step(action_delta_mis_eval)
                    t_idx_delta_mis += 1
                df[_, 3, simulate_traj, eval_step] = reward_delta_mis_eval

    a_file = open("./data_ol.pkl", "wb")
    pickle.dump(df, a_file)
    a_file.close()


if __name__ == "__main__":
    main()
