# Training of the deep hedging agent
import logging
import json
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributions as dist
from network import FeedForwardNet
import sys

sys.path.append("../")
from utils.formulas import fair_rate_bs_cfm

# import the config file
config = open("../config.json")
config_data = json.load(config)

# logger setting
logging.basicConfig(level=logging.NOTSET)

# useful functions
def torch_geometric_brownian_motion(
    S: float,
    mu: float,
    sigma: float,
    tau: float,
    n_steps: int,
    batch_size: int,
) -> torch.Tensor:
    """
    torch implementation of geometric brownian motion
    :param S: initial stock price
    :param mu: drift term
    :param sigma: volatility
    :param tau: term to maturity
    :param n_steps: number of steps to generate
    :param batch_size: batch size to generate
    :return: geometric brownian motion trajectories of shape(batch_size, n_steps + 1)
    """
    dt = tau / n_steps
    timeline = torch.arange(0, tau + dt, dt)
    dw = torch.normal(0, np.sqrt(dt), size=[batch_size, n_steps])
    wt = torch.cumsum(dw, dim=1)
    wt = torch.cat((torch.zeros([batch_size, 1]), wt), dim=1)
    return S * torch.exp((mu - 0.5 * (sigma**2)) * timeline + sigma * wt)


# some hyperparameters
trial_num = config_data["DH_training"]["trial_num"]
log_dir = "./dh_tb/trial_{}".format(trial_num)
model_save_path = "./model_trial_{}.pt".format(trial_num)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
train_logger = tb.SummaryWriter(log_dir, flush_secs=1)

# params for the FFN
net_in = config_data["DH_training"]["net_in"]
net_out = config_data["DH_training"]["net_out"]
net_structure = config_data["DH_training"]["net_structure"]

# model setup
model = FeedForwardNet(in_dim=net_in, out_dim=net_out, net_structure=net_structure).to(
    device, dtype=dtype
)

# params for the training algo
learning_rate = config_data["DH_training"]["lr"]
weight_decay = config_data["DH_training"]["weight_decay"]
patience = config_data["DH_training"]["patience"]
n_epoch = config_data["DH_training"]["n_epoch"]
batch_size = config_data["DH_training"]["batch_size"]
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

# params for market
S0 = config_data["market_params"]["S0"]
mu = config_data["market_params"]["mu"]
sigma = config_data["market_params"]["sigma"]
num_steps = config_data["market_params"]["num_steps"]
r = config_data["market_params"]["r"]
T = config_data["market_params"]["T"]
N0 = config_data["market_params"]["N0"]
Gm = config_data["market_params"]["Gm"]
Gd = config_data["market_params"]["Gd"]
rho = config_data["market_params"]["rho"]
fom = config_data["market_params"]["fom"]
m = fair_rate_bs_cfm(S0 * rho, Gm, Gd, fom, T, r, sigma)  # fair annual deduction rate
me = m * 0.95  # by default


def main():
    for _ in range(n_epoch):
        model.train()
        # generate a batch of data
        dt = T / num_steps
        timeline = torch.arange(0, T + dt, dt)
        tau_timeline = T - timeline
        batch_S_traj = torch_geometric_brownian_motion(
            S0, mu, sigma, T, num_steps, batch_size
        )
        batch_F_traj = rho * batch_S_traj * torch.exp(-m * timeline)
        batch_lifetime = dist.exponential.Exponential(fom).sample([batch_size, N0])
        batch_N_traj = torch.sum(batch_lifetime[:, :, None] > timeline, dim=1)
        batch_survival_rate_traj = torch.div(batch_N_traj, N0).float()

        # interaction with the environment
        Pt = torch.zeros([batch_size, 1])  # initial portfolio value
        for t in range(num_steps):
            batch_obs = torch.cat(
                (
                    torch.log(batch_F_traj[:, t].view([batch_size, 1])),  # log(Ft)
                    torch.div(Pt, N0),  # Pt / N0
                    batch_survival_rate_traj[:, t].view([batch_size, 1]),  # Nt / N0
                    tau_timeline[t] * torch.ones([batch_size, 1]),
                ),  # tau
                dim=1,
            )
            batch_action = model(batch_obs)
            Pt = (
                (
                    Pt
                    - batch_action
                    * batch_N_traj[:, t].view([batch_size, 1])
                    * batch_S_traj[:, t].view([batch_size, 1])
                    + me
                    * batch_F_traj[:, t].view([batch_size, 1])
                    * dt
                    * batch_N_traj[:, t].view([batch_size, 1])
                )
                * torch.exp(torch.tensor(r * dt))
                + batch_action
                * batch_N_traj[:, t].view([batch_size, 1])
                * batch_S_traj[:, t + 1].view([batch_size, 1])
                - (
                    batch_N_traj[:, t].view([batch_size, 1])
                    - batch_N_traj[:, t + 1].view([batch_size, 1])
                )
                * np.maximum(Gd - batch_F_traj[:, t + 1].view([batch_size, 1]), 0)
            )

        # batch loss
        batch_terminal_p = Pt
        batch_terminal_l = batch_N_traj[:, -1].view([batch_size, 1]) * np.maximum(
            Gm - batch_F_traj[:, -1].view([batch_size, 1]), 0
        )
        batch_terminal_p_l = batch_terminal_p - batch_terminal_l
        batch_squared_p_l = torch.div(batch_terminal_p_l**2, N0 + 0.0)
        batch_loss = torch.mean(batch_squared_p_l)

        # logger info
        train_logger.add_scalar("train/loss", batch_loss, global_step=_)

        # terminal info
        training_progress = (_ / n_epoch) * 100
        logging.info(
            "Training progress: {}%; training loss: {}".format(
                training_progress, batch_loss
            )
        )

        # update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    # save the model
    torch.save(model, model_save_path)


if __name__ == "__main__":
    main()
