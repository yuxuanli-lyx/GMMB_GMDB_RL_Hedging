from typing import List, Union, Tuple
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kde
import seaborn as sns
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import scipy.stats as stats

# logger setting
logging.basicConfig(level=logging.ERROR)


def training_log_plot(
    file: str, font_size: int, figsize: tuple, plot_type: str
) -> None:
    """
    The function that creates the plots for bootstrapped reward and entropy loss during training
    :param file: csv file name
    :param font_size: font size in the plot
    :param figsize: figure size of the plot
    :param plot_type: reward plot or entropy loss plot
    """
    if plot_type not in ["reward", "entropy"]:
        raise ValueError("the plot type has to be either 'reward' or 'entropy'")
    df = pd.read_csv(file)
    plt.rcParams.update({"font.size": font_size})
    plt.subplots(figsize=figsize)
    plt.plot(df["Step"], df["Value"])
    plt.xlabel("Training Step")
    if plot_type == "reward":
        plt.ylabel("Bootstrapped Sum of Rewards")
    else:
        plt.ylabel("Batch Entropy")
    plt.show()


def empirical_density_plot(
    file_names: List[str],
    font_size: int,
    figsize: tuple,
    bin: int,
    alpha: float,
    x_lim: List[float],
    x_num_grid: int,
    colors: List[str],
    markers: List[Union[str, int]],
    labels: List[str],
    markersize: int = 5,
    markevery: int = 8,
    density: bool = True,
) -> None:
    """
    Function that draws the empirical densities
    :param file_names: a list of file names for the raw data
    :param font_size: font size in the plot
    :param figsize: figure size of the plot
    :param bin: number of bins
    :param alpha: level of alpha (opacity)
    :param x_lim: the range of x-axis
    :param x_num_grid: number of grids to be used on the x-axis
    :param colors: a list of colors for the densities
    :param markers: a list of marker type for the densities
    :param labels: a list of labels for the densities
    :param markersize: marker size
    :param markevery: marker interval
    :param density: whether to normalize as a density function or not
    """
    # empirical densities
    bins = bin
    alpha = alpha
    x = np.linspace(x_lim[0], x_lim[1], x_num_grid)
    plt.rcParams.update({"font.size": font_size})
    plt.subplots(figsize=figsize)
    for _ in range(len(file_names)):
        data = pd.read_pickle(file_names[_])
        plt.hist(data, density=density, bins=bins, alpha=alpha, color=colors[_])
        density = kde.gaussian_kde(data)
        y_val = density(x)
        plt.plot(
            x,
            y_val,
            marker=markers[_],
            markersize=markersize,
            color=colors[_],
            markevery=markevery,
            label=labels[_],
        )
    plt.xlim((x_lim[0], x_lim[1]))
    plt.xlabel("Terminal Profit and Loss")
    plt.ylabel("Empirical Density")
    plt.legend()
    plt.show()


def empirical_cdf_plot(
    file_names: List[str],
    font_size: int,
    figsize: tuple,
    x_lim: List[float],
    colors: List[str],
    markers: List[Union[str, int]],
    labels: List[str],
    markersize: int = 8,
    markevery: int = 500,
    endpoint: bool = False,
) -> None:
    """
    Function that draws the empirical cdfs
    :param file_names: a list of file names for the raw data
    :param font_size: font size in the plot
    :param figsize: figure size of the plot
    :param x_lim: the range of x-axis
    :param colors: a list of colors for the densities
    :param markers: a list of marker type for the densities
    :param labels: a list of labels for the densities
    :param markersize: marker size
    :param markevery: marker interval
    :param endpoint: whether to include the endpoint or not
    """
    plt.rcParams.update({"font.size": font_size})
    plt.subplots(figsize=figsize)
    for _ in range(len(file_names)):
        data = pd.read_pickle(file_names[_])
        plt.plot(
            np.sort(data),
            np.linspace(0, 1, len(data), endpoint=endpoint),
            marker=markers[_],
            markersize=markersize,
            color=colors[_],
            markevery=markevery,
            label=labels[_],
        )
    plt.xlim((x_lim[0], x_lim[1]))
    plt.xlabel("Terminal Profit and Loss")
    plt.ylabel("Empirical Cumulative Distribution")
    plt.legend()
    plt.show()


def pathwise_diff(file_1: str, file_2: str, x_lim: List[float], font_size: int) -> None:
    """
    The function that plots the empirical density of pathwise difference of two strategies
    Note the difference is calculated as file_1 - file_2
    :param file_1: the file name of the first strategy
    :param file_2: the file name of the second strategy
    :param x_lim: the range of x-axis
    :param font_size: font size in the plot
    """
    plt.rcParams.update({"font.size": font_size})
    data_1 = pd.read_pickle(file_1)
    data_2 = pd.read_pickle(file_2)
    sns.distplot(data_1 - data_2)
    plt.xlim((x_lim[0], x_lim[1]))
    plt.xlabel("Terminal Profit and Loss Pathwise Difference")
    plt.ylabel("Empirical Density")


def summary_stats(file_names: List[str], labels: List[str]) -> pd.DataFrame:
    """
    The function that produces a table of summary statistics for each strategy
    :param file_names: a list of file names for the raw data
    :param labels: a list of labels for different rows
    :return: a pandas data frame of summary statistics
    """
    # placeholders
    mean = []
    std = []
    median = []
    Var_90 = []
    Var_95 = []
    Tvar_90 = []
    Tvar_95 = []
    RMSE = []

    for _ in range(len(file_names)):
        data = pd.read_pickle(file_names[_])
        mean.append(np.mean(data))
        std.append(np.std(data))
        median.append(np.median(data))
        Var_90.append(np.percentile(data, 10))
        Var_95.append(np.percentile(data, 5))
        var_90 = np.percentile(data, 10)
        Tvar_90.append(np.mean(data[data < var_90]))
        var_95 = np.percentile(data, 5)
        Tvar_95.append(np.mean(data[data < var_95]))
        RMSE.append(np.sqrt(np.mean(data**2)))
    d = {
        "Terminal P&L of Hedging Approach": labels,
        "Mean": mean,
        "Median": median,
        "Std. Dev.": std,
        "Var_90": Var_90,
        "Var_95": Var_95,
        "TVaR_90": Tvar_90,
        "TVaR_95": Tvar_95,
        "RMSE": RMSE,
    }
    df = pd.DataFrame(data=d)
    return df


def pathwise_summary_stats(
    main_file: str, other_file: List[str], labels: List[str]
) -> pd.DataFrame:
    """
    The function that produces a table of summary statistics for the pathwise difference
    :param main_file: the main data file to be compared with other data
    :param other_file: the other data file
    :param labels: a list of labels for different rows
    :return: a pandas data frame of summary statistics
    """
    main_data = pd.read_pickle(main_file)

    # placeholder
    mean = []
    std = []
    median = []
    prob = []

    for _ in range(len(other_file)):
        other_data = pd.read_pickle(other_file[_])
        diff = main_data - other_data
        mean.append(np.mean(diff))
        std.append(np.std(diff))
        median.append(np.median(diff))
        prob.append(np.mean(diff > 0))

    d = {
        "Pathwise Difference of Terminal P&Ls Comparing With": labels,
        "Mean": mean,
        "Median": median,
        "Std. Dev.": std,
        "Probability of Non-Negativity": prob,
    }
    df = pd.DataFrame(data=d)
    return df


class OnlineLearning:
    def __init__(
        self, file_name: str, update_freq: int = 30, days_per_year: int = 252
    ) -> None:
        """
        :param file_name: data file name for the online learning phase
        :param update_freq: update frequency for the online learning phase
        :param days_per_year: days per trading year for the online learning phase
        """
        super(OnlineLearning, self).__init__()
        self.data = pd.read_pickle(file_name)
        self.num_traj = self.data.shape[0]
        self.time_step = self.data.shape[-1]
        self.data_mean_traj = np.zeros([self.num_traj, 4, self.time_step])
        self.update_freq = update_freq
        self.days_per_year = days_per_year

        for _ in range(self.num_traj):
            self.data_mean_traj[_, 0, :] = np.mean(
                self.data[_, 0, :, :], axis=0
            )  # RL online
            self.data_mean_traj[_, 1, :] = np.mean(
                self.data[_, 1, :, :], axis=0
            )  # RL without online
            self.data_mean_traj[_, 2, :] = np.mean(
                self.data[_, 2, :, :], axis=0
            )  # Correct Delta
            self.data_mean_traj[_, 3, :] = np.mean(
                self.data[_, 3, :, :], axis=0
            )  # Incorrect Delta

    def best_worst_traj(
        self,
        best_idx: int,
        worst_idx: int,
        labels: List[str],
        markers: List[str],
        styles: List[str],
        ylim: List[float],
        marker_size: int = 6,
        marker_every: int = 1,
        update_freq: int = 30,
        day_per_year: int = 252,
        figsize: tuple = (9, 12),
        pad: float = 4.0,
        bbox_to_anchor: tuple = (1.05, 1.35),
        loc: str = "upper left",
        set_size: str = "large",
        font_size: int = 15,
    ) -> None:
        """
        The method to plot the best case vs worst case mean trajectories
        :param best_idx: the index for the best case trajectory
        :param worst_idx: the index for the worst case trajectory
        :param labels: labels for trajectory in the plot
        :param markers: markers for trajectory in the plot
        :param styles: styles for trajectory in the plot
        :param ylim: range limit of y-axis
        :param marker_size: marker size in the plot
        :param marker_every: marker interval in the plot
        :param update_freq: update frequency for the online learning
        :param day_per_year: days per year in the market environment
        :param figsize: size of the figure
        :param pad: pad value
        :param bbox_to_anchor: position of anchor
        :param loc: legend location
        :param set_size: font size for the legend
        :param font_size: font size for the labels for axis
        """
        plt.rcParams.update({"font.size": font_size})
        x = np.arange(0, self.time_step) * update_freq / day_per_year
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        fontP = FontProperties()
        fontP.set_size(set_size)
        fig.tight_layout(pad=pad)
        for _ in range(2):  # since we only have two cases
            if _ == 0:  # best case
                idx = best_idx
                (p_1,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 0, :],
                    styles[0],
                    label=labels[0],
                    marker=markers[0],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_2,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 1, :],
                    styles[1],
                    label=labels[1],
                    marker=markers[1],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_3,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 2, :],
                    styles[2],
                    label=labels[2],
                    marker=markers[2],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_4,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 3, :],
                    styles[3],
                    label=labels[3],
                    marker=markers[3],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                ax[_].set_xlabel("Year")
                ax[_].set_ylim(ylim)
                ax[_].set_ylabel("Sample Mean of Terminal Profit and Loss")
                ax[_].set_title("Best-Case Sample of Future Trajectories")
            else:
                idx = worst_idx
                (p_1,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 0, :],
                    styles[0],
                    label=labels[0],
                    marker=markers[0],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_2,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 1, :],
                    styles[1],
                    label=labels[1],
                    marker=markers[1],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_3,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 2, :],
                    styles[2],
                    label=labels[2],
                    marker=markers[2],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                (p_4,) = ax[_].plot(
                    x,
                    self.data_mean_traj[idx, 3, :],
                    styles[3],
                    label=labels[3],
                    marker=markers[3],
                    markersize=marker_size,
                    markevery=marker_every,
                )
                ax[_].set_xlabel("Year")
                ax[_].set_ylim(ylim)
                ax[_].set_ylabel("Sample Mean of Terminal Profit and Loss")
                ax[_].set_title("Worst-Case Sample of Future Trajectories")

        # legend
        plt.legend(
            handles=[p_1, p_2, p_3, p_4],
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            prop=fontP,
        )

    def _catch_up_time_mean_calculator(
        self, array_1: np.ndarray, array_2: np.ndarray
    ) -> float:
        """
        helper function for calculating the first hitting time based on the mean value
        in a particular trajectory
        :param array_1: first mean trajectory
        :param array_2: second mean trajectory
        :return: the first hitting time of the first mean trajectory to the second mean trajectory
        """
        time_diff = array_1 - array_2
        time_diff = list(time_diff)
        res = [i for i, x in enumerate(time_diff) if x >= 0]
        return (
            3
            if res == []
            else np.round(res[0] * self.update_freq / self.days_per_year, 2)
        )

    def catch_up_time_mean(self) -> None:
        """
        function to generate an array for the first catching-up times
        to the correct/incorrect Delta strategy in all the online learning trajectories
        """
        first_time_true = []
        first_time_incur = []
        for _ in range(self.num_traj):
            time_pts_cd = self._catch_up_time_mean_calculator(
                self.data_mean_traj[_, 0, :], self.data_mean_traj[_, 2, :]
            )
            time_pts_id = self._catch_up_time_mean_calculator(
                self.data_mean_traj[_, 0, :], self.data_mean_traj[_, 3, :]
            )
            first_time_true.append(time_pts_cd)
            first_time_incur.append(time_pts_id)
        self.first_time_cd_mean = np.array(first_time_true)
        self.first_time_id_mean = np.array(first_time_incur)

    def catch_up_time_mean_proba(self, type: str = "CD") -> np.ndarray:
        """
        Method that calculates the catch-up probability
        :param type: which catch-up probability is evaluated. If 'CD', it is evaluating the probability
        for correct Delta; if 'ID', it is evaluating the probability  for incorrect Delta
        :return: the catch-up probability
        """
        if type == "CD":
            return np.mean(self.first_time_cd_mean < 3)
        elif type == "ID":
            return np.mean(self.first_time_id_mean < 3)
        else:
            raise ValueError("the 'type' input must be either 'CD' or 'ID'")

    def catch_up_time_mean_plot(
        self,
        labels: List[str],
        figsize: tuple,
        font_size: int,
        bins: int,
        bwth: float,
        xlim: List[float],
        ylim: List[float],
        norm_hist: bool = True,
    ) -> None:
        """
        The method to produce the empirical density plot for the first catching-up time based on the mean
        :param labels: labels in the plot
        :param figsize: figure size of the plot
        :param font_size: font size in the plot
        :param bins: number of bins to be used
        :param bwth: bandwidth for the kernel
        :param xlim: range limit for x-axis
        :param ylim: range limit for y-axis
        :param norm_hist: normalize or not
        """
        plt.subplots(figsize=figsize)
        plt.rcParams.update({"font.size": font_size})
        for idx, data in enumerate([self.first_time_cd_mean, self.first_time_id_mean]):
            sns.distplot(
                data[self.first_time_cd_mean < 3],
                bins=bins,
                kde_kws={"bw": bwth},
                norm_hist=norm_hist,
                label=labels[idx],
            )
        plt.xlabel("Year")
        plt.ylabel("Empirical Density")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend()

    def catch_up_time_mean_stats(self) -> pd.DataFrame:
        """
        The method that creates a table of summary stats for the catching-up time based on mean
        :return: a pandas dataframe for summary statistics
        """

        # placeholders
        mean = []
        std = []
        median = []
        Var_90 = []
        Var_95 = []
        Tvar_90 = []
        Tvar_95 = []
        labels = ["Correct Delta", "Incorrect Delta"]

        for data in [
            self.first_time_cd_mean[self.first_time_cd_mean < 3],
            self.first_time_id_mean[self.first_time_cd_mean < 3],
        ]:
            mean.append(np.mean(data))
            std.append(np.std(data))
            median.append(np.median(data))
            Var_90.append(np.percentile(data, 90))
            Var_95.append(np.percentile(data, 95))
            var_90 = np.percentile(data, 90)
            Tvar_90.append(np.mean(data[data > var_90]))
            var_95 = np.percentile(data, 95)
            Tvar_95.append(np.mean(data[data > var_95]))

        d = {
            "Reinforcement Learning Agent"
            "with Online Learning Phase"
            "First Surpassing Time to": labels,
            "Mean": mean,
            "Median": median,
            "Std. Dev.": std,
            "Var_90": Var_90,
            "Var_95": Var_95,
            "TVaR_90": Tvar_90,
            "TVaR_95": Tvar_95,
        }
        df = pd.DataFrame(data=d)

        return df

    def p_val_traj(self) -> None:
        """
        The method to compute the p-value of the hypothesis test that the mean of
        the RL agent with online learning is less or equal to that of the correct/incorrect Delta
        """
        self.p_wrt_cd = []  # p-value w.r.t. the correct delta for a one-side t-test
        self.p_wrt_id = []  # p-value w.r.t. the incorrect delta for a one-side t-test
        for traj in tqdm(range(self.num_traj)):
            p_wrt_cd_buffer = []
            p_wrt_id_buffer = []
            for t in range(self.time_step):
                rl_obs_t = self.data[traj, 0, :, t]
                cd_obs_t = self.data[traj, 2, :, t]
                id_obs_t = self.data[traj, 3, :, t]

                # two sample t-test of RL vs correct delta and incorrect delta (null hypothesis: RL <= Delta)
                p_cd = stats.ttest_ind(rl_obs_t, cd_obs_t, alternative="greater")[1]
                p_id = stats.ttest_ind(rl_obs_t, id_obs_t, alternative="greater")[1]

                p_wrt_cd_buffer.append(p_cd)
                p_wrt_id_buffer.append(p_id)

            self.p_wrt_cd.append(p_wrt_cd_buffer)
            self.p_wrt_id.append(p_wrt_id_buffer)

    def _catch_up_time_p_calculator(
        self, crit_val: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The helper function to calculate the first catch-up time to the correct/incorrect Deltas
        based on the p-values compared with a critical value
        :param crit_val: the critical value to be compared with the p-values
        :return: an array contains the catch-up times
        """
        catch_up_time_p_cd = []
        catch_up_time_p_id = []
        for traj in range(self.num_traj):
            cd_buffer = [i for i, x in enumerate(self.p_wrt_cd[traj]) if x <= crit_val]
            if not cd_buffer:
                catch_up_time_p_cd.append(3)
            else:
                catch_up_time_p_cd.append(
                    np.round(cd_buffer[0] * self.update_freq / self.days_per_year, 2)
                )
            id_buffer = [i for i, x in enumerate(self.p_wrt_id[traj]) if x <= crit_val]
            if not id_buffer:
                catch_up_time_p_id.append(3)
            else:
                catch_up_time_p_id.append(
                    np.round(id_buffer[0] * self.update_freq / self.days_per_year, 2)
                )

        return np.array(catch_up_time_p_cd), np.array(catch_up_time_p_id)

    def catch_up_prob_p_table(self, crit_lvls: List[float]) -> pd.DataFrame:
        """
        The method that generates the table of catch-up probability at different critical levels
        :param crit_lvls: the list of critical values to be used for the catch-up probability computation
        :return: a dataframe for the catch-up probability at different critical levels
        """
        labels = ["Correct Delta", "Incorrect Delta"]
        crit_labels = ["a* = {}".format(lvls) for lvls in crit_lvls]
        cd_prob = []
        id_prob = []
        for lvls in crit_lvls:
            catch_up_cd, catch_up_id = self._catch_up_time_p_calculator(lvls)
            cd_prob.append(np.mean(catch_up_cd < 3))
            id_prob.append(np.mean(catch_up_id < 3))
        d = {"Estimated Proportion of Exceeding": labels}
        for _ in range(len(crit_labels)):
            d[crit_labels[_]] = [cd_prob[_], id_prob[_]]

        df = pd.DataFrame(d)
        return df

    def catch_up_time_p_plot(
        self,
        crit_lvl: float,
        labels: List[str],
        figsize: tuple,
        font_size: int,
        bins: int,
        bwth: float,
        xlim: List[float],
        ylim: List[float],
        norm_hist: bool = True,
    ) -> None:
        """
        The method to produce the empirical density plot for the first catching-up time based on the p-values
        :param crit_lvl: the critical level to be considered as a catchup if p-value <= crit_lvl
        :param labels: labels in the plot
        :param figsize: figure size of the plot
        :param font_size: font size in the plot
        :param bins: number of bins to be used
        :param bwth: bandwidth for the kernel
        :param xlim: range limit for x-axis
        :param ylim: range limit for y-axis
        :param norm_hist: normalize or not
        """
        plt.subplots(figsize=figsize)
        plt.rcParams.update({"font.size": font_size})
        first_time_cd_p, first_time_id_p = self._catch_up_time_p_calculator(crit_lvl)
        for idx, data in enumerate([first_time_cd_p, first_time_id_p]):
            sns.distplot(
                data[first_time_cd_p < 3],
                bins=bins,
                kde_kws={"bw": bwth},
                norm_hist=norm_hist,
                label=labels[idx],
            )
        plt.xlabel("Year")
        plt.ylabel("Empirical Density")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.legend()

    def catch_up_time_p_stats(self, crit_lvl: float) -> pd.DataFrame:
        """
        The method that creates a table of summary stats for the catching-up time based on p-values
        :param crit_lvl: the critical level to be considered as a catchup if p-value <= crit_lvl
        :return: a pandas dataframe for summary statistics
        """

        # placeholders
        mean = []
        std = []
        median = []
        Var_90 = []
        Var_95 = []
        Tvar_90 = []
        Tvar_95 = []
        labels = ["Correct Delta", "Incorrect Delta"]
        first_time_cd_p, first_time_id_p = self._catch_up_time_p_calculator(crit_lvl)

        for data in [
            first_time_cd_p[first_time_cd_p < 3],
            first_time_id_p[first_time_cd_p < 3],
        ]:
            mean.append(np.mean(data))
            std.append(np.std(data))
            median.append(np.median(data))
            Var_90.append(np.percentile(data, 90))
            Var_95.append(np.percentile(data, 95))
            var_90 = np.percentile(data, 90)
            Tvar_90.append(np.mean(data[data >= var_90]))
            var_95 = np.percentile(data, 95)
            Tvar_95.append(np.mean(data[data >= var_95]))

        d = {
            "Reinforcement Learning Agent"
            "with Online Learning Phase"
            "First Surpassing Time to": labels,
            "Mean": mean,
            "Median": median,
            "Std. Dev.": std,
            "Var_90": Var_90,
            "Var_95": Var_95,
            "TVaR_90": Tvar_90,
            "TVaR_95": Tvar_95,
        }
        df = pd.DataFrame(data=d)

        return df

    def snapshots_plot(
        self,
        t_idx: int,
        figsize: tuple,
        labels: List[str],
        x_lim: List[float],
        y_lim: List[float],
        font_size: int = 15,
        title_size: int = 15,
        label_size: int = 25,
        arrow_increment: float = 0.01,
        shape: str = "full",
        lw: float = 0,
        length_includes_head: bool = False,
        head_width: float = 0.05,
        loc: str = "upper left",
    ) -> None:
        """
        Method to create the snapshots for the empirical density of the mean terminal P&L over all online trajectory
        :param t_idx: time index for the snapshot
        :param figsize: figure size of the plot
        :param labels: labels in the plot
        :param x_lim: range limit for x-axis
        :param y_lim: range limit for y-axis
        :param font_size: font size in the plot
        :param title_size: fontsize of the axes title
        :param label_size: fontsize of the tick labels
        :param arrow_increment: increment for the arrow plot
        :param shape: shape of the arrow plot
        :param lw: line width of the arrow plot
        :param length_includes_head: whether to include the head of the arrow or not when determining the length
        :param head_width: head width of the arrow
        :param loc: location of the legend on the plot
        """

        plt.rc("legend", fontsize=font_size)  # legend fontsize
        plt.rc("axes", titlesize=title_size)  # fontsize of the axes title
        plt.rc("axes", labelsize=label_size)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=label_size)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=label_size)  # fontsize of the tick labels
        plt.subplots(figsize=figsize)
        FontProperties()
        if t_idx == 0:
            for strategy_idx in range(4):  # since we only have 4 strategies
                current_mean_data = self.data_mean_traj[0, strategy_idx, 0]
                temp_plot = plt.plot(
                    [current_mean_data, current_mean_data],
                    [0, 1],
                    label=labels[strategy_idx],
                )
                color = temp_plot[0].get_color()
                plt.arrow(
                    current_mean_data,
                    1,
                    0,
                    arrow_increment,
                    shape=shape,
                    lw=lw,
                    length_includes_head=length_includes_head,
                    head_width=head_width,
                    color=color,
                )

        else:
            for strategy_idx in range(4):  # again we only have 4 strategies
                mean_data = [
                    self.data_mean_traj[_, strategy_idx, t_idx] for _ in range(1000)
                ]
                sns.distplot(mean_data, label=labels[strategy_idx])

        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        plt.xlabel("Sample Mean of Terminal Profit and Loss")
        plt.ylabel("Empirical Density")
        plt.legend(loc=loc)

    def snapshots_stats(self, t_idx: int) -> pd.DataFrame:
        """
        The method to produce the table of summary statistics for the snapshot at different time index
        :param t_idx: time index for the snapshot
        :return: a pandas dataframe for the summary statistics
        """
        # placeholders
        mean = []
        std = []
        median = []
        Var_90 = []
        Var_95 = []
        Tvar_90 = []
        Tvar_95 = []
        labels = ["RL with OL", "RL without OL", "Correct Delta", "Incorrect Delta"]

        for strategy_idx in range(4):
            data = np.array(
                [self.data_mean_traj[_, strategy_idx, t_idx] for _ in range(1000)]
            )
            mean.append(np.mean(data))
            std.append(np.std(data))
            median.append(np.median(data))
            Var_90.append(np.percentile(data, 10))
            Var_95.append(np.percentile(data, 5))
            var_90 = np.percentile(data, 10)
            Tvar_90.append(np.mean(data[data <= var_90]))
            var_95 = np.percentile(data, 5)
            Tvar_95.append(np.mean(data[data <= var_95]))
        d = {
            "Sample Mean of Terminal P&L by": labels,
            "Mean": mean,
            "Median": median,
            "Std. Dev.": std,
            "Var_90": Var_90,
            "Var_95": Var_95,
            "TVaR_90": Tvar_90,
            "TVaR_95": Tvar_95,
        }
        df = pd.DataFrame(data=d)

        return df
