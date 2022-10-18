import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
matplotlib.rcParams.update({"font.size": 14})
matplotlib.rcParams.update({"axes.labelsize": 16})
matplotlib.rcParams.update({"legend.fontsize": 16})
matplotlib.rcParams.update({"xtick.labelsize": 16, "ytick.labelsize": 16})
# pd.options.display.float_format = '{:,.6f}'.format
# cmap = plt.get_cmap("Dark2")
# cmap = plt.get_cmap('tab10')

strain_name_df = pd.read_excel("./data/strain_num_matching.xlsx", index_col=0)
num_days = 27
start_day = 1
# remove the first day from the trajectory data
t = np.linspace(start_day, num_days, num_days - start_day + 1)

stress_list = ["TET", "KM", "NFLX", "SS", "PLM", "NQO", "SDC", "MMC"]
evo192_res = pd.read_excel("./data/192evo_ic50.xlsx", index_col=0, skiprows=1)
parent_ic50 = pd.read_excel("./data/192evo_parent_ic50.xlsx", index_col=1, skiprows=1)[
    "Parent mean IC50 [log2(μg/mL)]"
]

# choose the strains to plot by defining start_strain & num_strains
start_strain = 1
num_strains = 4


def plot_time_series(
    strain,
    stress,
    color,
    start_day=start_day,
    zorder=1,
    alpha=0.15,
    marker=".",
    i=None,
    print_title=True,
    label=None,
):
    """
    plot resistance for 'strain'
    strain: corresponds to strain_num for each trajectory file.
    stress: which stress resistance you want to plot.
    start_day: day to start the trajectory
    color, zorder, alpha, marker: args for plt.plot()
    i:
    """

    title = strain_name_df.iloc[strain - 1][0]
    df = pd.read_csv("./data/trajectories/strain" + str(strain) + ".csv", index_col=0)
    trajectory = df.loc[stress]
    if strain == 26:
        # because strain26 died at day 15
        traj_len = trajectory.shape[0]
        plt.plot(
            np.linspace(start_day, traj_len, traj_len - start_day + 1),
            trajectory.iloc[start_day - 1 :],
            color=color,
            alpha=alpha,
            zorder=zorder,
            marker=marker,
            label=label,
        )
    else:
        plt.plot(
            t,
            trajectory.iloc[start_day - 1 :],
            color=color,
            alpha=alpha,
            zorder=zorder,
            marker=marker,
            label=label,
        )
    if i == 0 and print_title:
        plt.title(title, fontsize=15)


def plot_strain(
    strain_num_start,
    strain_num_end,
    stress1,
    stress2,
    strain_label,
    roll_win=None,
    cmap_name=None,
    cmap_level=6,
):
    for strain in range(strain_num_start, strain_num_end + 1):
        # parent strains in TET
        if cmap_name is not None:
            # cmap2 = plt.get_cmap(cmap_name, cmap_level)
            cmap2 = sns.color_palette(cmap_name, cmap_level)
        df = pd.read_csv(
            "./data/trajectories/strain" + str(strain) + ".csv", index_col=0
        )
        trajectory1 = df.loc[stress1]
        trajectory2 = df.loc[stress2]
        if roll_win is not None:
            trajectory1 = trajectory1.T.rolling(
                roll_win, min_periods=1, win_type="triang"
            ).mean()
            trajectory2 = trajectory2.T.rolling(
                roll_win, min_periods=1, win_type="triang"
            ).mean()
        if strain == strain_num_end:
            plt.plot(
                trajectory1.values,
                trajectory2.values,
                ".-",
                lw=0.9,
                markersize=4.8,
                color=cmap2[strain - strain_num_start + 2],
                label=strain_label,
                zorder=3,
            )
        else:
            plt.plot(
                trajectory1.values,
                trajectory2.values,
                ".-",
                lw=0.9,
                markersize=4.8,
                color=cmap2[strain - strain_num_start + 2],
                zorder=3,
            )
        plt.scatter(trajectory1[0], trajectory2[0], color="k", alpha=1, zorder=1, s=32)
        plt.scatter(trajectory1[-1], trajectory2[-1], color="gray", alpha=1, zorder=1, s=25)
    plt.xlabel(stress1 + " resistance")
    plt.ylabel(stress2 + " resistance")
