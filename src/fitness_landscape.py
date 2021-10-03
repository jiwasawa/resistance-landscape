import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cmap = plt.get_cmap("tab10")
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
matplotlib.rcParams.update({"font.size": 14})
matplotlib.rcParams.update({"axes.labelsize": 16})
matplotlib.rcParams.update({"legend.fontsize": 16})
matplotlib.rcParams.update({"xtick.labelsize": 16, "ytick.labelsize": 16})
cmap = plt.get_cmap("tab10")

traj_dir = "./data/trajectories/strain"
env_list = pd.read_excel("./data/strain_num_matching.xlsx", index_col=0)

# initializing PCA for trajectories under selection pressure
stress_list = ["TET", "KM", "NFLX", "SS", "PLM", "NQO", "SDC", "MMC"]


# preparing PCA transformation over the main 44 strains
pca = PCA(n_components=0.9)
ss = StandardScaler()

full_df = pd.DataFrame()
for i in range(1, 45):
    traj_df = pd.read_csv(traj_dir + str(i) + ".csv", index_col=0)
    full_df = full_df.append(traj_df.T, ignore_index=True)

full_df = pd.DataFrame(ss.fit_transform(full_df))
pca.fit(full_df)
pca_full_df = pca.transform(full_df)


# define grid_size using Freedman - Diaconis rule
pc_axis = 0
q25, q75 = np.quantile(pca_full_df[:, pc_axis], [0.25, 0.75])
grid_size = 2 * (q75 - q25) / (pca_full_df.shape[0] ** (1 / 3))
# print('grid_size: ', grid_size)

x_min = pca_full_df[:, 0].min()
x_max = pca_full_df[:, 0].max()
y_min = pca_full_df[:, 1].min()
y_max = pca_full_df[:, 1].max()


def plot_pca_traj(
    strain_name,
    traj_pca_df,
    traj_dir=traj_dir,
    start_time=0,
    c="light:blue",
    pca=pca,
    alpha=0.9,
    zorder=1,
    plot=True,
):
    """
    Plots trajectories in 2D PCA space for strains in a specific selection pressure.
    strain_name: ex. 'Parent in TET'
    traj_pca_df: DataFrame to store trajectory coordinates in PCA space
    traj_dir: directory for original trajectory data
    start_time: From 0 to 27. Sets the first time point for the trajectory
    """
    # strain_name = 'KME1 in TET'  # example
    filt1 = env_list["strain_env"].str.find(
        strain_name
    )  # returns index 0 if strain is valid, otherwise returns -1
    bool_strain = (filt1 != -1).values
    filt_strain_num_list = env_list[bool_strain].index.tolist()
    cmap = plt.get_cmap("tab10")
    if c == "gray":
        cmap2 = [cmap(7)] * 6
    else:
        cmap2 = sns.color_palette(c, 6)

    for i, strain_num in enumerate(filt_strain_num_list):
        traj = pd.read_csv(traj_dir + str(strain_num) + ".csv", index_col=0)
        traj_pca = pca.transform(ss.transform(traj.T))
        if plot:
            if i == 3:
                plt.plot(
                    traj_pca[start_time:, 0],
                    traj_pca[start_time:, 1],
                    ".-",
                    lw=1,
                    color=cmap2[i + 2],
                    alpha=alpha,
                    label=strain_name,
                    zorder=zorder,
                )
            else:
                plt.plot(
                    traj_pca[start_time:, 0],
                    traj_pca[start_time:, 1],
                    ".-",
                    lw=1,
                    color=cmap2[i + 2],
                    alpha=alpha,
                    zorder=zorder,
                )

            if c != "gray":
                plt.scatter(
                    traj_pca[start_time, 0],
                    traj_pca[start_time, 1],
                    color="k",
                    alpha=alpha,
                    s=90,
                )
        traj_pca_df = traj_pca_df.append(
            pd.DataFrame(traj_pca[start_time:, :]), ignore_index=True
        )

    return traj_pca_df


cmap = sns.color_palette("deep", as_cmap=True)


def quantize_map(
    stress_index, grid_size=grid_size, pca_full_df=pca_full_df, full_df=full_df
):
    """
    Output:
    fitness: resistance levels for stress (stress_index) for all strains
    ave_quantized: quantized map for resistance in the PCA space.

    """
    x_min = pca_full_df[:, 0].min()
    x_max = pca_full_df[:, 0].max()
    y_min = pca_full_df[:, 1].min()
    y_max = pca_full_df[:, 1].max()
    num_bins_x = int(np.ceil((x_max - x_min) / grid_size)) + 1
    num_bins_y = int(np.ceil((y_max - y_min) / grid_size)) + 1

    # sum up all points in a quantized grid (storage_matrix)
    # and calculate average using count_matrix
    storage_matrix = np.zeros((num_bins_y, num_bins_x))
    count_matrix = np.zeros((num_bins_y, num_bins_x))

    fitness = full_df.iloc[:, stress_index].values

    for i in range(full_df.shape[0]):
        x_index = int(round((pca_full_df[i, 0] - x_min) / grid_size))
        y_index = int(round((pca_full_df[i, 1] - y_min) / grid_size))

        storage_matrix[y_index, x_index] += fitness[i]
        count_matrix[y_index, x_index] += 1

    storage_matrix[count_matrix == 0] = np.nan
    count_matrix[count_matrix == 0] = 1  # to avoide zero division
    ave_quantized = storage_matrix / count_matrix

    return fitness, ave_quantized


def gaussian(x, mu, sigma):
    """
    Returns Gaussian with mean: 'mu' and standard deviation: 'sigma'.
    """
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    X = ((x - mu) / sigma) ** 2
    return norm * np.exp(-X / 2)


def fitted_2d_func(x, y, mu1, mu2, h1, h2, rel_fit):
    """
    Returns the superposition of multiple Gaussians based on the fitnotype data.
    The function is normalized so that the sum is one.
    mu1, mu2: list of means of the Gaussians for the PC1, PC2 axes, respectively.
    h1, h2: list of bandwiths for the Gaussian kernels.
    rel_fit: fitness data based on the quantized averaged fitnotype
    data passed from 'landscape_props()'.
    """
    gauss_sum = 0
    for nn in range(len(mu1)):
        gauss_sum += (
            rel_fit[nn] * gaussian(x, mu1[nn], h1) / h1 * gaussian(y, mu2[nn], h2) / h2
        )
    return gauss_sum / np.array(rel_fit).sum()


def landscape_props(stress_index, h=0.9, grid_size=grid_size):
    """
    Returns parameters for wrap_grad, grad_fitted_2d_func.
    stress_index: 0 - 7
    h: bandwidth for KDE in array measures.
    grid_size: grid length for quantizing the fitnotype map.
    """
    fitness, ave_quantized = quantize_map(
        stress_index=stress_index, grid_size=grid_size
    )
    nonnan_loc = np.argwhere(~np.isnan(ave_quantized))

    rel_fit = ave_quantized - np.nanmin(ave_quantized)
    rel_fit = [
        rel_fit[nonnan_loc[i][0], nonnan_loc[i][1]] for i in range(nonnan_loc.shape[0])
    ]

    mu1 = nonnan_loc[:, 1]
    mu2 = nonnan_loc[:, 0]

    # convert parameters to units of the PCA space
    Mu1 = mu1 * grid_size + x_min
    Mu2 = mu2 * grid_size + y_min
    H = h * grid_size

    return Mu1, Mu2, H, rel_fit


def grad_gaussian(x, mu, sigma):
    """
    Returns the derivative for a Gaussian N(mu, sigma) at x.
    """
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    X = ((x - mu) / sigma) ** 2
    return -(x - mu) * norm * np.exp(-X / 2) / (sigma ** 2)


def grad_fitted_2d_func(x, y, mu1, mu2, h1, h2, rel_fit):
    """
    Returns derivative for the inferred landscape for PC1, PC2 direction
    """
    gauss_sum_x = 0
    gauss_sum_y = 0
    for nn in range(len(mu1)):
        gauss_sum_x += (
            rel_fit[nn]
            * grad_gaussian(x, mu1[nn], h1)
            / h1
            * gaussian(y, mu2[nn], h2)
            / h2
        )
        gauss_sum_y += (
            rel_fit[nn]
            * gaussian(x, mu1[nn], h1)
            / h1
            * grad_gaussian(y, mu2[nn], h2)
            / h2
        )

    grad_x = gauss_sum_x / np.array(rel_fit).sum()
    grad_y = gauss_sum_y / np.array(rel_fit).sum()
    return grad_x, grad_y


def grad_ascent(
    grad, init, stress_index, l_params, n_epochs=10, eta=1, noise_strength=0
):
    """
    Gradient ascent-like algorithm for evolution simulations.
    eta: step size for evolution
    noise: white noise
    """
    Mu1, Mu2, H, rel_fit = l_params

    state = np.array(init)
    state_traj = np.zeros([n_epochs + 1, 2])
    state_traj[0, :] = init
    v = 0
    for j in range(n_epochs):
        noise = noise_strength * np.random.randn(state.size)  # white noise
        grad_s = np.array(grad(state, Mu1, Mu2, H, rel_fit))

        # normalize grad to norm 1
        v = eta * grad_s / np.linalg.norm(grad_s) + noise
        state = state + v
        state_traj[j + 1, :] = state
    return state_traj


def wrap_grad(state, Mu1, Mu2, H, rel_fit):
    """
    Wrapper for landscape gradients.
    Designed to be passed to 'grad_ascent'.
    """
    x, y = state
    return grad_fitted_2d_func(x=x, y=y, mu1=Mu1, mu2=Mu2, h1=H, h2=H, rel_fit=rel_fit)


def plot_traj(traj, color, ls=".-", alpha=1):
    plt.scatter(traj[0, 0], traj[0, 1], color=color, marker="o")
    plt.plot(traj[:, 0], traj[:, 1], ls, zorder=3, color=color, alpha=alpha)


# Functions for drawing gradient ascent simulation results
def get_init_state(strain_name, start=0, env_list=env_list, pca=pca, ss=ss):
    """
    Returns the coordinates in 2D PCA space for strains (strain_name).
    All strains (typically 4) that correspond to strain_name will be returned.
    """
    filt1 = env_list["strain_env"].str.find(
        strain_name
    )  # returns index 0 if strain is valid, otherwise returns -1
    bool_strain = (filt1 != -1).values
    filt_strain_num_list = env_list[bool_strain].index.tolist()

    init_state = np.empty((len(filt_strain_num_list), 2))
    for i, strain_num in enumerate(filt_strain_num_list):
        traj = pd.read_csv(traj_dir + str(strain_num) + ".csv", index_col=0)
        traj_pca = pca.transform(ss.transform(traj.T))[:, :2]
        init_state[i, :] = traj_pca[start, :]
    return init_state


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
