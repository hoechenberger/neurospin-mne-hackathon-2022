from __future__ import division
from matplotlib import pyplot as plt
import mne
import numpy as np
import os.path as op
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker


# def run_cluster_permutation_test_1samp(
#     data,
#     ch_type="eeg",
#     nperm=2 ** 12,
#     threshold=None,
#     n_jobs=6,
#     tail=0,
#     tmin=0,
#     tmax=1,
#     max_step=1,
# ):
#     # If threshold is None, it will choose a t-threshold equivalent to p < 0.05 for the given number of observations
#     # (only valid when using an t-statistic).
#     data_tmp = data.copy()

#     # compute adjacency
#     adjacency, list = mne.channels.find_ch_adjacency(data[1].info, ch_type)

#     # subset of the data, as array
#     data_array_chtype = np.array(
#         [
#             np.transpose(data[i].get_data(picks=ch_type, tmin=tmin, tmax=tmax))
#             for i in range(len(data_tmp))
#         ]
#     )

#     # subset of the time array
#     times = data[0].times
#     times = times[
#         np.absolute(times - tmin).argmin() : np.absolute(times - tmax).argmin()
#     ]

#     # stat func
#     # cluster_stats = permutation_cluster_1samp_test(data_array_chtype, threshold=threshold, n_jobs=n_jobs, verbose=True,
#     #                                                   tail=tail, n_permutations=2**12, adjacency=adjacency,
#     #                                                   out_type='indices')

#     cluster_stats = spatio_temporal_cluster_1samp_test(
#         data_array_chtype,
#         threshold=threshold,
#         n_jobs=n_jobs,
#         verbose=True,
#         tail=tail,
#         n_permutations=nperm,
#         adjacency=adjacency,
#         out_type="indices",
#         max_step=max_step,
#     )

#     return cluster_stats, data_array_chtype, ch_type, times


def extract_info_cluster(
    cluster_stats, p_threshold, times, info, data_array_chtype, ch_type
):
    """
    This function takes the output of
        cluster_stats = permutation_cluster_1samp_test(...)
        and returns all the useful things for the plots

    :return: dictionnary containing all the information:
    - position of the sensors
    - number of clusters
    - The T-value of the cluster

    """
    cluster_info = {
        "times": times * 1e3,
        "p_threshold": p_threshold,
        "ch_type": ch_type,
    }

    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < p_threshold)[0]
    pos = mne.find_layout(info, ch_type=ch_type).pos
    print("We found %i positions for ch_type %s" % (len(pos), ch_type))

    if len(good_cluster_inds) > 0:

        # loop over significant clusters
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)
            signals = data_array_chtype[..., ch_inds].mean(
                axis=-1
            )  # is this correct ??
            # signals = data_array_chtype[..., ch_inds].mean(axis=1)  # is this correct ??
            # Fosca TODO does this clust_val makes sense ?
            a = T_obs[time_inds, :]
            b = a[:, ch_inds]
            clust_val = np.mean(b)
            sig_times = times[time_inds]
            p_value = p_values[clu_idx]

            cluster_info[i_clu] = {
                "sig_times": sig_times,
                "time_inds": time_inds,
                "signal": signals,
                "clust_val": clust_val,
                "channels_cluster": ch_inds,
                "p_values": p_value,
            }

        cluster_info["pos"] = pos
        cluster_info["ncluster"] = i_clu + 1
        cluster_info["T_obs"] = T_obs
        if ch_type == "eeg":
            cluster_info["data_info"] = info
        else:
            cluster_info["data_info"] = info

    return cluster_info


def plot_clusters(
    cluster_info,
    ch_type,
    T_obs_max=5.0,
    fname="",
    figname_initial="",
    filter_smooth=False,
    outfile=None,
):
    """
    This function plots the clusters

    :param cluster_info:
    :param good_cluster_inds: indices of the cluster to plot
    :param T_obs_max: colormap limit
    :param fname:
    :param figname_initial:
    :return:
    """
    color = "r"
    linestyle = "-"
    T_obs_min = -T_obs_max

    for i_clu in range(cluster_info["ncluster"]):
        cinfo = cluster_info[i_clu]
        T_obs_map = cluster_info["T_obs"][cinfo["time_inds"], ...].mean(axis=0)
        mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
        mask[cinfo["channels_cluster"], :] = True
        fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.0))

        if ch_type != "grad":
            # issue when plotting grad (pairs) when there is a mask ??!
            image, _ = plot_topomap(
                T_obs_map,
                cluster_info["data_info"],
                extrapolate="head",
                mask=mask,
                axes=ax_topo,
                vmin=T_obs_min,
                vmax=T_obs_max,
                show=False,
                ch_type=ch_type,
            )
        else:
            image, _ = plot_topomap(
                T_obs_map,
                cluster_info["data_info"],
                extrapolate="head",
                axes=ax_topo,
                vmin=0,
                vmax=T_obs_max,
                show=False,
                ch_type=ch_type,
            )

        divider = make_axes_locatable(ax_topo)
        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar, format="%0.1f")
        ax_topo.set_xlabel(
            "Averaged t-map\n({:0.2f} - {:0.2f} ms)".format(
                *cinfo["sig_times"][[0, -1]]
            )
        )
        ax_topo.set(title=ch_type + ": " + fname)

        # signal average & sem (over subjects)
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        # for signal, name, col, ls in zip(cinfo['signal'], [fname], colors, linestyles):
        #     ax_signals.plot(cluster_info['times'], signal * 1e6, color=col, linestyle=ls, label=name)
        mean = np.mean(cinfo["signal"], axis=0)
        ub = mean + sem(cinfo["signal"], axis=0)
        lb = mean - sem(cinfo["signal"], axis=0)
        if filter_smooth:
            mean = savgol_filter(mean, 9, 3)
            ub = savgol_filter(ub, 9, 3)
            lb = savgol_filter(lb, 9, 3)
        ax_signals.fill_between(cluster_info["times"], ub, lb, color=color, alpha=0.2)
        ax_signals.plot(
            cluster_info["times"], mean, color=color, linestyle=linestyle, label=fname
        )

        # ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
        ax_signals.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax_signals.set_xlim([cluster_info["times"][0], cluster_info["times"][-1]])
        ax_signals.set_xlabel("Time [ms]")
        ax_signals.set_ylabel("Amplitude")

        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax),
            cinfo["sig_times"][0],
            cinfo["sig_times"][-1],
            color="orange",
            alpha=0.3,
        )
        # ax_signals.legend(loc='lower right')
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax_signals.get_yaxis().set_major_formatter(fmt)
        ax_signals.get_yaxis().get_offset_text().set_position(
            (-0.07, 0)
        )  # move 'x10-x', does not work with y
        title = "Cluster #{0} (p < {1:0.3f})".format(i_clu + 1, cinfo["p_values"])
        # title = 'Cluster #{0} (p = %0.03f)'.format(i_clu + 1, cinfo['p_values'])
        if outfile is not None:
            outfile.write("\n")
            outfile.write("----- Cluster number %i ------ \n" % (i_clu + 1))
            time_str = (
                str(cinfo["sig_times"][0])
                + " to "
                + str(cinfo["sig_times"][-1])
                + " ms"
            )
            cluster_value_str = ", cluster-value= " + str(cinfo["clust_val"])
            p_value_str = ", p = " + str(cinfo["p_values"])
            outfile.write(time_str + cluster_value_str + p_value_str)

        ax_signals.set(ylim=[ymin, ymax], title=title)

        fig.tight_layout(pad=0.5, w_pad=0)
        fig.subplots_adjust(bottom=0.05)
        fig_name = figname_initial + "_clust_" + str(i_clu + 1) + ".svg"
        print("Saving " + fig_name)
        # plt.savefig(fig_name, dpi=300)
        plt.show()
    # plt.close('all)')

    return True
