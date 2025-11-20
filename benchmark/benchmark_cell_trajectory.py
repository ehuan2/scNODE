"""
benchmark_cell_trajectory.py

Measures the trajectory of each cell that we simulate,
and then compares whether this trajectory makes sense.
"""
import os

import numpy as np

from benchmark_cell_types import (
    load_model,
    tps_to_continuous,
    prep_traj_data,
    get_cell_pred_embeds_joint,
    get_cell_pred_embeds_sequential,
    get_cell_embed_by_timepoint,
    infer_cell_types_ot,
    add_args_to_parser,
    soft_labels_to_cell_types,
)

from benchmark.BenchmarkUtils import (
    loadSCData,
    create_parser,
)
import pandas as pd
from plot_sankey import plot_sankey_from_labels


def create_trajectory(inferred_cell_types):
    """
    Create a list of lists, where each list corresponds to the trajectory of a cell
    """
    n_tps = len(inferred_cell_types)
    trajectories = np.full(
        (inferred_cell_types[0]["labels"].shape[0], n_tps), "", dtype=object
    )

    for t in range(n_tps):
        pred_cell_types = soft_labels_to_cell_types(inferred_cell_types[t])

        for cell in range(pred_cell_types.shape[0]):
            trajectories[cell, t] = pred_cell_types[cell]

    return trajectories


def plot_trajectory(trajectories, times_sorted, cell_type):
    traj_dict = {tp: [] for tp in times_sorted}
    n_tps = len(times_sorted)

    for t in range(n_tps):
        for cell in range(trajectories.shape[0]):
            traj_dict[times_sorted[t]].append(trajectories[cell, t])

    # plot the trajectories that we care about
    trajectories = pd.DataFrame(traj_dict)

    path_dir = f"./figs/trajectories/{cell_type}/"
    os.makedirs(path_dir, exist_ok=True)
    plot_sankey_from_labels(
        trajectories,
        title="Inferred Cell Type Trajectories",
        path=os.path.join(path_dir, "sankey.html"),
    )


def plot_trajectory_per_cell_type(trajectories, times_sorted):
    for cell_type in set(trajectories.flatten()):
        # filter trajectories for this cell type
        filtered_trajectories = trajectories[trajectories[:, 0] == cell_type]
        plot_trajectory(filtered_trajectories, times_sorted, cell_type)


if __name__ == "__main__":
    parser = create_parser()
    add_args_to_parser(parser)
    args = parser.parse_args()

    data_name = args.dataset
    split_type = args.split_type.value

    # 154000 cells by 2000 genes (HVGs) if true
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(
        data_name,
        split_type,
        path_to_dir="../",
        use_hvgs=args.hvgs,
        normalize_data=args.normalize,
    )

    traj_data, tps, times_sorted = prep_traj_data(ann_data)
    tps = tps_to_continuous(tps, times_sorted)

    # simple: take the latent model
    # run prediction on it
    # take the latent_seq instead of recon_obs
    latent_ode_model = load_model(n_genes, split_type, args)
    print(f"Successfully loaded model")

    if args.use_sequential_pred:
        # TODO: need to actually fix this s.t. it's not just sequential, but sequential all the way
        pred_embeds = get_cell_pred_embeds_sequential(latent_ode_model, traj_data, tps)
    else:
        pred_embeds = get_cell_pred_embeds_joint(latent_ode_model, traj_data, tps)

    # ** Note cell prediction embeds are starting from time point 1, not time point 0 **
    true_embeds, true_cell_types = get_cell_embed_by_timepoint(
        ann_data, times_sorted, latent_ode_model
    )

    # now we use these true_embeds to infer the cell labels
    inferred_cell_types = infer_cell_types_ot(
        true_embeds, pred_embeds, true_cell_types, args
    )

    # now we can use these inferred cell types to create trajectories
    trajectories = create_trajectory(inferred_cell_types)
    plot_trajectory_per_cell_type(trajectories, times_sorted)
