# benchmark_decoder.py.
# used to benchmark the decoder and encoder
# test_dataset.py
# used to test the data and examine the dataset
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark.BenchmarkUtils import Dataset, SplitType, loadSCData, tunedOurPars
from optim.running import constructscNODEModel, get_checkpoint_train_path
from plotting.PlottingUtils import umapWithPCA


def load_model(n_genes, split_type, args):
    act_name = "relu"

    latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(
        data_name, split_type
    )  # use tuned hyperparameters
    latent_ode_model = constructscNODEModel(
        n_genes,
        latent_dim=latent_dim,
        enc_latent_list=enc_latent_list,
        dec_latent_list=dec_latent_list,
        drift_latent_size=drift_latent_size,
        latent_enc_act="none",
        latent_dec_act=act_name,
        drift_act=act_name,
        ode_method="euler",
    )

    device = torch.device("cpu")
    latent_ode_model = latent_ode_model.to(device)

    checkpoint_path = get_checkpoint_train_path(
        cell_type=args.cell_type_to_train,
        use_continuous=True,
        use_hvgs=args.hvgs,
        use_normalized=args.normalize,
        data_name=args.dataset,
        split_type=args.split_type.value,
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

    latent_ode_model.load_state_dict(torch.load(checkpoint_path))
    return latent_ode_model


def prep_traj_data(ann_data):
    # now we need to create an order list of cell time points
    # and create an index for it
    times_sorted = sorted(ann_data.obs["numerical_age"].unique().tolist())
    cell_tps = ann_data.obs["numerical_age"]

    # so we need to create the update train tps and updated test tps by taking the union
    # between train tps and cell tps
    all_tps = list(range(len(times_sorted)))
    print(f"Cell time points: {cell_tps}, times sorted: {times_sorted}")

    data = ann_data.X
    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data in PyTorch form
    traj_data = [
        torch.FloatTensor(data[np.where(cell_tps == t)[0], :].toarray())
        for t in times_sorted
    ]
    tps = torch.FloatTensor(all_tps)
    return traj_data, tps, times_sorted


def predict_latent_embeds(latent_ode_model, first_tp, tps, n_sim_cells):
    latent_ode_model.eval()
    _, latent_preds, _ = latent_ode_model.predict(first_tp, tps, n_sim_cells)
    latent_preds = latent_preds.detach().numpy()
    return latent_preds


def visualize_cluster_embeds(
    ann_data, data_name, split_type, t, title=None, fig_name=None
):
    """
    Visualize UMAP embeddings colored by cell major clusters.

    Parameters
    ----------
    ann_data : AnnData
        Annotated single-cell dataset.
    title : str, optional
        Title for the plot.
    fig_name : str, optional
        If provided, saves figure to figs/{fig_name}. Otherwise, shows the plot.
    """
    # --- Extract data ---
    true_data = ann_data.X.toarray()
    true_cell_clusters = ann_data.obs["major_clust"].to_numpy()
    print(
        f"True data shape: {true_data.shape}, true cell cluster shape: {true_cell_clusters.shape}"
    )

    save_dir = f"./checkpoints/vis_embeds/{data_name}/{split_type}/timepoints/t_{t}"
    save_path = os.path.join(save_dir, "vis_embed.pkl")

    if os.path.exists(save_path):
        # load the umap model and pca model
        with open(save_path, "rb") as umap_file:
            models = pickle.load(umap_file)
        pca_model = models["pca"]
        umap_model = models["umap"]
        true_umap_traj = umap_model.transform(pca_model.transform(true_data))
        print(f"Successfully loaded embeddings from {save_path}")
    else:
        # --- Run PCA + UMAP ---
        true_umap_traj, umap_model, pca_model = umapWithPCA(
            true_data, n_neighbors=50, min_dist=0.1, pca_pcs=50
        )

        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, "wb") as umap_file:
            pickle.dump({"pca": pca_model, "umap": umap_model}, umap_file)
        print(f"Successfully saved visualization embeddings to {save_path}")

    # --- Prepare colors for clusters ---
    unique_clusters = np.unique(true_cell_clusters).tolist()
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", 20)
    color_list = [cmap(i) for i in range(n_clusters)]
    num_cols = max(1, len(unique_clusters) // 7)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("UMAP colored by major cluster", fontsize=15)

    for i, clust in enumerate(unique_clusters):
        cluster_idx = np.where(true_cell_clusters == clust)[0]
        ax.scatter(
            true_umap_traj[cluster_idx, 0],
            true_umap_traj[cluster_idx, 1],
            label=str(clust),
            color=color_list[i],
            s=20,
            alpha=0.9,
        )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        ncol=num_cols,
        title="Major Cluster",
    )

    if title is not None:
        plt.suptitle(title)
    if fig_name is not None:
        plt.savefig(f"{fig_name}", bbox_inches="tight")
    else:
        plt.show()

    return true_umap_traj, umap_model, pca_model


def visualize_timepoint_embeds(ann_data, times_sorted, data_name, split_type):
    # first thing to do is to loop over all the tps
    for t in times_sorted:
        timepoint_data = ann_data[ann_data.obs["numerical_age"] == t].copy()
        fig_dir = f"figs/embedding/{data_name}/{split_type}"
        fig_path = f"{fig_dir}/t_{t:.3f}.png"
        os.makedirs(fig_dir, exist_ok=True)
        visualize_cluster_embeds(
            timepoint_data,
            data_name,
            split_type,
            t,
            title=f"True cell type embeddings for timepoint {t}",
            fig_name=fig_path,
        )
        print(f"Finish visualization for time point {t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset_sel = [dataset.value for dataset in list(Dataset)]
    parser.add_argument(
        "-d",
        "--dataset",
        type=Dataset,
        choices=list(Dataset),
        metavar=f"{dataset_sel}",
        default=Dataset.HERRING_GABA,
        help="The dataset to evaluate from",
    )
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--traj_view", action="store_true")
    parser.add_argument("--hvgs", action="store_true")
    parser.add_argument("--per_cell_type", action="store_true")

    split_type_sel = [split_type.value for split_type in list(SplitType)]
    parser.add_argument(
        "-s",
        "--split_type",
        type=SplitType,
        choices=list(SplitType),
        metavar=f"{split_type_sel}",
        default=SplitType.THREE_INTERPOLATION,
        help="split type to choose from",
    )
    parser.add_argument("-n", "--normalize", action="store_true")

    # so we add an argument to train a specific cell type, if it doesn't exist
    # then we train all cell types
    parser.add_argument("--cell_type_to_train", type=str, default="")
    parser.add_argument("--cell_type_to_vis", type=str, default="")

    args = parser.parse_args()

    data_name = args.dataset
    split_type = args.split_type.value

    # 27000 cells by 2000 genes (HVGs) if true
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(
        data_name,
        split_type,
        path_to_dir="../",
        use_hvgs=args.hvgs,
        normalize_data=args.normalize,
    )

    traj_data, tps, times_sorted = prep_traj_data(ann_data)
    print(f"Trajectory data: {traj_data}, tps: {tps}, times_sorted: {times_sorted}")

    # simple: take the latent model
    # run prediction on it
    # take the latent_seq instead of recon_obs
    latent_ode_model = load_model(n_genes, split_type, args)
    print(f"Successfully loaded model")

    # then we take these embeddings and check how the clusters look per cell-type
    # and evaluate it
    # umap_embeds = timesteps x (umap_embed, pca_embed)
    # returns a umap embedding and a pca embedding per time point
    visualize_timepoint_embeds(ann_data, times_sorted, data_name, split_type)

    # 1) todo: visualize the umap embeddings at a time point, and then colour per cell type
    # in progress
    # 2) todo: visualize the umap embeddings of the learned embeddings at a time point, colour per cell type
    # ! Can do this by aggregating together the cell predictions
    # ! i.e. I predict per cell type, the trajectory of 200 cells
    # ! then I visualize them altogether and see what that looks like

    # then we predict the embeddings per time point
    # need to do the following:
    # predict per cell type (starting with traj data of a cell type)
    # combine all the data together per time point
    # then umap/pca them

    print(f"Cell types: {cell_types}")

    """
    latent_embeddings = [[] for _ in tps]

    num_cells = 300

    for i, cell_type in enumerate(cell_types):
        cell_type_data = ann_data[ann_data.obs['major_clust'] == cell_type].copy()
        cell_traj_data, _, _ = prep_traj_data(cell_type_data)

        cell_type_latent_embeddings = predict_latent_embeds(
            latent_ode_model,
            cell_traj_data[0],
            tps,
            num_cells
        )
        print(f'cell type latent embeddings: {cell_type_latent_embeddings.shape}')
        # join this together based on time -- range over the time domain
        for i in range(cell_type_latent_embeddings.shape[1]):
            latent_embeddings[i].append(cell_type_latent_embeddings[:,i,:])
            if i == 2:
                break

    # latent_embeddings should be tps x (cell_type, cells, genes)
    print(f'Shape: ({len(latent_embeddings)}, {len(latent_embeddings[0])})')

    # so we concatenate it on the 0th axis, so that way it becomes (cells, genes)
    # however, we also need to create the (cells,) cell_type labels
    for tp_embed in latent_embeddings:
        np.concatenate([np.repeat(cell_types[cell_type_idx], 300) for cell_type_idx in range(tp_embed.shape[0])])
        tp_embed = np.concatenate(tp_embed, axis=0)
        latent_embeddings = np.concatenate(latent_embeddings, axis=0)

    # print(f'Successfully predicted the latent embeddings')



    # 3) todo: compare the above two by checking cluster distances?
    # finish the above by tomorrow? hopefully?
    """
