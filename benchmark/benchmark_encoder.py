# benchmark_decoder.py.
# used to benchmark the decoder and encoder
import ast
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from optim.evaluation import globalEvaluation
from benchmark.BenchmarkUtils import (
    loadSCData,
    tunedOurPars,
    create_parser,
)
from optim.running import constructscNODEModel, get_checkpoint_train_path, add_to_dir
from plotting.PlottingUtils import umapWithPCA, umapWithoutPCA
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import pprint


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

    _, checkpoint_path = get_checkpoint_train_path(
        use_continuous=True,
        use_normalized=args.normalize,
        cell_type=args.cell_type_to_train,
        data_name=args.dataset,
        use_hvgs=args.hvgs,
        split_type=args.split_type.value,
        kl_coeff=args.kl_coeff,
        pretrain_only=args.pretrain_only,
        freeze_enc_dec=args.freeze_enc_dec,
        args=args,
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

    print(f"Loaded model from {checkpoint_path}")

    latent_ode_model.load_state_dict(torch.load(checkpoint_path))
    return latent_ode_model


def get_description(args):
    """
    From the current args, get the description to put into logs/pred_embed_metrics
    """
    return f"""
Running for KL coefficient: {args.kl_coeff}, Pretrain Only: {args.pretrain_only}
Frozen Enc. Dec. Weights: {args.freeze_enc_dec} Full train KL coeff: {args.full_train_kl_coeff}
Beta: {args.beta}, LR: {args.lr}, Finetuning LR: {args.finetune_lr}, Vel Reg: {args.vel_reg}
Grad Norm: {args.grad_norm} Gamma: {args.gamma}
Batch size: {args.batch_size} OT Loss BS: {args.ot_loss_batch_size}
Epochs: {args.epochs}
"""


def get_embedding(data, latent_ode_model):
    """
    Given a numpy array of cells x genes, return the embedding
    """
    embeddings, _ = latent_ode_model.vaeReconstruct([data])
    embeddings = embeddings[0].detach().numpy()  # because we're only doing it for one
    return embeddings


def prep_traj_data(ann_data):
    # now we need to create an order list of cell time points
    # and create an index for it
    times_sorted = sorted(ann_data.obs["numerical_age"].unique().tolist())
    cell_tps = ann_data.obs["numerical_age"]

    # so we need to create the update train tps and updated test tps by taking the union
    # between train tps and cell tps
    all_tps = list(range(len(times_sorted)))

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
    true_data,
    true_cell_clusters,
    data_name,
    split_type,
    t,
    is_pred,
    args,
    is_embedding=False,
    title=None,
    plot_times=False,
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
    shared_path = f"{data_name}/{split_type}/{'pred' if is_pred else ('embed' if is_embedding else 'true')}"
    shared_path += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""
    shared_path += add_to_dir(args, args.pretrain_only)
    shared_path += f"_pretrain_only" if args.pretrain_only else ""

    fig_dir = f"figs/embedding/" + shared_path
    fig_dir += f"/measure_perfect" if args.measure_perfect else ""
    if args.use_all_embed_umap:
        fig_dir += f"/all_embed_umap"
    else:
        fig_dir += f"/time_specific_umap"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = f"{fig_dir}/t_{f'{t:.3f}' if not isinstance(t, str) else t}.png"

    # --- Extract data ---
    save_dir = f"./checkpoints/vis_embeds/" + shared_path
    if not args.use_all_embed_umap and args.measure_perfect:
        save_dir += f"/measure_perfect/t_{t}"
    save_path = os.path.join(save_dir, "vis_embed.pkl")

    if os.path.exists(save_path):
        # load the umap model and pca model
        with open(save_path, "rb") as umap_file:
            models = pickle.load(umap_file)
        if not is_pred and not is_embedding:
            pca_model = models["pca"]
            umap_model = models["umap"]
            true_umap_traj = umap_model.transform(pca_model.transform(true_data))
        else:
            umap_model = models["umap"]
            true_umap_traj = umap_model.transform(true_data)
        print(f"Successfully loaded embeddings from {save_path}")
    else:
        # --- Run PCA + UMAP ---
        print(f"Running umap with pca...")
        if not is_pred and not is_embedding:
            true_umap_traj, umap_model, pca_model = umapWithPCA(
                true_data,
                n_neighbors=10 if true_data.shape[0] > 100_000 else 50,
                min_dist=0.5,
                pca_pcs=50,
            )
            save_dict = {"pca": pca_model, "umap": umap_model}
        else:
            true_umap_traj, umap_model = umapWithoutPCA(
                true_data,
                n_neighbors=10 if true_data.shape[0] > 100_000 else 50,
                min_dist=0.5,
            )
            save_dict = {"umap": umap_model}

        os.makedirs(save_dir, exist_ok=True)

        with open(save_path, "wb") as umap_file:
            pickle.dump(save_dict, umap_file)
        print(f"Successfully saved visualization embeddings to {save_path}")

    # --- Prepare colors for clusters ---
    unique_clusters = np.unique(true_cell_clusters).tolist()
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap("tab20", 20)
    color_list = [cmap(i) for i in range(n_clusters)]
    num_cols = max(1, len(unique_clusters) // 10)

    if plot_times:
        norm = plt.Normalize(vmin=min(unique_clusters), vmax=max(unique_clusters))
        cmap = plt.cm.get_cmap("viridis")  # continuous colormap
        colors = cmap(norm(true_cell_clusters))  # get color for each point

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("UMAP colored by major cluster (continuous)", fontsize=15)

        sc = ax.scatter(
            true_umap_traj[:, 0],
            true_umap_traj[:, 1],
            c=true_cell_clusters,
            cmap=cmap,
            s=0.05 if true_data.shape[0] > 100_000 else 10,
            alpha=0.7,
        )

        # Add colorbar for continuous values
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Cluster (continuous scale)", rotation=270, labelpad=15)

    else:
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
                s=0.05 if true_data.shape[0] > 100_000 else 10,
                alpha=0.7,
            )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            markerscale=50 if true_data.shape[0] > 100_000 else 1,
            ncol=num_cols,
            title="Major Cluster",
        )

    if title is not None:
        plt.suptitle(title)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight")
    else:
        plt.show()


def visualize_timepoint_embeds(ann_data, times_sorted, data_name, split_type, args):
    # first thing to do is to loop over all the tps
    for t in [*times_sorted]:
        timepoint_data = ann_data[ann_data.obs["numerical_age"] == t].copy()
        true_data = timepoint_data.X.toarray()
        true_cell_clusters = timepoint_data.obs["major_clust"].to_numpy()
        print(
            f"True data shape: {true_data.shape}, true cell cluster shape: {true_cell_clusters.shape}"
        )

        visualize_cluster_embeds(
            true_data,
            true_cell_clusters,
            data_name,
            split_type,
            t,
            args=args,
            is_pred=False,
            title=(f"True cell type embeddings for timepoint {t:.3g}"),
        )
        print(f"Finish visualization for time point {t}")

    # visualize all of them together
    true_data = ann_data.X.toarray()
    true_cell_clusters = ann_data.obs["major_clust"].to_numpy()
    visualize_cluster_embeds(
        true_data,
        true_cell_clusters,
        data_name,
        split_type,
        "all",
        args=args,
        is_pred=False,
        title=(f"True cell type embeddings for all"),
    )
    print(f"Finish visualization for all")


def evaluate_ari(cell_embed, cell_labels):
    """
    This function is used to evaluate ARI using the lower-dimensional embedding
    cell_embed of the single-cell data, alongside its labels
    """
    adata = sc.AnnData(X=cell_embed)
    adata.obs["cell_type"] = cell_labels

    sc.pp.neighbors(adata, use_rep="X", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs["cell_type"], adata.obs["louvain"])
    return ari


def visualize_pred_embeds(ann_data, latent_ode_model, tps, metric_only, args):
    # Can do this by aggregating together the cell predictions
    # i.e. I predict per cell type, the trajectory of 200 cells
    # then I visualize them altogether and see what that looks like

    # need to do the following:
    # 1. Predict per cell type for all time points
    # a. predict per cell type (starting with traj data of a cell type)
    # b. combine all the data together per time point
    # c. then umap/pca them

    # skip the first args.mature_cell_tp_index time points
    tps = tps[args.mature_cell_tp_index :]
    print(f"Tps to evaluate: {tps}")

    cell_types = ann_data.obs["major_clust"].unique().tolist()

    latent_embeddings = [[] for _ in tps]

    for cell_type in cell_types:
        print(f"--- Running for cell type {cell_type} ---")
        cell_type_data = ann_data[ann_data.obs["major_clust"] == cell_type].copy()
        cell_traj_data, _, _ = prep_traj_data(cell_type_data)

        # start from args.mature_cell_tp_index time point
        if args.mature_cell_tp_index >= len(cell_traj_data):
            print(f"Skipping cell type {cell_type} as it has insufficient time points")
            continue

        start_data = cell_traj_data[args.mature_cell_tp_index]

        # the number of cells should just be the number of cells for that cell type at that timepoint
        cell_type_latent_embeddings = predict_latent_embeds(
            latent_ode_model, start_data, tps, n_sim_cells=start_data.shape[0]
        )
        print(f"cell type latent embeddings: {cell_type_latent_embeddings.shape}")
        # join this together based on time -- range over the time domain
        for j in range(cell_type_latent_embeddings.shape[1]):
            latent_embeddings[j].append(
                {"embed": cell_type_latent_embeddings[:, j, :], "cell_type": cell_type}
            )

    # latent_embeddings should be tps x (cell_type, cells, genes)
    print(
        f"time points x cell types x (cells, genes): ({len(latent_embeddings)}, "
        f"{len(latent_embeddings[0])}, {latent_embeddings[0][0]['embed'].shape})"
    )

    # so we concatenate it on the 0th axis, so that way it becomes (cells, genes)
    # however, we also need to create the (cells,) cell_type labels
    cell_type_final_labels = []
    time_labels = []
    final_embeds = []

    metrics = {}
    metrics["ari"] = {}
    for t_idx, tp_embed in enumerate(latent_embeddings):
        # tp_embed is cell_type x (cells, genes)
        cell_type_labels = np.concatenate(
            [
                np.repeat(cell_type_dict["cell_type"], cell_type_dict["embed"].shape[0])
                for cell_type_dict in tp_embed
            ]
        )
        tp_embed = np.concatenate(
            [cell_type_dict["embed"] for cell_type_dict in tp_embed], axis=0
        )

        # for altogether predictions
        cell_type_final_labels.append(cell_type_labels)
        time_labels.append(
            np.repeat(
                [times_sorted[t_idx + args.mature_cell_tp_index]], tp_embed.shape[0]
            )
        )
        print(time_labels[-1].shape)
        final_embeds.append(tp_embed)

        # now we just have to make sure that the timepoint index is correct
        timepoint = times_sorted[t_idx + args.mature_cell_tp_index]

        # now we visualize this, only if it's not the time visualization:
        if not metric_only and not args.vis_pred_times:
            visualize_cluster_embeds(
                tp_embed,
                cell_type_labels,
                data_name,
                split_type,
                timepoint,
                args=args,
                is_pred=True,
                title=f"Predicted encoder cell type embeddings for timepoint {timepoint:.3f}",
            )

        # now we should run the NMI and ARI metrics on this
        metrics["ari"][timepoint] = evaluate_ari(tp_embed, cell_type_labels)
        print(
            f"Successfully visualized the latent embeddings for time point: {timepoint}"
        )

    final_embeds = np.concatenate(final_embeds, axis=0)
    cell_type_final_labels = np.concatenate(cell_type_final_labels, axis=0)
    time_labels = np.concatenate(time_labels, axis=0)

    if args.ari_all:
        # finally, visualize all the embeddings together:
        metrics["ari"]["all"] = evaluate_ari(final_embeds, cell_type_final_labels)

    if not metric_only:
        final_labels = time_labels if args.vis_pred_times else cell_type_final_labels
        visualize_cluster_embeds(
            final_embeds,
            final_labels,
            data_name,
            split_type,
            "all" if not args.vis_pred_times else "all_times",
            args=args,
            is_pred=True,
            title=f"Predicted encoder cell type embeddings for all",
            plot_times=args.vis_pred_times,
        )

    with open(f"./logs/pred_embed_metrics.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)
    print(f"Finished writing ARI metrics for predicted embeddings")
    return metrics


def get_embed_metric_dir():
    shared_path = f"{data_name}/{split_type}/embed_metrics"
    shared_path += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""
    shared_path += add_to_dir(args, args.pretrain_only)
    fig_dir = f"figs/" + shared_path
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def plot_tp_starts(all_metrics):
    """
    Given the ARI metrics per time point start, do:
    1) Plot them together
    2) Plot the average ARI depending on the time point start
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    avg_ari_per_time = {}
    median_ari_per_time = {}
    min_ari_per_time = {}
    max_ari_per_time = {}

    for tp_index in all_metrics:
        ari_metrics = all_metrics[tp_index]["ari"]
        times = list([key for key in ari_metrics.keys() if key != "all"])
        ari_values = [ari_metrics[t] for t in times if t != "all"]

        # measure the mean, median, min and max
        avg_ari_per_time[tp_index] = np.mean(ari_values)
        median_ari_per_time[tp_index] = np.median(ari_values)
        min_ari_per_time[tp_index] = np.min(ari_values)
        max_ari_per_time[tp_index] = np.max(ari_values)

        ax.plot(
            range(tp_index, tp_index + len(times)),
            ari_values,
            marker="o",
            label=f"Start TP index {tp_index} (time {times[0]})",
        )

    ax.set_xlabel("Time Point Index")
    ax.set_ylabel("ARI")
    ax.set_title("ARI metrics across different time point starts")
    ax.grid(True)
    fig_dir = get_embed_metric_dir()
    fig.savefig(f"{fig_dir}/ari_metrics_different_tp_starts.png")
    plt.close(fig)

    # now plot the average ARI per time point start
    fig, ax = plt.subplots(figsize=(8, 6))
    tp_indices = list(avg_ari_per_time.keys())

    # now calculate the values we want to hold
    avg_ari_values = [avg_ari_per_time[idx] for idx in tp_indices]
    median_ari_values = [median_ari_per_time[idx] for idx in tp_indices]
    min_ari_values = [min_ari_per_time[idx] for idx in tp_indices]
    max_ari_values = [max_ari_per_time[idx] for idx in tp_indices]

    ax.plot(
        tp_indices,
        avg_ari_values,
        marker="o",
    )

    # Add error bars representing min/max range
    ax.vlines(
        tp_indices,
        min_ari_values,
        max_ari_values,
        color="tab:gray",
        alpha=0.6,
        label="Min–Max Range",
    )

    # Add median markers
    ax.scatter(
        tp_indices, median_ari_values, color="tab:red", marker="x", label="Median ARI"
    )

    ax.set_xlabel("Time Point Start Index")
    ax.set_ylabel("Average ARI")
    ax.set_title("Average ARI across time points vs. Time Point Start Index")
    ax.grid(True)
    fig.savefig(f"{fig_dir}/average_ari_vs_tp_start_index.png")
    plt.close(fig)

    print(f"Finished plotting ARI metrics across different time point starts")


def visualize_all_embeds(ann_data, latent_ode_model, metric_only, args):
    """
    Takes the model given, takes its embeddings and calculate
    its umap visualization, its ARI and its kBET
    """
    data = ann_data.X.toarray()
    labels = ann_data.obs["major_clust"].to_numpy()
    embeddings = get_embedding(data, latent_ode_model)

    metrics = {}
    metrics["ari"] = {}

    # embeddings should be cells (140000) x 50, (cells,)
    print(embeddings.shape, labels.shape)
    if not metric_only:
        visualize_cluster_embeds(
            embeddings,
            labels,
            data_name,
            split_type,
            "all",
            args=args,
            is_pred=False,
            is_embedding=True,
            title=f"Encoder cell type embeddings for all",
        )

    metrics["ari"]["all"] = evaluate_ari(embeddings, labels)
    with open(f"./logs/embed_metrics.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)

    print(f"Finish measuring the encoder on all cells")


def tps_to_continuous(tps, times_sorted):
    return torch.FloatTensor([times_sorted[int(tp)] for tp in tps])


def measure_perfect(latent_ode_model, ann_data, times_sorted, args):
    """
    Given your annotated data, and the timepoints given, calculate the
    ARI for each time point.
    """
    metrics = {}

    for tp in times_sorted:
        # calculate the annotated data's timepoint
        # we need to get the labels and the data itself
        tp_ann_data = ann_data[ann_data.obs["numerical_age"] == tp].copy()
        data = tp_ann_data.X.toarray()
        labels = tp_ann_data.obs["major_clust"].to_numpy()

        embeddings = get_embedding(data, latent_ode_model)

        if not args.metric_only:
            # visualize the embeddings now based on each time as well
            visualize_cluster_embeds(
                embeddings,
                labels,
                data_name,
                split_type,
                tp,
                args=args,
                is_pred=False,
                is_embedding=True,
                title=f"VAE embeddings of timepoint {tp}",
            )
            print(f"Finish visualizing for timepoint {tp}")

        metrics[tp] = evaluate_ari(embeddings, labels)

    print(f"Printing the ARI metrics per timepoint")

    with open(f"./logs/perfect_ari.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)

    # now we try to plot these values
    # we generate a plot with two lines, one for cur_and_pred_ot and one for pred_and_next_ot
    fig, ax = plt.subplots(figsize=(8, 6))
    # we get all of the timepoints until the last one, makes sense
    aris = [metrics[tp] for tp in metrics]
    times = range(len(times_sorted))
    ax.plot(
        times,
        aris,
        marker="o",
    )

    ax.set_xlabel("Time (index)")
    ax.set_ylabel("ARI value")
    ax.set_title(f"ARI over time")
    ax.grid(True)
    fig_dir = get_embed_metric_dir()
    fig.savefig(f"{fig_dir}/perfect.png")
    plt.close(fig)
    return metrics


def measure_ot_reg(latent_ode_model, ann_data, args):
    """
    Measures the OT regularization loss for every single timepoint
    """
    # steps:
    # 1) subset the data into each timepoint and iterate over it
    # 2) predict from the first timepoint what the next future time point embeddings will be
    # 3) calculate the wasserstein distances
    n_sim_cells = 2000
    traj_data, tps, times_sorted = prep_traj_data(ann_data)
    tps = tps_to_continuous(tps, times_sorted)

    latent_preds = predict_latent_embeds(
        latent_ode_model, traj_data[0], tps, n_sim_cells
    )

    metrics = {}

    for t_idx, t in enumerate(times_sorted):
        # calculate the distance from the predicted to the actual ones
        # now calculate the VAE of the traj data
        embeddings = get_embedding(traj_data[t_idx], latent_ode_model)

        pred_global_metric = globalEvaluation(latent_preds[:, t_idx, :], embeddings)

        metrics[t] = pred_global_metric

    with open(f"./logs/ot_reg.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)

    ots = [metrics[t]["ot"] for t in times_sorted]
    times = range(len(times_sorted))

    plt.figure(figsize=(8, 6))
    plt.plot(times, ots, marker="o")
    plt.xlabel("Time (index)")
    plt.ylabel("OT value")
    plt.title("OT metric across time")
    plt.grid(True)

    fig_dir = get_embed_metric_dir()
    plt.savefig(f"{fig_dir}/ot_reg.png")
    plt.close()

    return metrics


def measure_ot_pred(latent_ode_model, ann_data, args):
    """
    Measures the OT regularizations between the current time point
    and the next predicted timepoint
    """
    # steps:
    # 1) subset the data into each timepoint and iterate over it
    # 2) predict from the first timepoint what the next future time point embeddings will be
    # 3) calculate the wasserstein distances
    n_sim_cells = 2000
    traj_data, tps, times_sorted = prep_traj_data(ann_data)
    tps = tps_to_continuous(tps, times_sorted)

    # so we now have two ways we want to predict the next time point
    # 1) either from embed at t0 to t1, embed at t1 to t2, embed t2 to t3
    # 2) or from t0 to t1, t0 to t1 to t2, t0 to t1 to t2 to t3
    # number 2) is defined by the flag --use_time_zero_embed

    if args.use_time_zero_embed:
        # shape is (n_sim_cells, timepoints, latent_dim)
        # TODO: if the measure does not match up, it's likely due to
        # TODO: first latent sample != latent_seq at the first step
        latent_preds = predict_latent_embeds(
            latent_ode_model, traj_data[0], tps, n_sim_cells
        )
        # reorganize the shape so that we have timepoints x (n_sim_cells, latent_dim)
        latent_preds = np.transpose(latent_preds, (1, 0, 2))
    else:
        latent_preds = []
        for t in range(len(times_sorted) - 1):
            curr_tps = torch.FloatTensor([tps[t + 1]])
            # only predict the next time point
            latent_pred = predict_latent_embeds(
                latent_ode_model, traj_data[t], curr_tps, n_sim_cells
            )
            latent_preds.append(latent_pred[:, -1, :])

    metrics = {}

    for t_idx in range(len(times_sorted) - 1):
        t = times_sorted[t_idx]
        # calculate the distance from the predicted to the actual ones
        # now calculate the VAE of the traj data
        embeddings = get_embedding(traj_data[t_idx], latent_ode_model)
        next_embeddings = get_embedding(traj_data[t_idx + 1], latent_ode_model)

        metrics[t] = {
            "cur_and_pred_ot": globalEvaluation(latent_preds[t_idx][:, :], embeddings),
            "pred_and_next_ot": globalEvaluation(
                latent_preds[t_idx][:, :], next_embeddings
            ),
            "next_time": times_sorted[t_idx + 1],
        }

    with open(f"./logs/ot_pred.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)

    print(f"Finish writing the metrics... now plotting...")

    # now we try to plot these values
    # we generate a plot with two lines, one for cur_and_pred_ot and one for pred_and_next_ot
    fig, ax = plt.subplots(figsize=(8, 6))
    # we get all of the timepoints until the last one, makes sense
    cur_and_pred_ots = [metrics[t]["cur_and_pred_ot"]["ot"] for t in times_sorted[:-1]]
    pred_and_next_ots = [
        metrics[t]["pred_and_next_ot"]["ot"] for t in times_sorted[:-1]
    ]
    times = range(len(times_sorted) - 1)

    (line1,) = ax.plot(
        times,
        cur_and_pred_ots,
        marker="o",
        label="Current and Predicted OT",
        color="blue",
    )
    (line2,) = ax.plot(
        times, pred_and_next_ots, marker="o", label="Predicted and Next OT", color="red"
    )

    ax.set_xlabel("Time (index)")
    ax.set_ylabel("OT value")
    ax.set_title(
        f"OT metric across time ({'Uses time zero embed' if args.use_time_zero_embed else 'Uses sequential embeds'})"
    )
    ax.grid(True)
    ax.legend(
        handles=[line1, line2],
        loc="upper left",
    )

    fig_dir = get_embed_metric_dir()
    fig.savefig(f"{fig_dir}/ot_pred_use_time_zero_embed_{args.use_time_zero_embed}.png")
    plt.close(fig)


def measure_cell_counts(ann_data, times_sorted, args):
    """
    Measure the number of cells there are per timepoint, and calculate the dictionary
    such that dict[t] = dictionary of cell types to counts
    """

    cell_count_path = "./logs/cell_type_counts.txt"
    if not os.path.exists(cell_count_path):
        cell_types = ann_data.obs["major_clust"].unique().tolist()
        data = {}
        for t in times_sorted:
            timepoint_data = ann_data[ann_data.obs["numerical_age"] == t]
            data[t] = {}

            total = 0

            for cell_type in cell_types:
                print(f"Working on timepoint {t}, cell type: {cell_type}")
                data[t][cell_type] = (
                    (timepoint_data.obs["major_clust"] == cell_type).sum().item()
                )
                total += data[t][cell_type]

            data[t]["total"] = total

        pprint.pprint(data)

        with open(cell_count_path, "w") as f:
            pprint.pprint(data, stream=f, sort_dicts=True)

    with open(cell_count_path, "r") as f:
        text = f.read().strip()

    data = ast.literal_eval(text)
    data = {float(k): v for k, v in data.items()}  # ensure float timepoints

    time_points = sorted(data.keys())
    cell_types = [k for k in data[time_points[0]].keys() if k != "total"]
    raw_cell_counts = {
        cell: [data[t][cell] for t in time_points] for cell in cell_types
    }

    cell_ratios = {
        cell: [data[t][cell] / data[t]["total"] for t in time_points]
        for cell in cell_types
    }
    cell_counts = raw_cell_counts if not args.measure_cell_proportion else cell_ratios

    if args.measure_cell_count_time_idxs:
        time_points = range(len(data.keys()))

    # --- Step 3: Plot either one or all ---
    counts_or_ratio = "Counts" if not args.measure_cell_proportion else "Ratio"

    if args.measure_cell_specific:
        cell = args.measure_cell_specific
        if cell not in cell_counts:
            raise ValueError(
                f"Cell type '{cell}' not found. Available: {', '.join(cell_types)}"
            )

        plt.figure(figsize=(6, 4))
        plt.plot(time_points, cell_counts[cell], marker="o", color="tab:blue")
        plt.title(f"{cell} {counts_or_ratio} Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cell count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        if not args.measure_cell_count_time_idxs:
            plt.savefig(f"./figs/{cell}_{counts_or_ratio}.png")
        else:
            plt.savefig(f"./figs/{cell}_{counts_or_ratio}_time_idxs.png")

    else:
        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
        axes = axes.flatten()

        for i, cell in enumerate(cell_types):
            ax = axes[i]
            ax.plot(time_points, cell_counts[cell], marker="o")
            ax.set_title(cell, fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=8)
            if i // cols == rows - 1:
                ax.set_xlabel("Time")
            if i % cols == 0:
                ax.set_ylabel(f"{counts_or_ratio}")

        for j in range(len(cell_types), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Cell Type {counts_or_ratio} Over Time", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if not args.measure_cell_count_time_idxs:
            plt.savefig(f"./figs/cell_type_plot_{counts_or_ratio}.png")
        else:
            plt.savefig(f"./figs/cell_type_plot_time_idxs_{counts_or_ratio}.png")


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--vis_true", action="store_true")
    parser.add_argument("--vis_pred", action="store_true")
    parser.add_argument("--vis_pred_times", action="store_true")
    parser.add_argument("--vis_all_embeds", action="store_true")
    parser.add_argument("--metric_only", action="store_true")
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--measure_perfect", action="store_true")
    parser.add_argument("--use_all_embed_umap", action="store_true")
    parser.add_argument("--measure_ot_reg", action="store_true")
    parser.add_argument("--measure_ot_pred", action="store_true")
    parser.add_argument("--use_time_zero_embed", action="store_true")
    parser.add_argument("--mature_cell_tp_index", type=int, default=0)
    parser.add_argument("--measure_all_tp_starts", action="store_true")
    parser.add_argument("--ari_all", action="store_true")
    parser.add_argument("--measure_cell_counts", action="store_true")
    parser.add_argument("--measure_cell_count_time_idxs", action="store_true")
    parser.add_argument(
        "--measure_cell_specific",
        type=str,
        default=None,
        help="Optional: plot only this single cell type (e.g., --cell Astro)",
    )
    parser.add_argument("--measure_cell_proportion", action="store_true")

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

    # then we take these embeddings and check how the clusters look per cell-type
    # and evaluate it
    # umap_embeds = timesteps x (umap_embed, pca_embed)
    # returns a umap embedding and a pca embedding per time point
    if args.vis_true:
        visualize_timepoint_embeds(
            ann_data, times_sorted, data_name, split_type, args=args
        )

    # 2) visualize the umap embeddings of the learned embeddings at a time point, colour per cell type
    # basically also catch any errors (of forgetting the vis_pred flag) if they include the other ones
    if args.vis_pred or args.measure_all_tp_starts or args.mature_cell_tp_index != 0:
        if args.measure_all_tp_starts:
            # we want to run the prediction starting from each time point
            all_metrics = {}
            for start_tp_index in range(len(times_sorted) - 1):
                args.mature_cell_tp_index = start_tp_index
                print(
                    f"Measuring predicted embeddings starting from time point index {start_tp_index} (time {times_sorted[start_tp_index]})"
                )
                all_metrics[start_tp_index] = visualize_pred_embeds(
                    ann_data,
                    latent_ode_model,
                    tps,
                    metric_only=args.metric_only,
                    args=args,
                )

            # now we plot these ARI metrics across different tp starts
            plot_tp_starts(all_metrics)

        else:
            # 7) Measures the ARI metrics only after a specified timepoint (we would expect these timepoints to match up)
            visualize_pred_embeds(
                ann_data, latent_ode_model, tps, metric_only=args.metric_only, args=args
            )

    # 3) Measure how good the encoder is generally (encode all the ann_data measure its ARI)
    if args.vis_all_embeds:
        # we want to encode it first
        visualize_all_embeds(
            ann_data, latent_ode_model, metric_only=args.metric_only, args=args
        )

    # 4) Measures the perfect ARI
    if args.measure_perfect:
        measure_perfect(latent_ode_model, ann_data, times_sorted, args)

    # 5) Measures the regularization loss that we have
    # i.e. between Z^t and Z^{t + 1}
    if args.measure_ot_reg:
        measure_ot_reg(latent_ode_model, ann_data, args)

    # 6) Measures the OT between Z^t and predicted Z^{t + 1} and compare it with
    # the OT between predicted Z^{t + 1} and actual Z^{t + 1}
    if args.measure_ot_pred:
        measure_ot_pred(latent_ode_model, ann_data, args)

    # 8) Measures the raw cell type counts, and plots them as well
    if args.measure_cell_counts:
        measure_cell_counts(ann_data, times_sorted, args)
