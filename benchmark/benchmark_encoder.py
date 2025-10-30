# benchmark_decoder.py.
# used to benchmark the decoder and encoder
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

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
"""


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

    cell_types = ann_data.obs["major_clust"].unique().tolist()

    latent_embeddings = [[] for _ in tps]
    num_cells = 300

    for cell_type in cell_types:
        print(f"--- Running for cell type {cell_type} ---")
        cell_type_data = ann_data[ann_data.obs["major_clust"] == cell_type].copy()
        cell_traj_data, _, _ = prep_traj_data(cell_type_data)

        cell_type_latent_embeddings = predict_latent_embeds(
            latent_ode_model, cell_traj_data[0], tps, num_cells
        )
        print(f"cell type latent embeddings: {cell_type_latent_embeddings.shape}")
        # join this together based on time -- range over the time domain
        for j in range(cell_type_latent_embeddings.shape[1]):
            latent_embeddings[j].append(cell_type_latent_embeddings[:, j, :])

    # latent_embeddings should be tps x (cell_type, cells, genes)
    print(
        f"time points x cell types x (cells, genes): ({len(latent_embeddings)}, {len(latent_embeddings[0])}, {latent_embeddings[0][0].shape})"
    )

    # so we concatenate it on the 0th axis, so that way it becomes (cells, genes)
    # however, we also need to create the (cells,) cell_type labels
    final_labels = []
    final_embeds = []

    metrics = {}
    metrics["ari"] = {}
    for t_idx, tp_embed in enumerate(latent_embeddings):
        # tp_embed is cell_type x (cells, genes)
        cell_type_labels = np.concatenate(
            [
                np.repeat(cell_types[cell_type_idx], tp_embed[cell_type_idx].shape[0])
                for cell_type_idx in range(len(tp_embed))
            ]
        )
        tp_embed = np.concatenate(tp_embed, axis=0)

        # for altogether predictions
        final_labels.append(cell_type_labels)
        final_embeds.append(tp_embed)

        timepoint = times_sorted[t_idx]

        # now we visualize this:
        if not metric_only:
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
            f"Successfully visualized the latent embeddings for time point: {times_sorted[t_idx]}"
        )

    # finally, visualize all the embeddings together:
    final_labels = np.concatenate(final_labels, axis=0)
    final_embeds = np.concatenate(final_embeds, axis=0)
    if not metric_only:
        visualize_cluster_embeds(
            final_embeds,
            final_labels,
            data_name,
            split_type,
            "all",
            args=args,
            is_pred=True,
            title=f"Predicted encoder cell type embeddings for all",
        )
    metrics["ari"]["all"] = evaluate_ari(final_embeds, final_labels)

    with open(f"./logs/pred_embed_metrics.txt", "a") as f:
        f.write(get_description(args))
        pprint.pprint(metrics, stream=f, sort_dicts=True)
    print(f"Finished writing ARI metrics for predicted embeddings")


def visualize_all_embeds(ann_data, latent_ode_model, metric_only, args):
    """
    Takes the model given, takes its embeddings and calculate
    its umap visualization, its ARI and its kBET
    """
    data = ann_data.X.toarray()
    labels = ann_data.obs["major_clust"].to_numpy()
    embeddings, _ = latent_ode_model.vaeReconstruct([data])
    embeddings = embeddings[0].detach().numpy()  # because we're only doing it for one

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

        embeddings, _ = latent_ode_model.vaeReconstruct([data])
        embeddings = (
            embeddings[0].detach().numpy()
        )  # because we're only doing it for one

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
    pprint.pprint(metrics)
    return metrics


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--vis_true", action="store_true")
    parser.add_argument("--vis_pred", action="store_true")
    parser.add_argument("--vis_all_embeds", action="store_true")
    parser.add_argument("--metric_only", action="store_true")
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--measure_perfect", action="store_true")
    parser.add_argument("--use_all_embed_umap", action="store_true")

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
    if args.vis_pred:
        visualize_pred_embeds(
            ann_data, latent_ode_model, tps, metric_only=args.metric_only, args=args
        )

    # 3) Measure how good the encoder is generally (encode all the ann_data measure its ARI)
    if args.vis_all_embeds:
        # we want to encode it first
        visualize_all_embeds(
            ann_data, latent_ode_model, metric_only=args.metric_only, args=args
        )

    if args.measure_perfect:
        measure_perfect(latent_ode_model, ann_data, times_sorted, args)
