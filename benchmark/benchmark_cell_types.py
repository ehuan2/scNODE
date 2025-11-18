# benchmark_cell_types.py.
# used to benchmark the cell types that are inferred
# this one in particular infers the cell types using OT
import os
import pickle

from geomloss import SamplesLoss
from pykeops.torch import generic_sum
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark.BenchmarkUtils import (
    loadSCData,
    tunedOurPars,
    create_parser,
)
from optim.running import constructscNODEModel, get_checkpoint_train_path, add_to_dir
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


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
Unbalanced OT: {args.unbalanced_ot}, Scaling: {args.unbalanced_ot_scaling},
Blur: {args.unbalanced_ot_blur}, Reach: {args.unbalanced_ot_reach}
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


def get_embed_metric_dir():
    shared_path = f"{data_name}/{split_type}/embed_metrics"
    shared_path += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""
    shared_path += add_to_dir(args, args.pretrain_only)
    fig_dir = f"figs/" + shared_path
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def tps_to_continuous(tps, times_sorted):
    return torch.FloatTensor([times_sorted[int(tp)] for tp in tps])


def get_cell_pred_embeds_sequential(
    latent_ode_model, traj_data, tps, args, n_sim_cells=2000
):
    """
    Given the latent ODE model, the trajectory data (list of tensors),
    and the time points (tensor), get the predicted cell embeddings
    for all time points
    """
    all_cell_preds = []

    # as a good test, include the very first time point as well
    all_cell_preds.append(
        predict_latent_embeds(
            latent_ode_model, traj_data[0], tps[0:1], n_sim_cells=n_sim_cells
        )[:, 0, :]
    )

    for i in range(len(traj_data) - 1):
        first_tp = traj_data[i]
        next_tps = tps[i : i + 2]
        # we get the cell embeddings at the current time step and the next time step
        cell_preds = predict_latent_embeds(
            latent_ode_model, first_tp, next_tps, n_sim_cells=n_sim_cells
        )
        all_cell_preds.append(cell_preds[:, -1, :])  # only take the next time point
    return all_cell_preds


def get_cell_pred_embeds_joint(latent_ode_model, traj_data, tps, n_sim_cells=2000):
    """
    Given the latent ODE model, the trajectory data (list of tensors),
    and the time points (tensor), get the predicted cell embeddings
    for all time points using joint prediction
    """
    first_tp = traj_data[0]
    all_cell_preds = predict_latent_embeds(
        latent_ode_model, first_tp, tps, n_sim_cells * (len(tps) - 1)
    )
    return all_cell_preds


def get_cell_embed_by_timepoint(ann_data, times_sorted, latent_ode_model):
    """
    Given the annotated data and the sorted time points,
    get the cell embeddings by time point using the latent ODE model
    """
    all_cell_embeds = []
    all_cell_labels = []
    for t in times_sorted:
        tp_ann_data = ann_data[ann_data.obs["numerical_age"] == t].copy()
        data = tp_ann_data.X.toarray()
        labels = tp_ann_data.obs["major_clust"].to_numpy()
        embeddings = get_embedding(data, latent_ode_model)
        all_cell_embeds.append(embeddings)
        all_cell_labels.append(labels)

    return all_cell_embeds, all_cell_labels


def cell_types_to_one_hot(cell_types):
    """
    Given a list of cell types, convert to one-hot encoding
    """
    unique_clusters = np.unique(cell_types).tolist()
    type_to_index = {tp: i for i, tp in enumerate(unique_clusters)}
    # ! Important: This needs to be torch and not numpy or else it causes issues!
    one_hot = torch.zeros((len(cell_types), len(unique_clusters)))
    for i, tp in enumerate(cell_types):
        one_hot[i][type_to_index[tp]] = 1
    return one_hot, type_to_index, unique_clusters


def infer_cell_types_knn(true_embeds, pred_embeds, true_cell_types, args):
    """
    Based on the true embeddings, its true cell types and the predicted embeddings
    infer the cell types for the predicted embeddings using k-NN

    Return: Array of inferred cell types per time point
    """
    inferred_labels_by_time = []

    k = args.knn_k

    for t, true_embed in enumerate(true_embeds):
        tp_cell_types = true_cell_types[t]
        pred_embed = pred_embeds[t]

        # get the one-hot encoding of the true cell types
        one_hot_true_cell_types, type_to_index, unique_clusters = cell_types_to_one_hot(
            tp_cell_types
        )

        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(true_embed)
        _, indices = nn.kneighbors(pred_embed)

        # now we take the indices and get the labels by averaging over all the indices
        infer_label = np.zeros((pred_embed.shape[0], one_hot_true_cell_types.shape[1]))
        for i in range(pred_embed.shape[0]):
            neighbor_labels = one_hot_true_cell_types[indices[i], :]
            infer_label[i, :] = torch.mean(neighbor_labels, axis=0)

        inferred_labels_by_time.append(
            {"labels": infer_label, "mapping": type_to_index, "types": unique_clusters}
        )

    return inferred_labels_by_time


def get_labels(true_embed, pred_embed, args, one_hot_labels):
    """
    Given the true embeddings, predicted embeddings and one-hot encoding of true cell types,
    get the transport plan using optimal transport
    """
    ot_solver = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=args.unbalanced_ot_blur,
        debias=True,
        backend="tensorized",
        scaling=args.unbalanced_ot_scaling,
        potentials=True,
        reach=args.unbalanced_ot_reach if args.unbalanced_ot_reach else None,
    )

    F, G = ot_solver(pred_embed, true_embed)

    # ! Important: everything that is inputted to KeOps needs to be torch FloatTensor
    # ! Or else it just won't work... I hate this sometimes
    transfer = generic_sum(
        "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",  # See the formula above
        f"Lab = Vi({one_hot_labels.shape[1]})",  # Output:  one vector of size one_hot_labels per line
        "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
        f"X_i = Vi({pred_embed.shape[1]})",  # 2nd arg: one 2d-point per line
        f"Y_j = Vj({true_embed.shape[1]})",  # 3rd arg: one 2d-point per column
        "F_i = Vi(1)",  # 4th arg: one scalar value per line
        "G_j = Vj(1)",  # 5th arg: one scalar value per column
        f"L_j = Vj({one_hot_labels.shape[1]})",
    )  # 6th arg: one vector of size 3 per column

    # And apply it on the data (KeOps is pretty picky on the input shapes...):
    labels_i = (
        transfer(
            torch.Tensor([args.unbalanced_ot_blur**2]).type(torch.FloatTensor),
            pred_embed,
            true_embed,
            F.view(-1, 1),
            G.view(-1, 1),
            one_hot_labels,
        )
        / true_embed.shape[0]
    )

    return labels_i


def soft_labels_to_cell_types(labels_dict):
    """
    Given a dictionary given by infer_cell_types_ot at a specific timepoint,
    convert the soft labels to hard cell type labels
    """
    #  oh the labels are wrong here
    labels = labels_dict["labels"]
    clusters = labels_dict["types"]

    cell_type_labels = np.full(labels.shape[0], "", dtype=object)
    for i in range(labels.shape[0]):
        # if sum(labels[i]) == 0 or any(labels[i] == np.nan):
        #     cell_type_labels[i] = 'unknown'
        # else:
        # cell_type_labels[i] = clusters[np.argmax(labels[i]).item()]
        cell_type_labels[i] = clusters[np.argmax(labels[i]).item()]

    return cell_type_labels


def infer_cell_types_ot(true_embeds, pred_embeds, true_cell_types, args):
    """
    Based on the true embeddings, its true cell types and the predicted embeddings
    infer the cell types for the predicted embeddings using optimal transport

    Return: Array of soft inferred cell types per time point
    """
    # now we should use OT to infer the cell types
    # iterate over the time point for each embedding
    inferred_labels_by_time = []

    for t, true_embed in tqdm(enumerate(true_embeds)):
        tp_cell_types = true_cell_types[t]
        pred_embed = pred_embeds[t]

        # get the one-hot encoding of the true cell types
        one_hot_true_cell_types, type_to_index, unique_clusters = cell_types_to_one_hot(
            tp_cell_types
        )
        labels_i = get_labels(
            torch.tensor(true_embed),
            torch.tensor(pred_embed),
            args,
            one_hot_true_cell_types,
        )

        # now let's also calculate the transferred labels!
        inferred_labels_by_time.append(
            {"labels": labels_i, "mapping": type_to_index, "types": unique_clusters}
        )

    return inferred_labels_by_time


def get_all_embed_umap_path():
    shared_path = f"{data_name}/{split_type}/embed"
    shared_path += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""
    shared_path += add_to_dir(args, args.pretrain_only)
    shared_path += f"_pretrain_only" if args.pretrain_only else ""

    # --- Extract data ---
    save_dir = f"./checkpoints/vis_embeds/" + shared_path
    save_path = os.path.join(save_dir, "vis_embed.pkl")
    assert os.path.exists(save_path)
    return save_path


def plot_umap_embeddings(
    true_embeds, pred_embeds, true_cell_types, inferred_cell_types, times_sorted, args
):
    """
    Given the true embeddings, predicted embeddings, true cell types and inferred cell types,
    plot the UMAP embeddings for each time point. We will plot:

    All of these will share a single UMAP embedding space across time.

    1. The true embeddings colored by true cell types
    2. The predicted embeddings colored by inferred cell types
    3. The true and predicted embeddings together, colored by their cell types
    """
    shared_path = f"{data_name}/{split_type}"
    shared_path += add_to_dir(args, args.pretrain_only)
    shared_path += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""

    fig_dir = f"figs/embedding/" + shared_path
    fig_dir += "/knn" if args.use_knn else "/ot"
    os.makedirs(fig_dir, exist_ok=True)

    # ** NOTE: We rely on benchmark_encoder to create this. I don't know why but I can't get the
    # ** UMAP to look nice anymore, so we're relying on an old one.
    pkl_path = get_all_embed_umap_path()

    with open(pkl_path, "rb") as f:
        save_dict = pickle.load(f)
        umap_model = save_dict["umap"]

    def plot_by_cell_type(title, fig_path, umap_embed, cell_types):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(title, fontsize=15)

        unique_clusters = np.unique(cell_types).tolist()
        n_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap("tab20", 20)
        color_list = [cmap(i) for i in range(n_clusters)]
        num_cols = max(1, len(unique_clusters) // 10)

        for i, clust in enumerate(unique_clusters):
            cluster_idx = np.where(cell_types == clust)[0]
            ax.scatter(
                umap_embed[cluster_idx, 0],
                umap_embed[cluster_idx, 1],
                label=str(clust),
                color=color_list[i],
                s=10,
                alpha=0.7,
            )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            markerscale=2,
            ncol=num_cols,
            title="Major Cluster",
        )

        plt.savefig(fig_path, bbox_inches="tight")

    def plot_together(
        title, fig_path, true_umap_embed, pred_umap_embed, cell_types, pred_cell_types
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(title, fontsize=15)

        unique_clusters = np.unique(
            np.concatenate([cell_types, pred_cell_types])
        ).tolist()
        n_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap("tab20", 20)
        color_list = [cmap(i) for i in range(n_clusters)]
        num_cols = max(1, len(unique_clusters) // 10)

        for i, clust in enumerate(unique_clusters):
            true_cluster_idx = np.where(cell_types == clust)[0]
            pred_cluster_idx = np.where(pred_cell_types == clust)[0]
            ax.scatter(
                true_umap_embed[true_cluster_idx, 0],
                true_umap_embed[true_cluster_idx, 1],
                label=f"True {str(clust)}",
                color=color_list[i],
                s=10,
                alpha=0.7,
                marker="x",
            )
            ax.scatter(
                pred_umap_embed[pred_cluster_idx, 0],
                pred_umap_embed[pred_cluster_idx, 1],
                label=f"Predicted {str(clust)}",
                color=color_list[i],
                s=10,
                alpha=0.7,
                marker="o",
            )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            markerscale=2,
            ncol=num_cols,
            title="Major Cluster",
        )

        plt.savefig(fig_path, bbox_inches="tight")

    # for each time point, let's create the umap mappings
    for i, t in tqdm(enumerate(times_sorted)):
        tp_true_embed = true_embeds[i]
        tp_true_cell_type = true_cell_types[i]

        # now let's plot the true embeddings accordingly
        true_umap_embed = umap_model.transform(tp_true_embed)

        # 1) plot the true embeddings colored by true cell types
        title = f"UMAP colored by major cluster for time {t}"
        os.makedirs(f"{fig_dir}/umap_true", exist_ok=True)
        # plot_by_cell_type(
        #     title,
        #     f"{fig_dir}/umap_true/time_{t}.png",
        #     true_umap_embed,
        #     tp_true_cell_type
        # )

        # 2) plot the predicted embeddings colored by inferred cell types
        pred_embed = pred_embeds[i]
        pred_cell_type = soft_labels_to_cell_types(inferred_cell_types[i])

        pred_umap_embed = umap_model.transform(pred_embed)

        title = f"UMAP colored by inferred major cluster for time {t}"
        os.makedirs(f"{fig_dir}/umap_inferred", exist_ok=True)
        plot_by_cell_type(
            title,
            f"{fig_dir}/umap_inferred/time_{t}.png",
            pred_umap_embed,
            pred_cell_type,
        )

        # 3) plot the true and predicted embeddings together, colored by their cell types
        # os.makedirs(f"{fig_dir}/umap_together", exist_ok=True)
        # plot_together(
        #     f"UMAP of true and predicted embeddings at time {t}",
        #     f"{fig_dir}/umap_together/time_{t}.png",
        #     true_umap_embed,
        #     pred_umap_embed,
        #     tp_true_cell_type,
        #     pred_cell_type,
        # )


def calculate_ari(pred_embeds, inferred_cell_types, times_sorted):
    """
    Given the true cell types and inferred cell types across all time points,
    calculate the ARI score
    """
    assert len(pred_embeds) == len(inferred_cell_types) and len(pred_embeds) == len(
        times_sorted
    )
    for i, t in enumerate(times_sorted):
        pred_cell_type = soft_labels_to_cell_types(inferred_cell_types[i])
        ari_score = evaluate_ari(pred_embeds[i], pred_cell_type)
        print(f"Time {t} ARI score: {ari_score}")


def measure_metric(inferred_cell_types, true_cell_types, times_sorted, args):
    """
    Given the predicted embeddings, inferred cell types and true cell types,
    measure how accurate the cell labels are now, as they should be in order
    """
    for i, t in enumerate(times_sorted):
        pred_cell_type = soft_labels_to_cell_types(inferred_cell_types[i])
        true_cell_type = true_cell_types[i]
        # now we match up how many of them are equal
        assert len(pred_cell_type) == len(true_cell_type)
        accuracy = np.sum(pred_cell_type == true_cell_type) / len(true_cell_type)
        print(f"Time {t} Inferred Cell Type Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--use_sequential_pred", action="store_true")

    parser.add_argument("--measure_metric", action="store_true")

    # unbalanced OT measurements, such as scaling, blur, reach which are all floats
    parser.add_argument("--unbalanced_ot", action="store_true")
    parser.add_argument("--unbalanced_ot_scaling", type=float, default=0.5)
    parser.add_argument("--unbalanced_ot_blur", type=float, default=0.05)
    parser.add_argument("--unbalanced_ot_reach", type=float, default=None)

    parser.add_argument("--use_knn", action="store_true")
    parser.add_argument("--knn_k", type=int, default=10)

    parser.add_argument("--original_ari", action="store_true")

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

    if args.original_ari:
        embeds, cell_labels = get_cell_embed_by_timepoint(
            ann_data, times_sorted, latent_ode_model
        )
        # measure the original ARI as well
        for t in range(len(times_sorted)):
            ari = evaluate_ari(embeds[t], cell_labels[t])
            print(f"Original ARI at time {times_sorted[t]}: {ari}")
        exit()

    # now let's get the predicted cell embeddings, the true cell embeddings and its labels
    if args.measure_metric:
        pred_embeds, _ = get_cell_embed_by_timepoint(
            ann_data, times_sorted, latent_ode_model
        )
    elif args.use_sequential_pred:
        print("Using sequential prediction for cell embeddings")
        pred_embeds = get_cell_pred_embeds_sequential(
            latent_ode_model, traj_data, tps, args
        )
    else:
        print("Using joint prediction for cell embeddings")
        pred_embeds = get_cell_pred_embeds_joint(latent_ode_model, traj_data, tps, args)

    # ** Note cell prediction embeds are starting from time point 1, not time point 0 **
    true_embeds, true_cell_types = get_cell_embed_by_timepoint(
        ann_data, times_sorted, latent_ode_model
    )

    inferred_cell_types = []
    # and then calculate the inferred cell types using k-NN and/or OT
    if args.use_knn:
        print("Using k-NN to infer cell types")
        inferred_cell_types = infer_cell_types_knn(
            true_embeds, pred_embeds, true_cell_types, args
        )
    else:
        print("Using OT to infer cell types")
        inferred_cell_types = infer_cell_types_ot(
            true_embeds, pred_embeds, true_cell_types, args
        )

    if args.measure_metric:
        print(
            f'Measuring {"kNN" if args.use_knn else "OT"} inferred cell types against true cell types'
        )
        measure_metric(inferred_cell_types, true_cell_types, times_sorted, args)

    plot_umap_embeddings(
        true_embeds,
        pred_embeds,
        true_cell_types,
        inferred_cell_types,
        times_sorted,
        args,
    )
    calculate_ari(pred_embeds, inferred_cell_types, times_sorted)
