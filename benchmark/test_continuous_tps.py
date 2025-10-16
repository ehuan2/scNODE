# test_dataset.py
# used to test the data and examine the dataset
import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist

from datetime import datetime
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec, Dataset, sampleGaussian, SplitType
from optim.running import constructscNODEModel, scNODETrainWithPreTrain, scNODEPredict


# cell types are simply based on the clustering, not anything else
# let's visualize these clusters using UMAP
def visualize_seurat_cluster(ann_data, data_name):
    # for either of them, we need to use PCA to reduce the dimensionality
    plt.figure(figsize=(20, 15))
    sc.tl.pca(ann_data, svd_solver='arpack')
    
    sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=40)
    sc.tl.umap(ann_data)
    sc.pl.umap(
        ann_data,
        title=f'U-Map of {data_name} dataset by Cell-Type',
        color='cell_type',
        legend_loc='right margin',
        legend_fontsize=6,
        legend_fontoutline=1
    )
    plt.tight_layout()

def prep_splits_herring(ann_data, n_tps, data_name, cell_types):
    # now we need to create an order list of cell time points
    # and create an index for it
    times_sorted = sorted(ann_data.obs['numerical_age'].unique().tolist())
    print(times_sorted, n_tps)
    
    train_tps, test_tps = tpSplitInd(data_name, split_type, n_tps)
    data = ann_data.X

    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data in PyTorch form
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :].toarray()) for t in times_sorted]
    traj_cell_types = [cell_types[np.where(cell_tps == t)[0]] for t in times_sorted]

    all_tps = list(range(n_tps))
    train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)

    tps = torch.FloatTensor(all_tps)
    train_tps = torch.FloatTensor(train_tps)
    test_tps = torch.FloatTensor(test_tps)
    
    n_cells = [tp_data.shape[0] for tp_data in traj_data]

    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))
    return train_data, train_tps, test_data, test_tps, traj_data, tps, traj_cell_types

# next we want to prepare the data for training
def prep_splits(ann_data, n_tps, data_name, cell_types):
    if data_name in [Dataset.HERRING, Dataset.HERRING_GABA]:
        return prep_splits_herring(ann_data, n_tps, data_name, cell_types=cell_types)
    
    train_tps, test_tps = tpSplitInd(data_name, split_type, n_tps)
    data = ann_data.X

    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data
    # np.where returns a tuple, the array we care about is the first element
    traj_data = [torch.FloatTensor(data[np.where(cell_tps == t)[0], :]) for t in range(1, n_tps + 1)]
    traj_cell_types = None
    if cell_types is not None:
        traj_cell_types = [
            cell_types[np.where(cell_tps == t)[0]] for t in range(1, n_tps + 1)
        ]

    all_tps = list(range(n_tps))
    train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
    tps = torch.FloatTensor(all_tps)
    train_tps = torch.FloatTensor(train_tps)
    test_tps = torch.FloatTensor(test_tps)
    n_cells = [each.shape[0] for each in traj_data]

    print(f'{train_data}, {test_data}')
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))
    return train_data, train_tps, test_data, test_tps, traj_data, tps, traj_cell_types


def tps_to_continuous(tps, times_sorted):
    return torch.FloatTensor([times_sorted[int(tp)] for tp in tps])

def model_training(
    train_data,
    train_tps,
    traj_data,
    tps,
    n_genes,
    split_type,
    use_hvgs,
    times_sorted,
    use_normalized=False
):
    # Model training
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = 1.0 # regularization coefficient: beta
    epochs = 10
    iters = 100
    batch_size = 32
    lr = 1e-3
    act_name = "relu"

    latent_dim, drift_latent_size, enc_latent_list, dec_latent_list = tunedOurPars(
        data_name, split_type
    ) # use tuned hyperparameters
    latent_ode_model = constructscNODEModel(
        n_genes,
        latent_dim=latent_dim,
        enc_latent_list=enc_latent_list,
        dec_latent_list=dec_latent_list,
        drift_latent_size=drift_latent_size,
        latent_enc_act="none",
        latent_dec_act=act_name,
        drift_act=act_name,
        ode_method="euler"
    )

    device = torch.device('cpu')
    latent_ode_model = latent_ode_model.to(device)

    train_tps = tps_to_continuous(train_tps, times_sorted)
    print(f'All the train tps: {train_tps}')

    train_tps = train_tps.to(device)

    return scNODETrainWithPreTrain(
        train_data,
        train_tps,
        latent_ode_model,
        latent_coeff=latent_coeff,
        epochs=epochs,
        iters=iters,
        batch_size=batch_size,
        lr=lr,
        pretrain_iters=pretrain_iters,
        pretrain_lr=pretrain_lr,
        device=device,
        data_name=data_name,
        split_type=split_type,
        use_hvgs=use_hvgs,
        use_continuous=True,
        use_normalized=use_normalized
    )


def visualize_umap_embeds(
    all_recon_obs,
    traj_data,
    times_sorted,
    test_tps,
    split_type,
    cell_type='all',
    use_normalized=False
):
    from plotting.PlottingUtils import umapWithPCA
    from plotting.visualization import plotPredAllTime, plotPredTestTime
    from optim.evaluation import globalEvaluation
    import os

    # Visualization - 2D UMAP embeddings
    print("Compare true and reconstructed data...")
    true_data = [each.detach().numpy() for each in traj_data]
    
    # basically create true_cell_tps and pred_cell_tps which will be annotations for both the true data and predicted ones
    true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
    pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])

    # we have now an array of length t, of 2000 cells x genes
    reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]

    # then we map the umap of the true data
    true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    # and then the predicted one
    pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))

    # create the directories if they don't exist
    dir = f'continuous{"/normalized" if use_normalized else ""}/{data_name}/{split_type}'
    os.makedirs(f'figs/{dir}', exist_ok=True)

    plotPredAllTime(
        true_umap_traj,
        pred_umap_traj,
        true_cell_tps,
        pred_cell_tps,
        fig_name=f'{dir}/cell_type_{cell_type}_pred_all.png',
        title=f'Reconstruction of {data_name} with {cell_type}'
    )
    # plots the predicted time points reconstruction as well
    plotPredTestTime(
        true_umap_traj,
        pred_umap_traj,
        true_cell_tps,
        pred_cell_tps,
        test_tps.detach().numpy(),
        fig_name=f'{dir}/cell_type_{cell_type}_pred_test.png',
        title=f'Prediction of {data_name} with {cell_type}'
    )

    """
    # Compute evaluation metrics
    print("Compute metrics...")
    test_tps_list = [int(t) for t in test_tps]
    for t in test_tps_list:
        logging.info("-" * 70)
        logging.info("t = {}".format(t))
        # -----
        pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
        # we'll get all the distances
        logging.info(pred_global_metric)
    """


def get_umap_embed(all_recon_obs, traj_data, train_data, train_tps):
    """
    Gets the umap embeddings of the 3 types we want to examine
    """
    from plotting.PlottingUtils import umapWithPCA
    n_neighbors = 50
    min_dist = 0.1
    pca_pcs = 50

    # Visualization - 2D UMAP embeddings
    print("Compare true and reconstructed data...")
    true_data = [each.detach().numpy() for each in traj_data]
    
    # it's now that we have an array of length t of 2000 cells x genes
    reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]

    # We want to build:
    # 1. True umap embedding
    # 2. True - test time points embeddings
    # 3. True - test + predicted test time point embeddings
    aug_traj_data = []
    for i, x in enumerate(reorder_pred_data):
        if i in train_tps:
            aug_traj_data.append(traj_data[i]) # use the original data if it exists
        else:
            aug_traj_data.append(x)

    # now we have:
    # 1 = traj_data
    # 2 = train_data
    # 3 = aug_traj_data
    # now we want to grab the tps associated with each one
    all_cell_tps = np.concatenate([np.repeat(idx, x.shape[0]) for idx, x in enumerate(traj_data)])
    removed_tps = np.concatenate([np.repeat(train_tps[idx], x.shape[0]) for idx, x in enumerate(train_data)])
    aug_cell_tps = np.concatenate([np.repeat(idx, x.shape[0]) for idx, x in enumerate(aug_traj_data)])
    
    # then we map the umap of the true data
    true_umap_traj, umap_model, pca_model = umapWithPCA(
        np.concatenate(true_data, axis=0),
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        pca_pcs=pca_pcs
    )

    # and then the predicted one
    removed_model_traj = umap_model.transform(
        pca_model.transform(
            np.concatenate(train_data, axis=0)
        )
    )

    # and finally the enhanced one
    aug_traj = umap_model.transform(
        pca_model.transform(
            np.concatenate(aug_traj_data, axis=0)
        )
    )

    # next we need to annotate the data
    def annotate_data(data, tps):
        ann_data = sc.AnnData(data)
        ann_data.obs["time"] = tps
        return ann_data

    all_ann_data = annotate_data(true_umap_traj, all_cell_tps)
    removed_ann_data = annotate_data(removed_model_traj, removed_tps)
    aug_ann_data = annotate_data(aug_traj, aug_cell_tps)
    return (all_ann_data, true_umap_traj, all_cell_tps), (removed_ann_data, removed_model_traj, removed_tps), (aug_ann_data, aug_traj, aug_cell_tps)


def get_paga_graph(ann_data, traj_data):
    """
    Given the whole data, the partially removed data and the augmented one,
    find the graph based on the trajectory
    """
    thr = 0.1

    data_conn = ann_data.uns["paga"]["connectivities"].todense()
    data_conn[np.tril_indices_from(data_conn)] = 0
    data_conn[data_conn < thr] = 0
    data_cell_types = ann_data.obs.louvain.values
    data_node_pos = [
        np.mean(traj_data[np.where(data_cell_types == str(c))[0], :], axis=0)
        for c in np.arange(len(np.unique(data_cell_types)))
    ]
    data_node_pos = np.asarray(data_node_pos)
    data_edge = np.asarray(np.where(data_conn != 0)).T
    return data_node_pos, data_edge, data_conn


def plot_paga(data, title, save_to):
    from plotting import linearSegmentCMap
    ann_data, traj_data, tps = data

    print("Neighbors")
    sc.pp.neighbors(ann_data, n_neighbors=5)
    print("Louvain")
    sc.tl.louvain(ann_data, resolution=0.5)
    print("PAGA")
    sc.tl.paga(ann_data, groups='louvain')

    data_node_pos, data_edge, data_conn = get_paga_graph(ann_data, traj_data)

    unique_tps = np.unique(tps).astype(int).tolist()
    n_tps = len(unique_tps)

    color_list = linearSegmentCMap(n_tps, "viridis")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    for t_idx, t in enumerate(tps):
        cell_idx = np.where(tps == t)[0]
        ax.scatter(
            traj_data[cell_idx, 0], traj_data[cell_idx, 1],
            color=color_list[t_idx], s=20, alpha=1.0
        )

    # the extra graph
    ax.scatter(data_node_pos[:, 0], data_node_pos[:, 1], s=30, color="k", alpha=1.0)
    for e in data_edge:
        ax.plot(
            [
                data_node_pos[e[0]][0],
                data_node_pos[e[1]][0]
            ],
            [
                data_node_pos[e[0]][1],
                data_node_pos[e[1]][1]
            ],
            "k-",
            lw=1.5
        )

    plt.tight_layout()
    plt.savefig(save_to) # can set dpi for better resolution
    plt.close(fig)
    return data_conn


def compare_paga_graphs(all_conn, other_conn, title):
    import networkx as ntx
    from netrd.distance import IpsenMikhailov
    all_graph = ntx.from_numpy_array(all_conn)
    other_graph = ntx.from_numpy_array(other_conn)
    dist_func = IpsenMikhailov()
    ip_dist = dist_func.dist(all_graph, other_graph)
    logging.info(f"{title}: {ip_dist}")


def predict_cell_traj(all_recon_obs, traj_data, train_data, train_tps, all_tps, data_name):
    """
    Based on the trajectory data and its cell types, predict the cell trajectories
    """
    # first off, we'll compute the umap of the reconstructed observations
    # then we'll use the PAGA format to construct the cell trajectories
    # then we'll take these PAGA graphs and construct the distances between them
    # through the use of IpsenMikhailov metric
    all_data, removed_data, aug_data = get_umap_embed(
        all_recon_obs,
        traj_data,
        train_data,
        train_tps
    )

    all_conn = plot_paga(
        all_data,
        f"Original ({data_name})",
        save_to=f'figs/original_paga_{data_name}.png'
    )
    removed_conn = plot_paga(
        removed_data,
        f"Removed timepoints ({data_name})",
        save_to=f'figs/removed_paga_{data_name}.png'
    )
    aug_conn = plot_paga(
        aug_data,
        f"Augmented timepoints ({data_name})",
        save_to=f'figs/aug_paga_{data_name}.png'
    )

    compare_paga_graphs(all_conn, removed_conn, 'All - Removed Divergence')
    compare_paga_graphs(all_conn, aug_conn, 'All - Augmented Divergence')


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"./logs/logging/app_{timestamp}.log"
    
    # Configure basic logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,                        # Minimum severity to log
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Format
        datefmt="%Y-%m-%d %H:%M:%S",               # Time format
    )

    parser = argparse.ArgumentParser()
    dataset_sel = [dataset.value for dataset in list(Dataset)]
    parser.add_argument(
        '-d',
        '--dataset',
        type=Dataset,
        choices=list(Dataset),
        metavar=f'{dataset_sel}',
        default=Dataset.HERRING_GABA,
        help='The dataset to evaluate from'
    )
    parser.add_argument('-v', action="store_true")
    parser.add_argument('--traj_view', action="store_true")
    parser.add_argument('--hvgs', action='store_true')
    parser.add_argument('--per_cell_type', action='store_true')

    split_type_sel = [split_type.value for split_type in list(SplitType)]
    parser.add_argument(
        '-s',
        '--split_type',
        type=SplitType,
        choices=list(SplitType),
        metavar=f'{split_type_sel}',
        default=SplitType.THREE_INTERPOLATION,
        help='split type to choose from'
    )
    parser.add_argument('-n', '--normalize', action='store_true')

    args = parser.parse_args()

    data_name = args.dataset
    split_type = args.split_type.value

    # 27000 cells by 2000 genes (HVGs) if true
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(
        data_name,
        split_type,
        path_to_dir='../',
        use_hvgs=args.hvgs,
        normalize_data=args.normalize
    )

    # GABA: 27500 cells x 22500 genes
    # Full: 154000 cells x 26700 genes -- way too many probably...
    # Think about the total number of cells per timepoint
    # print(ann_data)
    train_data, train_tps, test_data, test_tps, traj_data, tps, traj_cell_types =\
        prep_splits(
            ann_data,
            n_tps,
            data_name,
            cell_types
        )

    # now let's train the data
    times_sorted = sorted(ann_data.obs['numerical_age'].unique().tolist())
    latent_ode_model, _, _, _, _ = model_training(
        train_data,
        train_tps,
        traj_data,
        tps,
        n_genes,
        split_type=split_type,
        use_hvgs=args.hvgs,
        times_sorted=times_sorted,
        use_normalized=args.normalize
    )

    print(f'Finished training...')

    n_sim_cells = 2000
    
    tps = tps_to_continuous(tps, times_sorted)
    
    # based on all the cells in the first time point, predict the next time points
    # INCLUDING the TEST time points
    all_recon_obs = scNODEPredict(
        latent_ode_model,
        traj_data[0],
        tps,
        n_cells=n_sim_cells
    )  # (# cells, # tps, # genes)

    if args.v:
        visualize_umap_embeds(
            all_recon_obs,
            traj_data,
            times_sorted,
            test_tps,
            split_type=split_type,
            use_normalized=args.normalize
        )

        if args.per_cell_type and data_name in [Dataset.HERRING, Dataset.HERRING_GABA]:
            # we want to predict for each type of cell type
            major_clust = ann_data.obs['major_clust'].unique().tolist()
            print(f'Cell types: {major_clust}, num: {len(major_clust)}')

            all_times_sorted = sorted(ann_data.obs['numerical_age'].unique().tolist())
            for cell_type in major_clust:
                cell_type_data = ann_data[ann_data.obs['major_clust'] == cell_type].copy()
                cell_tps = cell_type_data.obs["numerical_age"]
                times_sorted = sorted(cell_type_data.obs['numerical_age'].unique().tolist())

                # cell type for traj data at ...
                cell_traj_data = [
                    torch.FloatTensor(
                        cell_type_data.X[np.where(cell_tps == t)[0], :].toarray()
                    )
                    for t in times_sorted
                ]

                # only predict the relevant time points based on times_sorted
                tps = torch.FloatTensor(times_sorted)
                print(f'Time points: {tps}, test_tps: {test_tps}')

                recon_obs = scNODEPredict(
                    latent_ode_model,
                    cell_traj_data[0],
                    tps,
                    n_cells=n_sim_cells
                )
                visualize_umap_embeds(
                    recon_obs,
                    cell_traj_data,
                    times_sorted,
                    test_tps,
                    split_type=split_type,
                    cell_type=cell_type,
                    use_normalized=args.normalize
                )

    if args.traj_view:
        print(tps)
        predict_cell_traj(all_recon_obs, traj_data, train_data, train_tps, tps, data_name)

    print(f'Finish everything')
