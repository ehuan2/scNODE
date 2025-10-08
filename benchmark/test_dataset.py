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

from benchmark.BenchmarkUtils import loadSCData, tpSplitInd, tunedOurPars, splitBySpec, Dataset, sampleGaussian
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
    return train_data, train_tps, test_data, test_tps, traj_data, tps, traj_cell_types, times_sorted

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


def model_training(train_data, train_tps, traj_data, tps):
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

    # now we turn everything to GPU if we can:
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    latent_ode_model = latent_ode_model.to(device)

    # dist.init_process_group('nccl')
    # if torch.cuda.device_count() > 1:
        # latent_ode_model = DDP(latent_ode_model, device_ids=[local_rank], output_device=local_rank)

    train_tps = train_tps.to(device)

    # latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq = scNODETrainWithPreTrain(
    latent_ode_model = scNODETrainWithPreTrain(
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
        data_name=data_name
    )

    print(f'Finish entire pretraining!')

    # dist.destroy_process_group()

    # now we want to take the trajectory data and do something with it...
    return latent_ode_model


def visualize_umap_embeds(all_recon_obs, traj_data):
    from plotting.PlottingUtils import umapWithPCA
    from plotting.visualization import plotPredAllTime, plotPredTestTime
    from optim.evaluation import globalEvaluation

    # Visualization - 2D UMAP embeddings
    print("Compare true and reconstructed data...")
    true_data = [each.detach().numpy() for each in traj_data]
    
    # basically create true_cell_tps and pred_cell_tps which will be annotations for both the true data and predicted ones
    true_cell_tps = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(true_data)])
    pred_cell_tps = np.concatenate([np.repeat(t, all_recon_obs[:, t, :].shape[0]) for t in range(all_recon_obs.shape[1])])

    # now, we remove the t's from the equation, i.e. it's back to cells x genes
    reorder_pred_data = [all_recon_obs[:, t, :] for t in range(all_recon_obs.shape[1])]

    # then we map the umap of the true data
    true_umap_traj, umap_model, pca_model = umapWithPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50)
    # and then the predicted one
    pred_umap_traj = umap_model.transform(pca_model.transform(np.concatenate(reorder_pred_data, axis=0)))
    plotPredAllTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, fig_name=f'{data_name}_pred_all.png', title=f'Reconstruction of {data_name}')
    # plots the predicted time points reconstruction as well
    plotPredTestTime(true_umap_traj, pred_umap_traj, true_cell_tps, pred_cell_tps, test_tps.detach().numpy(), fig_name=f'{data_name}_pred_test.png', title=f'Prediction of {data_name}')

    # Compute evaluation metrics
    print("Compute metrics...")
    test_tps_list = [int(t) for t in test_tps]
    for t in test_tps_list:
        logging.debug("-" * 70)
        logging.debug("t = {}".format(t))
        # -----
        pred_global_metric = globalEvaluation(traj_data[t].detach().numpy(), all_recon_obs[:, t, :])
        # we'll get all the distances
        logging.debug(pred_global_metric)


def get_umap_embeddings(data):
    """
    Calculates the umap embeddings of the given data.
    """
    n_neighbors = 50
    min_dist = 0.1
    pca_pcs = 50
    # -----
    # Original data
    # print("All tps...")
    # all_tps = list(range(len(traj_data)))
    # all_cell_tps = np.concatenate([np.repeat(all_tps[idx], x.shape[0]) for idx, x in enumerate(traj_data)])
    # all_umap_traj, all_umap_model, all_pca_model = umapWithPCA(
    #     np.concatenate(traj_data, axis=0), n_neighbors=n_neighbors, min_dist=min_dist, pca_pcs=pca_pcs
    # )


def predict_cell_traj(traj_data, traj_cell_types):
    """
    Based on the trajectory data and its cell types, predict the cell trajectories
    """
    # first off, we'll compute the umap of the reconstructed observations
    # then we'll use the PAGA format to construct the cell trajectories
    # then we'll take these PAGA graphs and construct the distances between them
    # through the use of IpsenMikhailov metric
    pass
    

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"./logs/logging/app_{timestamp}.log"
    
    # Configure basic logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,                        # Minimum severity to log
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

    args = parser.parse_args()

    data_name = args.dataset
    split_type = "three_interpolation"

    # 27000 cells by 2000 genes (HSGs)
    ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(data_name, split_type, path_to_dir='../')

    # GABA: 27500 cells x 22500 genes
    # Full: 154000 cells x 26700 genes -- way too many probably...
    # Think about the total number of cells per timepoint
    # print(ann_data)
    train_data, train_tps, test_data, test_tps, traj_data, tps, traj_cell_types, times_sorted = prep_splits(ann_data, n_tps, data_name, cell_types)

    # now let's train the data
    latent_ode_model, _, _, _, _ = model_training(train_data, train_tps, traj_data, tps)
    n_sim_cells = 2000
    if args.v:
        # based on all the cells in the first time point, predict the next time points
        # INCLUDING the TEST time points
        all_recon_obs = scNODEPredict(
            latent_ode_model,
            traj_data[0],
            tps,
            n_cells=n_sim_cells
        )  # (# cells, # tps, # genes)
        visualize_umap_embeds(all_recon_obs, traj_data)

    print(f'Finish everything')
