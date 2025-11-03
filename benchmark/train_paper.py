# test_dataset.py
# used to test the data and examine the dataset
import logging
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

from benchmark.BenchmarkUtils import (
    Dataset,
    loadSCData,
    splitBySpec,
    tpSplitInd,
    tunedOurPars,
    create_parser,
)
from optim.running import constructscNODEModel, scNODETrainWithPreTrain

# next we want to prepare the data for training
def prep_splits(ann_data, n_tps, data_name, split_type, cell_type):
    train_tps, test_tps = tpSplitInd(data_name, split_type, n_tps)
    data = ann_data.X

    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data
    # np.where returns a tuple, the array we care about is the first element
    traj_data = [
        torch.FloatTensor(data[np.where(cell_tps == t)[0], :])
        for t in range(1, n_tps + 1)
    ]

    all_tps = list(range(n_tps))
    train_data, test_data = splitBySpec(traj_data, train_tps, test_tps)
    tps = torch.FloatTensor(all_tps)
    train_tps = torch.FloatTensor(train_tps)
    test_tps = torch.FloatTensor(test_tps)
    n_cells = [each.shape[0] for each in traj_data]

    print(f"{train_data}, {test_data}")
    print("# tps={}, # genes={}".format(n_tps, n_genes))
    print("# cells={}".format(n_cells))
    print("Train tps={}".format(train_tps))
    print("Test tps={}".format(test_tps))
    return train_data, train_tps, test_data, test_tps, traj_data, tps


def model_training(
    train_data,
    train_tps,
    traj_data,
    tps,
    n_genes,
    split_type,
    use_hvgs,
    times_sorted,
    use_normalized=False,
    cell_type="",
    args={},
):
    # Model training
    pretrain_iters = 200
    pretrain_lr = 1e-3
    latent_coeff = args.beta  # regularization coefficient: beta
    epochs = 10
    iters = 100
    batch_size = 32
    lr = args.lr
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
        kl_coeff=args.kl_coeff,
        pretrain_iters=pretrain_iters,
        pretrain_lr=pretrain_lr,
        device=device,
        data_name=data_name,
        split_type=split_type,
        use_hvgs=use_hvgs,
        use_continuous=True,
        use_normalized=use_normalized,
        cell_type=cell_type,
        freeze_enc_dec=args.freeze_enc_dec,
        args=args,
    )

def train_and_visualize(
    train_data,
    train_tps,
    traj_data,
    tps,
    n_genes,
    args,
    test_tps,
    split_type,
    data_name,
    cell_type,
    all_times_sorted=None,
):
    # now let's train the data
    latent_ode_model, _, _, _, _ = model_training(
        train_data,
        train_tps,
        traj_data,
        tps,
        n_genes,
        split_type=split_type,
        use_hvgs=args.hvgs,
        times_sorted=all_times_sorted,
        use_normalized=args.normalize,
        cell_type=cell_type,
        args=args,
    )

    print(f"Finished training...")
    return latent_ode_model


if __name__ == "__main__":
    parser = create_parser()
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

    (
        train_data,
        train_tps,
        test_data,
        test_tps,
        traj_data,
        tps
    ) = prep_splits(ann_data, n_tps, data_name, split_type=split_type, cell_type="")

    latent_ode_model = train_and_visualize(
        train_data,
        train_tps,
        traj_data,
        tps,
        n_genes,
        args,
        test_tps,
        split_type,
        data_name,
        cell_type="",
    )

    print(f"Finish everything")
