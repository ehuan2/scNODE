# benchmark_time_infer.py.
# used to benchmark how good this is at making inference across time
import os

import numpy as np
import torch
import pprint
from tqdm import tqdm

from benchmark.BenchmarkUtils import (
    loadSCData,
    tunedOurPars,
    create_parser,
    tpSplitInd,
    Dataset,
)
from optim.evaluation import globalEvaluation
from optim.running import (
    constructscNODEModel,
    get_checkpoint_train_path,
    scNODEPredict,
)


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
"""


def prep_traj_data(ann_data, cell_type):
    # now we need to create an order list of cell time points
    # and create an index for it

    # now we get the cell ann data if we have that case instead
    if cell_type != "":
        cell_ann_data = ann_data[ann_data.obs["major_clust"] == cell_type].copy()
    else:
        cell_ann_data = ann_data.copy()

    times_sorted = sorted(cell_ann_data.obs["numerical_age"].unique().tolist())
    cell_tps = cell_ann_data.obs["numerical_age"]

    # so we need to create the update train tps and updated test tps by taking the union
    # between train tps and cell tps
    all_tps = list(range(len(times_sorted)))

    data = cell_ann_data.X
    # Convert to torch project
    # so right now, we have it s.t. if the time points do match up, we get the data in PyTorch form
    traj_data = [
        torch.FloatTensor(data[np.where(cell_tps == t)[0], :].toarray())
        for t in times_sorted
    ]
    tps = torch.FloatTensor(all_tps)
    return traj_data, tps, times_sorted


def prep_traj_data_non_herring(ann_data, n_tps):
    data = ann_data.X
    traj_data = [
        torch.FloatTensor(data[np.where(cell_tps == t)[0], :])
        for t in range(1, n_tps + 1)
    ]

    all_tps = list(range(n_tps))
    tps = torch.FloatTensor(all_tps)

    return traj_data, tps


def evaluate_time_inference(
    latent_ode_model, traj_data, tps, test_tps, times_sorted=None
):
    n_sim_cells = 2000
    all_recon_obs = scNODEPredict(
        latent_ode_model, traj_data[0], tps, n_cells=n_sim_cells
    )

    # now we want to measure the metrics found in the paper by predicting
    # the data, and then
    test_tps_list = [int(t) for t in test_tps]

    metrics = {}

    for t in test_tps_list:
        pred_global_metric = globalEvaluation(
            traj_data[t].detach().numpy(), all_recon_obs[:, t, :]
        )

        metrics[
            f"{times_sorted[t]}" if times_sorted is not None else t
        ] = pred_global_metric
    return metrics


def tps_to_continuous(tps, times_sorted):
    return torch.FloatTensor([times_sorted[int(tp)] for tp in tps])


def get_traj_data(data_name, split_type, ann_data, args, cell_type=None):
    if cell_type is not None:
        assert args.cell_type_to_train == ""
    else:
        cell_type = args.cell_type_to_train

    times_sorted = None
    if data_name in [Dataset.HERRING, Dataset.HERRING_GABA]:
        # now if the cell_type_to_train exists, we try to split the data for that portion
        traj_data, dis_tps, times_sorted = prep_traj_data(ann_data, cell_type)
        tps = tps_to_continuous(dis_tps, times_sorted)
    else:
        traj_data, tps = prep_traj_data_non_herring(ann_data, n_tps)

    _, test_tps = tpSplitInd(data_name, split_type, n_tps)

    # now we get the intersection of test_tps and tps, in case it's cell-type specific
    test_tps = [t for t in test_tps if t in dis_tps]
    print(f"Actual test time points: {test_tps}, with all tps being: {tps}")
    return traj_data, tps, test_tps, times_sorted


def get_cell_types(ann_data):
    return ann_data.obs["major_clust"].unique().tolist()


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--pretrain_only", action="store_true")
    # now if this specifies the per_cell_type flag, then we measure the OT
    # per cell type, creating a cell type x time points metrics
    parser.add_argument("--per_cell_type", action="store_true")

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

    # simple: take the latent model
    # take the latent_seq instead of recon_obs
    latent_ode_model = load_model(n_genes, split_type, args)
    print(f"Successfully loaded model")

    if not args.per_cell_type:
        traj_data, tps, test_tps, times_sorted = get_traj_data(
            data_name, split_type, ann_data, args
        )

        print(
            f'Evaluating for {args.cell_type_to_train if args.cell_type_to_train != "" else "all"}'
        )
        time_metrics = evaluate_time_inference(
            latent_ode_model, traj_data, tps, test_tps, times_sorted=times_sorted
        )
        print(f"Metrics: {time_metrics}")
        exit()

    cell_types = get_cell_types(ann_data)
    print(f"Evaluate for all cell types: {cell_types}")

    per_cell_types_metrics = {}

    for cell_type in tqdm(cell_types):
        traj_data, tps, test_tps, times_sorted = get_traj_data(
            data_name, split_type, ann_data, args, cell_type=cell_type
        )

        print(
            f'Evaluating for {args.cell_type_to_train if args.cell_type_to_train != "" else "all"}'
        )
        time_metrics = evaluate_time_inference(
            latent_ode_model, traj_data, tps, test_tps, times_sorted=times_sorted
        )
        per_cell_types_metrics[cell_type] = time_metrics

    pprint.pprint(per_cell_types_metrics)
