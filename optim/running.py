"""
Description:
    Main codes for scNODE and Dummy model training and prediction.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
"""
import itertools
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dist
from tqdm import tqdm

from benchmark.BenchmarkUtils import sampleGaussian
from model.diff_solver import ODE
from model.dynamic_model import scNODE
from model.layer import LinearNet, LinearVAENet
from optim.loss_func import MSELoss, SinkhornLoss

# =============================================


def constructscNODEModel(
    n_genes,
    latent_dim,
    enc_latent_list=None,
    dec_latent_list=None,
    drift_latent_size=[64],
    latent_enc_act="none",
    latent_dec_act="relu",
    drift_act="relu",
    ode_method="euler",
):
    """
    Construct scNODE model.
    :param n_genes (int): Number of genes.
    :param latent_dim (int): Latent diemension.
    :param enc_latent_list (None or list): VAE encoder hidden layer size. Either None indicates no hidden layers or a
                                           list of integers representing size of every hidden layers.
    :param dec_latent_list (None or list): VAE decoder hidden layer size. Either None indicates no hidden layers or a
                                           list of integers representing size of every hidden layers.
    :param drift_latent_size (None or list): ODE solver drift network hidden layer size. Either None indicates no hidden
                                             layers or a list of integers representing size of every hidden layers.
    :param latent_enc_act (str): Activation function for VAE encoder.
    :param latent_dec_act (str): Activation function for VAE decoder.
    :param drift_act (str): Activation function for ODE solver drift network.
    :param ode_method (str): ODE solver method. Default as "euler". See torchdiffeq documentation for more details.
    :return: (torch.Model) scNODE model.
    """
    latent_encoder = LinearVAENet(
        input_dim=n_genes,
        latent_size_list=enc_latent_list,
        output_dim=latent_dim,
        act_name=latent_enc_act,
    )  # VAE encoder
    obs_decoder = LinearNet(
        input_dim=latent_dim,
        latent_size_list=dec_latent_list,
        output_dim=n_genes,
        act_name=latent_dec_act,
    )  # VAE decoder
    diffeq_drift_net = LinearNet(
        input_dim=latent_dim,
        latent_size_list=drift_latent_size,
        output_dim=latent_dim,
        act_name=drift_act,
    )  # drift network
    diffeq_decoder = ODE(
        input_dim=latent_dim, drift_net=diffeq_drift_net, ode_method=ode_method
    )  # ODE solver
    latent_ode_model = scNODE(
        input_dim=n_genes,
        latent_dim=latent_dim,
        output_dim=n_genes,
        latent_encoder=latent_encoder,
        diffeq_decoder=diffeq_decoder,
        obs_decoder=obs_decoder,
    )
    return latent_ode_model


# =============================================
def add_to_dir(args, pretrain_only):
    dir = ""
    if not pretrain_only:
        dir += f"/freeze_enc_dec" if args.freeze_enc_dec else ""
        if args.adjusted_full_train:
            dir += f"/adjusted_full_train"
            dir += f"/full_train_kl_coeff_{args.full_train_kl_coeff}"
        if args.finetune_lr != 1e-3 or args.lr != 1e-3:
            dir += f"/finetune_lr_{args.finetune_lr}_lr_{args.lr}"
        dir += f"/beta_{args.beta}" if args.beta != 1.0 else ""
        if args.vel_reg:
            dir += "/vel_reg"
    return dir


def get_checkpoint_train_path(
    use_continuous,
    use_normalized,
    cell_type,
    data_name,
    use_hvgs,
    split_type,
    kl_coeff,
    pretrain_only,
    freeze_enc_dec,
    args=None,
):
    dir = f'./checkpoints{"/continuous" if use_continuous else ""}{"/normalized" if use_normalized else ""}{f"/cell_type_{cell_type}" if cell_type != "" else ""}'
    dir += f"/kl_coeff_{kl_coeff}" if kl_coeff != 0.0 else ""
    dir += add_to_dir(args, pretrain_only)
    return dir, (
        f"{dir}/{data_name}_{'full_train' if not pretrain_only else 'pretrain'}_split_type_{split_type}_use_hvgs_{use_hvgs}.pth"
    )


def scNODETrainWithPreTrain(
    train_data,
    train_tps,
    latent_ode_model,
    latent_coeff,
    epochs,
    iters,
    batch_size,
    lr,
    pretrain_iters=200,
    pretrain_lr=1e-3,
    kl_coeff=0.0,
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    device=torch.device("cpu"),
    data_name="Dataset.HERRING_GABA",
    visualize_pretrain=False,
    split_type="three_interpolation",
    use_hvgs=False,
    use_continuous=False,
    use_normalized=False,
    cell_type="",
    freeze_enc_dec=False,
    args=None,
):
    """
    Train scNODE model.
    :param train_data (list of torch.FloatTensor): Expression matrices at all training timepoints.
    :param train_tps (torch.FloatTensor): A list of training timepoints indices.
    :param latent_ode_model (torch.Model): scNODE model.
    :param latent_coeff (float): Regularization coefficient (beta).
    :param epochs (int): Training epochs.
    :param iters (int): Number of iterations in each epoch.
    :param batch_size (int): Batch size.
    :param lr (float): Learning rate. We recommend using a small learning rate, e.g., 1e-3.
    :param pretrain_iters (int): Number of pre-training iterations.
    :param pretrain_lr (float): Pre-training learning rate. We recommend using a small learning rate, e.g., 1e-3.
    :return:
        (torch.Model) Trained scNODE model.
        (list) Training Loss at each iteration.
        (torch.FloatTensor): Reconstructed expression at training timepoints.
        (torch.dist.Normal): VAE latent distribution.
        (torch.FloatTensor): Latent variables at training timepoints.
    """
    # Pre-training the VAE component
    latent_encoder = latent_ode_model.latent_encoder
    obs_decoder = latent_ode_model.obs_decoder

    # make a single training dataset -- this is the X
    all_train_data = torch.cat(train_data, dim=0).to(device)
    all_train_tps = np.concatenate(
        [np.repeat(t, train_data[i].shape[0]) for i, t in enumerate(train_tps)]
    )

    dir, checkpoint_train_path = get_checkpoint_train_path(
        use_continuous=use_continuous,
        use_normalized=use_normalized,
        cell_type=cell_type,
        data_name=data_name,
        use_hvgs=use_hvgs,
        split_type=split_type,
        kl_coeff=kl_coeff,
        pretrain_only=False,
        freeze_enc_dec=freeze_enc_dec,
        args=args,
    )
    os.makedirs(dir, exist_ok=True)

    if os.path.exists(checkpoint_train_path):
        latent_ode_model.load_state_dict(torch.load(checkpoint_train_path))
        # latent_ODE model prediction
        latent_ode_model.eval()
        # get the reconstruction, first (time step 0) latent distribution and latent sequence
        recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(
            train_data, train_tps, batch_size=None
        )
        # avoid the loss list this time
        return latent_ode_model, None, recon_obs, first_latent_dist, latent_seq

    run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = f"./logs/scNODE_runs/{data_name}/continuous_{use_continuous}/normalized_{use_normalized}/split_type_{split_type}"
    log_dir += f"/kl_coeff_{kl_coeff}" if kl_coeff != 0.0 else ""
    log_dir += add_to_dir(args, False)
    log_dir = os.path.join(
        log_dir,
        run_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"Running checkpoint training for: {checkpoint_train_path}")

    _, checkpoint_pretrain_path = get_checkpoint_train_path(
        use_continuous=use_continuous,
        use_normalized=use_normalized,
        cell_type=cell_type,
        data_name=data_name,
        use_hvgs=use_hvgs,
        split_type=split_type,
        kl_coeff=kl_coeff,
        pretrain_only=True,
        freeze_enc_dec=freeze_enc_dec,
        args=args,
    )

    if os.path.exists(checkpoint_pretrain_path):
        latent_ode_model.load_state_dict(torch.load(checkpoint_pretrain_path))

    # see if we want to first pre-train
    elif pretrain_iters > 0:
        dim_reduction_params = itertools.chain(
            *[latent_encoder.parameters(), obs_decoder.parameters()]
        )

        # only optimize the dimensionatliy reduction parameters, makes sense!
        dim_reduction_optimizer = torch.optim.Adam(
            params=dim_reduction_params, lr=pretrain_lr, betas=(0.95, 0.99)
        )

        dim_reduction_pbar = tqdm(range(pretrain_iters), desc="[ Pre-Training ]")
        latent_encoder.train()
        obs_decoder.train()

        # for each step
        for t in dim_reduction_pbar:
            dim_reduction_optimizer.zero_grad()
            # first we use the encoder to get the latent space
            latent_mu, latent_std = latent_encoder(all_train_data)
            latent_sample = sampleGaussian(latent_mu, latent_std, device)

            # then we try to reconstruct the observations
            recon_obs = obs_decoder(latent_sample)

            # MSE reconstruction loss
            dim_reduction_loss = MSELoss(all_train_data, recon_obs)

            # KL div between latent dist and N(0, 1)
            kl_div = (
                latent_std**2 + latent_mu**2 - 2 * torch.log(latent_std + 1e-5)
            ).mean()
            kl_div = kl_div * kl_coeff
            vae_loss = dim_reduction_loss + kl_div
            # ** kl divergence is set to 0 for some reason...**

            # Backward
            dim_reduction_pbar.set_postfix({"Loss": "{:.3f}".format(vae_loss)})
            vae_loss.backward()
            dim_reduction_optimizer.step()

            writer.add_scalar("Pretrain-Loss/KL", kl_div.item(), t)
            writer.add_scalar("Pretrain-Loss/NLL", dim_reduction_loss.item(), t)
            writer.add_scalar("Pretrain-Loss/VAE", vae_loss.item(), t)

        torch.save(latent_ode_model.state_dict(), checkpoint_pretrain_path)

    print(f"Latent ODE model is ready for visualization...")

    #####################################
    # VAE reconstruction visualization -- if they match it's a good reconstruction! Else, it's pretty shit
    # Pre-training the VAE component
    if visualize_pretrain:
        latent_encoder.eval()
        obs_decoder.eval()

        latent_mu, latent_std = latent_encoder(all_train_data)
        latent_sample = sampleGaussian(latent_mu, latent_std, device)
        recon_obs = obs_decoder(latent_sample)
        from plotting.PlottingUtils import umapWithPCA
        from plotting.visualization import plotPredAllTime

        true_umap, umap_model, pca_model = umapWithPCA(
            all_train_data.detach().numpy(), n_neighbors=50, min_dist=0.1, pca_pcs=50
        )
        pred_umap = umap_model.transform(
            pca_model.transform(recon_obs.detach().numpy())
        )
        plotPredAllTime(
            true_umap,
            pred_umap,
            all_train_tps,
            all_train_tps,
            fig_name=f"{data_name}_pretrain.png",
        )

        print(f"Finish pretraining VAE...")
        exit()
    #####################################
    # Train the entire model
    blur = 0.05
    scaling = 0.5
    loss_list = []

    neural_ode_params = latent_ode_model.diffeq_decoder.parameters()
    if freeze_enc_dec:
        optimizer = torch.optim.Adam(
            params=neural_ode_params,
            lr=lr,
            betas=(0.95, 0.99),
        )
    else:
        dim_reduction_params = itertools.chain(
            *[latent_encoder.parameters(), obs_decoder.parameters()]
        )
        optimizer = torch.optim.Adam(
            [
                {
                    "params": neural_ode_params,
                    "lr": args.lr,
                },
                {
                    "params": dim_reduction_params,
                    "lr": args.finetune_lr,
                },
            ],
            betas=(0.95, 0.99),
        )

    latent_ode_model.train()

    for e in range(epochs):
        epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
        for t in epoch_pbar:
            # first we set the optimizer to reset the gradient
            optimizer.zero_grad()

            # then we recreate the first observation, which is simply the training time points
            # BUT it's in time points that's the important part
            (
                recon_obs,
                first_latent_dist,
                first_time_true_batch,
                latent_seq,
            ) = latent_ode_model(train_data, train_tps, batch_size=batch_size)

            # recall: train data is cells x genes by for a specific time point
            # so this one here will take each cells x genes for a specific time point
            # and get their encoded
            encoder_latent_seq = [
                latent_ode_model.vaeReconstruct(
                    [  # okay this is kinda silly now I realize, we're giving it a list of size 1 of a single batch
                        each[
                            np.random.choice(
                                np.arange(each.shape[0]),
                                size=batch_size,
                                replace=(each.shape[0] < batch_size),
                            ),
                            :,
                        ]
                    ]  # chooses a random batch of the cells
                )[0][
                    0
                ]  # this means the latent encoding's first batch, i.e. the only batch we gave it
                for each in train_data
            ]

            # -----
            # OT loss between true and reconstructed cell sets at each time point
            # Note: we compare the predicted batch with 200 randomly picked true cells, in order to save computational
            #       time. With sufficient number of training iterations, all true cells can be used.
            ot_loss = SinkhornLoss(
                train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200
            )
            # Dynamic regularization: Difference between encoder latent and DE latent
            dynamic_reg = SinkhornLoss(
                encoder_latent_seq,
                latent_seq,
                blur=blur,
                scaling=scaling,
                batch_size=None,
            )

            loss = ot_loss + latent_coeff * dynamic_reg

            # add an extra KL loss term
            if args.adjusted_full_train:
                normal_dist = dist.Normal(
                    torch.zeros_like(first_latent_dist.mean),
                    torch.ones_like(first_latent_dist.stddev),
                )
                kl_loss = dist.kl_divergence(first_latent_dist, normal_dist).sum()
                loss += args.full_train_kl_coeff * kl_loss

                epoch_pbar.set_postfix(
                    {
                        "Loss": "{:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}, KL_Reg={:.3f}".format(
                            loss, ot_loss, dynamic_reg, kl_loss
                        )
                    }
                )

                logging.debug(
                    "Step: {} | Loss: {:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}, KL_Reg={:.3f}".format(
                        e * iters + t, loss, ot_loss, dynamic_reg, kl_loss
                    )
                )
            else:
                epoch_pbar.set_postfix(
                    {
                        "Loss": "{:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}".format(
                            loss, ot_loss, dynamic_reg
                        )
                    }
                )

                logging.debug(
                    "Step: {} | Loss: {:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}".format(
                        e * iters + t, loss, ot_loss, dynamic_reg
                    )
                )

            if args.vel_reg:
                # if we will regularize the neural ODE's velocity:
                vel_reg_loss = latent_ode_model.diffeq_decoder.net.regularization_loss
                loss += vel_reg_loss
                epoch_pbar.set_postfix(
                    {
                        "Loss": "{:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}, Vel_Reg={:.3f}".format(
                            loss, ot_loss, dynamic_reg, vel_reg_loss
                        )
                    }
                )

                logging.debug(
                    "Step: {} | Loss: {:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}, Vel_Reg={:.3f}".format(
                        e * iters + t, loss, ot_loss, dynamic_reg, vel_reg_loss
                    )
                )
                writer.add_scalar("Loss/Vel_Reg", vel_reg_loss.item(), e * iters + t)

            writer.add_scalar("Loss/OT", ot_loss.item(), e * iters + t)
            writer.add_scalar("Loss/Dynamic_Reg", dynamic_reg.item(), e * iters + t)
            if args.adjusted_full_train:
                writer.add_scalar("Loss/KL_Reg", kl_loss.item(), e * iters + t)
            writer.add_scalar("Loss", loss.item(), e * iters + t)

            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), dynamic_reg.item()])

    torch.save(latent_ode_model.state_dict(), checkpoint_train_path)

    # latent_ODE model prediction
    latent_ode_model.eval()
    # get the reconstruction, first (time step 0) latent distribution and latent sequence
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(
        train_data, train_tps, batch_size=None
    )
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq


def scNODEPredict(latent_ode_model, first_tp_data, tps, n_cells):
    """
    scNODE predicts expressions.
    :param latent_ode_model (torch.Model): scNODE model.
    :param first_tp_data (torch.FloatTensor): Expression at the first timepoint.
    :param tps (torch.FloatTensor): A list of timepoints to predict.
    :param n_cells (int): The number of cells to predict at each timepoint.
    :param batch_size (None or int): Either None indicates predicting in a whole or an integer representing predicting
                                     batch-wise to save computational costs. Default as None.
    :return: (torch.FloatTensor) Predicted expression with the shape of (# cells, # tps, # genes).
    """
    latent_ode_model.eval()
    _, _, all_pred_data = latent_ode_model.predict(first_tp_data, tps, n_cells=n_cells)
    all_pred_data = all_pred_data.detach().numpy()  # (# cells, # tps, # genes)
    return all_pred_data
