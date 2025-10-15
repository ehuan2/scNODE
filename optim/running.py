'''
Description:
    Main codes for scNODE and Dummy model training and prediction.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import logging
from tqdm import tqdm
import numpy as np
import itertools
from datetime import datetime
import os

from model.layer import LinearNet, LinearVAENet
from model.diff_solver import ODE
from model.dynamic_model import scNODE
from optim.loss_func import SinkhornLoss, MSELoss
from benchmark.BenchmarkUtils import sampleGaussian

from torch.utils.tensorboard import SummaryWriter

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
    ode_method="euler"
):
    '''
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
    '''
    latent_encoder = LinearVAENet(
        input_dim=n_genes, latent_size_list=enc_latent_list, output_dim=latent_dim, act_name=latent_enc_act
    )  # VAE encoder
    obs_decoder = LinearNet(
        input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=n_genes, act_name=latent_dec_act
    )  # VAE decoder
    diffeq_drift_net = LinearNet(
        input_dim=latent_dim, latent_size_list=drift_latent_size, output_dim=latent_dim, act_name=drift_act
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
        obs_decoder=obs_decoder
    )
    return latent_ode_model

# =============================================

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
    device=torch.device('cpu'),
    data_name="Dataset.HERRING_GABA",
    visualize_pretrain=False,
    split_type='three_interpolation',
    use_hvgs=False,
    use_continuous=False
):
    '''
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
    '''
    # Pre-training the VAE component
    latent_encoder = latent_ode_model.latent_encoder
    obs_decoder = latent_ode_model.obs_decoder

    # make a single training dataset -- this is the X
    all_train_data = torch.cat(train_data, dim=0).to(device)
    all_train_tps = np.concatenate([np.repeat(t, train_data[i].shape[0]) for i, t in enumerate(train_tps)])

    if use_continuous:
        os.makedirs(f'./checkpoints/continuous/', exist_ok=True)

    checkpoint_train_path = (
        f'./checkpoints/{data_name}_full_train_split_type_{split_type}_use_hvgs_{use_hvgs}.pth'
        if not use_continuous else
        f'./checkpoints/continuous/{data_name}_full_train_split_type_{split_type}_use_hvgs_{use_hvgs}.pth'
    )
    if os.path.exists(checkpoint_train_path):
        latent_ode_model.load_state_dict(torch.load(checkpoint_train_path))
        # latent_ODE model prediction
        latent_ode_model.eval()
        # get the reconstruction, first (time step 0) latent distribution and latent sequence
        recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, batch_size=None)
        # avoid the loss list this time
        return latent_ode_model, None, recon_obs, first_latent_dist, latent_seq

    run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(f"./logs/scNODE_runs/{data_name}/continuous_{use_continuous}/split_type_{split_type}/", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    checkpoint_pretrain_path = (
        f'./checkpoints/{data_name}_pretrain_split_type_{split_type}_use_hvgs_{use_hvgs}.pth'
        if not use_continuous else
        f'./checkpoints/continuous/{data_name}_pretrain_split_type_{split_type}_use_hvgs_{use_hvgs}.pth'
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
            params=dim_reduction_params,
            lr=pretrain_lr,
            betas=(0.95, 0.99)
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
            kl_div = (latent_std**2 + latent_mu**2 - 2*torch.log(latent_std + 1e-5)).mean()
            kl_div = kl_div * kl_coeff
            vae_loss = dim_reduction_loss + kl_div
            # ** kl divergence is set to 0 for some reason...**

            # Backward
            dim_reduction_pbar.set_postfix({"Loss": "{:.3f}".format(vae_loss)})
            vae_loss.backward()
            dim_reduction_optimizer.step()

            writer.add_scalar('Pretrain-Loss/KL', kl_div.item(), t)
            writer.add_scalar('Pretrain-Loss/NLL', dim_reduction_loss.item(), t)
            writer.add_scalar('Pretrain-Loss/VAE', vae_loss.item(), t)
        
        torch.save(latent_ode_model.state_dict(), checkpoint_pretrain_path)

    print(f'Latent ODE model is ready for visualization...')

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
        true_umap, umap_model, pca_model = umapWithPCA(all_train_data.detach().numpy(), n_neighbors=50, min_dist=0.1, pca_pcs=50)
        pred_umap = umap_model.transform(pca_model.transform(recon_obs.detach().numpy()))
        plotPredAllTime(true_umap, pred_umap, all_train_tps, all_train_tps, fig_name=f'{data_name}_pretrain.png')

        print(f'Finish pretraining VAE...')
        exit()
    #####################################
    # Train the entire model
    blur = 0.05
    scaling = 0.5
    loss_list = []
    optimizer = torch.optim.Adam(params=latent_ode_model.parameters(), lr=lr, betas=(0.95, 0.99))
    latent_ode_model.train()

    for e in range(epochs):
        epoch_pbar = tqdm(range(iters), desc="[ Epoch {} ]".format(e + 1))
        for t in epoch_pbar:
            # first we set the optimizer to reset the gradient
            optimizer.zero_grad()

            # then we recreate the first observation, which is simply the training time points
            # BUT it's in time points that's the important part
            recon_obs, first_latent_dist, first_time_true_batch, latent_seq = latent_ode_model(
                train_data,
                train_tps,
                batch_size=batch_size
            )

            # recall: train data is cells x genes by for a specific time point
            # so this one here will take each cells x genes for a specific time point
            # and get their encoded
            encoder_latent_seq = [
                latent_ode_model.vaeReconstruct(
                    [ # okay this is kinda silly now I realize, we're giving it a list of size 1 of a single batch
                        each[
                            np.random.choice(
                            np.arange(each.shape[0]),
                            size=batch_size,
                            replace=(each.shape[0] < batch_size)), : ]
                    ] # chooses a random batch of the cells
                )[0][0] # this means the latent encoding's first batch, i.e. the only batch we gave it
                for each in train_data
            ]

            # -----
            # OT loss between true and reconstructed cell sets at each time point
            # Note: we compare the predicted batch with 200 randomly picked true cells, in order to save computational
            #       time. With sufficient number of training iterations, all true cells can be used.
            ot_loss = SinkhornLoss(train_data, recon_obs, blur=blur, scaling=scaling, batch_size=200)
            # Dynamic regularization: Difference between encoder latent and DE latent
            dynamic_reg = SinkhornLoss(encoder_latent_seq, latent_seq, blur=blur, scaling=scaling, batch_size=None)

            loss = ot_loss + latent_coeff * dynamic_reg

            epoch_pbar.set_postfix(
                {"Loss": "{:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}".format(loss, ot_loss, dynamic_reg)})

            logging.debug("Step: {} | Loss: {:.3f} | OT={:.3f}, Dynamic_Reg={:.3f}".format(e * iters + t, loss, ot_loss, dynamic_reg))

            writer.add_scalar('Loss/OT', ot_loss.item(), e * iters + t)
            writer.add_scalar('Loss/Dynamic_Reg', dynamic_reg.item(), e * iters + t)
            writer.add_scalar('Loss', loss.item(), e * iters + t)

            loss.backward()
            optimizer.step()
            loss_list.append([loss.item(), ot_loss.item(), dynamic_reg.item()])

    torch.save(latent_ode_model.state_dict(), checkpoint_train_path)

    # latent_ODE model prediction
    latent_ode_model.eval()
    # get the reconstruction, first (time step 0) latent distribution and latent sequence
    recon_obs, first_latent_dist, _, latent_seq = latent_ode_model(train_data, train_tps, batch_size=None)
    return latent_ode_model, loss_list, recon_obs, first_latent_dist, latent_seq


def scNODEPredict(latent_ode_model, first_tp_data, tps, n_cells):
    '''
    scNODE predicts expressions.
    :param latent_ode_model (torch.Model): scNODE model.
    :param first_tp_data (torch.FloatTensor): Expression at the first timepoint.
    :param tps (torch.FloatTensor): A list of timepoints to predict.
    :param n_cells (int): The number of cells to predict at each timepoint.
    :param batch_size (None or int): Either None indicates predicting in a whole or an integer representing predicting
                                     batch-wise to save computational costs. Default as None.
    :return: (torch.FloatTensor) Predicted expression with the shape of (# cells, # tps, # genes).
    '''
    latent_ode_model.eval()
    _, _, all_pred_data = latent_ode_model.predict(first_tp_data, tps, n_cells=n_cells)
    all_pred_data = all_pred_data.detach().numpy()  # (# cells, # tps, # genes)
    return all_pred_data

