"""
benchmark_cell_trajectory.py

Measures the trajectory of each cell that we simulate,
and then compares whether this trajectory makes sense.

This is to evaluate NODE and its components. Mainly through:
1. Plotting the trajectory of each cell type over time, and seeing if the trajectories
are expected visually.
2. Measuring some general metrics for this trajectory.

TODO:
    3. Measuring the differentially expressed genes over time for each cell type, and comparing
    them to the true DE genes.
    4. Measuring everything compared to cell ontology
"""
import os

import torch
import numpy as np
from pprint import pprint
import gseapy as gp

from benchmark_cell_types import (
    load_model,
    tps_to_continuous,
    prep_traj_data,
    get_cell_pred_embeds_joint,
    get_cell_pred_embeds_sequential,
    get_cell_embed_by_timepoint,
    infer_cell_types_ot,
    infer_cell_types_knn,
    add_args_to_parser,
    soft_labels_to_cell_types,
)

from benchmark.BenchmarkUtils import (
    loadSCData,
    create_parser,
)
import pandas as pd
from plot_sankey import plot_sankey_from_labels
import matplotlib.pyplot as plt

from optim.running import add_to_dir
from scipy.stats import spearmanr


def create_traj_dir(args):
    fig_dir = "./figs/trajectories/"
    fig_dir += f"/kl_coeff_{args.kl_coeff}" if args.kl_coeff != 0.0 else ""
    fig_dir += add_to_dir(args, args.pretrain_only)

    fig_dir += "/knn" if args.use_knn else "/ot"

    if not args.use_knn:
        fig_dir += "_unbalanced" if args.unbalanced_ot else "_balanced"
        fig_dir += (
            f"_scaling_{args.unbalanced_ot_scaling}" if args.unbalanced_ot else ""
        )
        fig_dir += f"_reach_{args.unbalanced_ot_reach}" if args.unbalanced_ot else ""
        fig_dir += f"_blur_{args.unbalanced_ot_blur}" if args.unbalanced_ot else ""

    fig_dir += "/seq" if args.use_sequential_pred else "/joint"

    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def create_trajectory(inferred_cell_types):
    """
    Create a list of lists, where each list corresponds to the trajectory of a cell
    """
    n_tps = len(inferred_cell_types)
    trajectories = np.full(
        (inferred_cell_types[0]["labels"].shape[0], n_tps), "", dtype=object
    )

    for t in range(n_tps):
        pred_cell_types = soft_labels_to_cell_types(inferred_cell_types[t])

        for cell in range(pred_cell_types.shape[0]):
            trajectories[cell, t] = pred_cell_types[cell]

    return trajectories


def plot_trajectory(trajectories, times_sorted, cell_type, args):
    traj_dict = {tp: [] for tp in times_sorted}
    n_tps = len(times_sorted)

    for t in range(n_tps):
        for cell in range(trajectories.shape[0]):
            traj_dict[times_sorted[t]].append(trajectories[cell, t])

    # plot the trajectories that we care about
    trajectories = pd.DataFrame(traj_dict)

    path_dir = os.path.join(create_traj_dir(args), f"{cell_type}")
    os.makedirs(path_dir, exist_ok=True)
    plot_sankey_from_labels(
        trajectories,
        title="Inferred Cell Type Trajectories",
        path=os.path.join(path_dir, "sankey.html"),
    )


def plot_trajectory_per_cell_type(trajectories, times_sorted, args):
    for cell_type in set(trajectories.flatten()):
        # filter trajectories for this cell type
        filtered_trajectories = trajectories[trajectories[:, 0] == cell_type]
        plot_trajectory(filtered_trajectories, times_sorted, cell_type, args)


def plot_switch_rate(trajectories, args):
    """
    Given the trajectories, plot the switch rate of the cells.
    """
    n_cells, n_tps = trajectories.shape

    # iterate over the time points and count switches
    switch_counts = np.zeros(n_tps - 1)
    for t in range(n_tps - 1):
        for cell in range(n_cells):
            if trajectories[cell, t] != trajectories[cell, t + 1]:
                switch_counts[t] += 1
        switch_counts[t] /= n_cells  # normalize

    plt.figure(figsize=(10, 6))
    plt.plot(range(n_tps - 1), switch_counts, marker="o", color="purple")
    plt.xlabel("Time Point")
    plt.ylabel("Switch Rate")
    plt.title("Cell Type Switch Rate over Time")
    plt.grid()
    plt.savefig(os.path.join(create_traj_dir(args), "switch_rate.png"))
    plt.close()


def plot_entropy_over_time(trajectories, args):
    """
    Given the trajectories, plot the entropy of cell type distribution over time.
    """
    n_cells, n_tps = trajectories.shape

    # as a first step, we need to get the one-hot encoding of cell types
    cell_types = sorted(np.unique(trajectories).tolist())

    def one_hot_mapping(cell_type):
        one_hot = np.zeros(len(cell_types))
        one_hot[cell_types.index(cell_type)] = 1
        return one_hot

    def get_entropy_of_cell(cell_traj):
        """
        For every single cell, compute the entropy of its cell type distribution
        over time.
        """
        cell_label_probs = np.zeros(len(cell_types))
        for t in range(n_tps):
            cell_label_probs += one_hot_mapping(cell_traj[t])
        cell_label_probs /= n_tps  # normalize to get probabilities
        entropy = -np.sum([p * np.log2(p) for p in cell_label_probs if p > 0])
        entropy /= np.log2(len(cell_types))  # normalize to [0, 1]
        return entropy

    entropies = []
    for cell in range(n_cells):
        cell_traj = trajectories[cell, :]
        entropy = get_entropy_of_cell(cell_traj)
        entropies.append(entropy)

    # now let's do a histogram of entropies
    plt.figure(figsize=(10, 6))
    plt.hist(entropies, bins=30, color="teal", alpha=0.7)
    plt.xlabel("Entropy")
    plt.ylabel("Number of Cells")
    plt.title("Distribution of Cell Type Entropy over Time")
    plt.grid()
    plt.savefig(os.path.join(create_traj_dir(args), "entropy_distribution.png"))
    plt.close()


def reconstruct_gene_expression_from_embeddings(embeddings, model):
    """
    Given the embeddings and the model, reconstruct the gene expression profiles.
    """
    reconstructed = []
    for t in range(len(embeddings)):
        recon_t = model.obs_decoder(torch.FloatTensor(embeddings[t])).detach().numpy()
        reconstructed.append(recon_t)
    return reconstructed


def pseudo_bulk_by_cell_type(reconstructed, cell_types):
    """
    Given the reconstructed gene expression profiles and the cell types,
    pseudo-bulk the gene expression profiles by cell type at each time point.
    """
    n_tps = len(reconstructed)
    pseudo_bulk = {}

    for t in range(n_tps):
        pseudo_bulk[t] = {}
        unique_cell_types = set(cell_types[t])
        for cell_type in unique_cell_types:
            # get indices of cells of this type, 0 because np.where returns a tuple
            cell_indices = np.where(cell_types[t] == cell_type)[0]
            pseudo_bulk[t][cell_type] = np.mean(reconstructed[t][cell_indices], axis=0)

    return pseudo_bulk


def create_time_aggregate_pseudo_bulk(pseudo_bulk):
    """
    Given the pseudo-bulked gene expression profiles, the cell types,
    create a pseudo-bulk profile for a specific cell type across all time points.

    i.e. we will be aggregating the pseudo-bulk profiles for this cell type
    across all time points.
    """
    cell_type_pseudo_bulk = {}
    time_points = sorted(pseudo_bulk.keys())

    cell_types = set()
    for tp in time_points:
        cell_types.update(pseudo_bulk[tp].keys())

    for cell_type in cell_types:
        samples = []
        for tp in time_points:
            if cell_type in pseudo_bulk[tp]:
                samples.append(pseudo_bulk[tp][cell_type])

        if samples:
            cell_type_pseudo_bulk[cell_type] = np.mean(samples, axis=0)

    return cell_type_pseudo_bulk


def create_celltype_aggregate_pseudo_bulk(pseudo_bulk):
    """
    Given the pseudo-bulked gene expression profiles, create a pseudo-bulk profile
    for each time point across all cell types.

    i.e. we will be aggregating the pseudo-bulk profiles for all cell types
    at each time point.
    """
    timepoint_pseudo_bulk = {}
    time_points = sorted(pseudo_bulk.keys())

    for tp in time_points:
        samples = []
        for cell_type in pseudo_bulk[tp]:
            samples.append(pseudo_bulk[tp][cell_type])

        if samples:
            timepoint_pseudo_bulk[tp] = np.mean(samples, axis=0)

    return timepoint_pseudo_bulk


def mse_between_pseudo_bulks(pseudo_bulk1, pseudo_bulk2):
    """
    Given two pseudo-bulked gene expression profiles, compute the MSE between them.
    """
    return np.mean((pseudo_bulk1 - pseudo_bulk2) ** 2)


def time_de_analysis(pseudo_bulk, cell_type):
    """
    Given the pseudo-bulked gene expression profiles, perform differential expression
    analysis over time for a given cell type.
    """

    # keep track of a ranked list of DE genes
    de_genes = {}
    time_points = sorted(pseudo_bulk.keys())
    n_tps = len(time_points)

    for i in range(n_tps - 1):
        tp1 = time_points[i]
        tp2 = time_points[i + 1]

        expr1, expr2 = None, None
        if cell_type is None:
            expr1 = pseudo_bulk[tp1]
            expr2 = pseudo_bulk[tp2]
        elif cell_type in pseudo_bulk[tp1] and cell_type in pseudo_bulk[tp2]:
            expr1 = pseudo_bulk[tp1][cell_type]
            expr2 = pseudo_bulk[tp2][cell_type]

        if expr1 is None or expr2 is None:
            continue

        # log fold change, future timepoint / past timepoint, sorted descending
        de_genes_tp = np.log2((expr2 + 1e-8) / (expr1 + 1e-8))
        de_genes[(tp1, tp2)] = de_genes_tp

    return de_genes


def plot_time_de_all(true_de_set, pred_de_set):
    """
    Plots the number of overlapping DE genes, and Spearman correlation, but aggregated over all cell types.
    """
    # Collect metrics for all cell types
    metrics_data = {
        "time_pair": [],
        "overlap": [],
        "spearman_corr": [],
    }

    for tp_pair in true_de_set:
        if tp_pair in pred_de_set:
            # compute overlap in top 100 DE genes
            true_de_genes_tp = np.argsort(
                -true_de_set[tp_pair]
            )  # descending order, store their indices
            pred_de_genes_tp = np.argsort(
                -pred_de_set[tp_pair]
            )  # descending order, store their indices
            true_top_genes = set(true_de_genes_tp[:100])
            pred_top_genes = set(pred_de_genes_tp[:100])
            overlap = true_top_genes.intersection(pred_top_genes)
            print(
                f"Overall Time DE Analysis, Time points {tp_pair}: Overlap in top 100 DE genes: {len(overlap)}"
            )

            # finally, let's calculate the Spearman correlation
            spearman = spearmanr(true_de_set[tp_pair], pred_de_set[tp_pair])
            print(
                f"Overall Time DE Analysis, Time points {tp_pair}: Spearman correlation: {spearman.statistic}"
            )
            metrics_data["time_pair"].append(f"{tp_pair[0]}->{tp_pair[1]}")
            metrics_data["overlap"].append(len(overlap))
            metrics_data["spearman_corr"].append(spearman.statistic)

    # now let's plot it all:
    metrics_df = pd.DataFrame(metrics_data)

    fig_dir = "./figs/differential_expression/"
    os.makedirs(fig_dir, exist_ok=True)

    fig_overlap, ax_overlap = plt.subplots(figsize=(20, 16))
    fig_corr, ax_corr = plt.subplots(figsize=(20, 16))

    time_pairs = metrics_df["time_pair"].values
    x_pos = range(len(time_pairs))

    # Plot overlap as line graph
    ax_overlap.plot(
        x_pos,
        metrics_df["overlap"].values,
        marker="o",
        color="steelblue",
        linewidth=2,
        markersize=6,
    )
    ax_overlap.set_title(f"All Cell Types Overlap in Top 100 DE Genes", fontsize=10)
    ax_overlap.set_xlabel("Time Transition")
    ax_overlap.set_ylabel("Overlap Count")
    ax_overlap.set_xticks(x_pos)
    ax_overlap.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
    ax_overlap.set_ylim(0, 100)
    ax_overlap.grid(True, alpha=0.3)

    # Plot Spearman correlation as line graph
    ax_corr.plot(
        x_pos,
        metrics_df["spearman_corr"].values,
        marker="s",
        color="coral",
        linewidth=2,
        markersize=6,
    )
    ax_corr.set_title(f"All Cell Types Spearman Correlation", fontsize=10)
    ax_corr.set_xlabel("Time Transition")
    ax_corr.set_ylabel("Spearman r")
    ax_corr.set_xticks(x_pos)
    ax_corr.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
    ax_corr.set_ylim(-1, 1)
    ax_corr.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax_corr.grid(True, alpha=0.3)

    plt.figure(fig_overlap.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "time_de_overlap_all.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Saved overlap plot to {fig_dir}")
    plt.close(fig_overlap)

    plt.figure(fig_corr.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "time_de_spearman_all.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Saved Spearman correlation plot to {fig_dir}")
    plt.close(fig_corr)


def plot_time_de(true_de_set, pred_de_set, cell_types):
    """
    Plots the number of overlapping DE genes, and Spearman correlation between true and predicted sets over time for each cell type.
    """

    # Collect metrics for all cell types
    metrics_data = {
        "cell_type": [],
        "time_pair": [],
        "overlap": [],
        "spearman_corr": [],
        "accuracy": [],
    }

    for cell_type in cell_types:
        for tp_pair in true_de_set[cell_type]:
            if tp_pair in pred_de_set[cell_type]:
                # compute overlap in top 100 DE genes
                true_de_genes_tp = np.argsort(
                    -true_de_set[cell_type][tp_pair]
                )  # descending order, store their indices
                pred_de_genes_tp = np.argsort(
                    -pred_de_set[cell_type][tp_pair]
                )  # descending order, store their indices
                true_top_genes = set(true_de_genes_tp[:100])
                pred_top_genes = set(pred_de_genes_tp[:100])
                overlap = true_top_genes.intersection(pred_top_genes)
                print(
                    f"Cell type {cell_type}, Time points {tp_pair}: Overlap in top 100 DE genes: {len(overlap)}"
                )

                # finally, let's calculate the Spearman correlation between the log fold changes
                spearman = spearmanr(
                    true_de_set[cell_type][tp_pair], pred_de_set[cell_type][tp_pair]
                )
                print(true_de_set[cell_type][tp_pair], pred_de_set[cell_type][tp_pair])
                print(
                    f"Cell type {cell_type}, Time points {tp_pair}: Spearman correlation: {spearman.statistic}"
                )

                # let's also add an "accuracy" score, where we count how many of the genes
                # have the same sign in the log fold change, as long as the true absolute value or pred. absolute value > 0.1
                correct_count = 0
                total_count = 0
                for i in range(len(true_de_set[cell_type][tp_pair])):
                    true_val = true_de_set[cell_type][tp_pair][i]
                    pred_val = pred_de_set[cell_type][tp_pair][i]
                    if abs(true_val) > 0.1 or abs(pred_val) > 0.1:
                        total_count += 1
                        if (true_val >= 0 and pred_val >= 0) or (
                            true_val < 0 and pred_val < 0
                        ):
                            correct_count += 1

                accuracy = correct_count / total_count if total_count > 0 else 0.0

                # Store metrics
                metrics_data["cell_type"].append(cell_type)
                metrics_data["time_pair"].append(f"{tp_pair[0]}->{tp_pair[1]}")
                metrics_data["overlap"].append(len(overlap))
                metrics_data["spearman_corr"].append(spearman.statistic)
                metrics_data["accuracy"].append(accuracy)

            else:
                print(
                    f"Cell type {cell_type}, Time points {tp_pair}: No predicted DE genes found."
                )

    # Create two large visualizations with 4x5 grid for each metric type
    if metrics_data["cell_type"]:
        metrics_df = pd.DataFrame(metrics_data)
        unique_cell_types = sorted(metrics_df["cell_type"].unique())
        n_cell_types = len(unique_cell_types)

        fig_dir = "./figs/differential_expression/"
        os.makedirs(fig_dir, exist_ok=True)

        # Create figure for overlap with 4x5 grid
        fig_overlap, axes_overlap = plt.subplots(4, 5, figsize=(20, 16))
        axes_overlap_flat = axes_overlap.flatten()

        fig_accuracy, axes_accuracy = plt.subplots(4, 5, figsize=(20, 16))
        axes_accuracy_flat = axes_accuracy.flatten()

        # Create figure for Spearman correlation with 4x5 grid
        fig_corr, axes_corr = plt.subplots(4, 5, figsize=(20, 16))
        axes_corr_flat = axes_corr.flatten()

        for idx, cell_type in enumerate(unique_cell_types):
            cell_type_data = metrics_df[metrics_df["cell_type"] == cell_type]
            time_pairs = cell_type_data["time_pair"].values
            x_pos = range(len(time_pairs))

            # Plot overlap as line graph
            ax_overlap = axes_overlap_flat[idx]
            ax_overlap.plot(
                x_pos,
                cell_type_data["overlap"].values,
                marker="o",
                color="steelblue",
                linewidth=2,
                markersize=6,
            )
            ax_overlap.set_title(f"{cell_type}", fontsize=10)
            ax_overlap.set_xlabel("Time Transition")
            ax_overlap.set_ylabel("Overlap Count")
            ax_overlap.set_xticks(x_pos)
            ax_overlap.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
            ax_overlap.set_ylim(0, 100)
            ax_overlap.grid(True, alpha=0.3)

            # Plot accuracy as line graph
            ax_accuracy = axes_accuracy_flat[idx]
            ax_accuracy.plot(
                x_pos,
                cell_type_data["accuracy"].values,
                marker="o",
                color="steelblue",
                linewidth=2,
                markersize=6,
            )
            ax_accuracy.set_title(f"{cell_type}", fontsize=10)
            ax_accuracy.set_xlabel("Time Transition")
            ax_accuracy.set_ylabel("Accuracy")
            ax_accuracy.set_xticks(x_pos)
            ax_accuracy.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
            ax_accuracy.set_ylim(0, 1)
            ax_accuracy.grid(True, alpha=0.3)

            # Plot Spearman correlation as line graph
            ax_corr = axes_corr_flat[idx]
            ax_corr.plot(
                x_pos,
                cell_type_data["spearman_corr"].values,
                marker="s",
                color="coral",
                linewidth=2,
                markersize=6,
            )
            ax_corr.set_title(f"{cell_type}", fontsize=10)
            ax_corr.set_xlabel("Time Transition")
            ax_corr.set_ylabel("Spearman r")
            ax_corr.set_xticks(x_pos)
            ax_corr.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
            ax_corr.set_ylim(-1, 1)
            ax_corr.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax_corr.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cell_types, 20):
            axes_accuracy_flat[idx].set_visible(False)
            axes_overlap_flat[idx].set_visible(False)
            axes_corr_flat[idx].set_visible(False)

        # Save overlap figure
        fig_overlap.suptitle("Overlap in Top 100 DE Genes", fontsize=14, y=0.995)
        plt.figure(fig_overlap.number)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, "time_de_overlap.png"), dpi=300, bbox_inches="tight"
        )
        print(f"Saved overlap plot to {fig_dir}")
        plt.close(fig_overlap)

        # Save overlap figure
        fig_accuracy.suptitle(
            "Accuracy in Sign of Log Fold Changes", fontsize=14, y=0.995
        )
        plt.figure(fig_accuracy.number)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, "log_fold_changes.png"), dpi=300, bbox_inches="tight"
        )
        print(f"Saved accuracy plot to {fig_dir}")
        plt.close(fig_accuracy)

        # Save Spearman correlation figure
        fig_corr.suptitle(
            "Spearman Correlation between True and Predicted DE Genes",
            fontsize=14,
            y=0.995,
        )
        plt.figure(fig_corr.number)
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, "time_de_spearman.png"), dpi=300, bbox_inches="tight"
        )
        print(f"Saved Spearman correlation plot to {fig_dir}")
        plt.close(fig_corr)


def gsea_analysis(true_pseudobulk, pred_pseudobulk, gene_names, cell_types):
    """
    Given the true and predicted DE gene sets, perform GSEA analysis.
    """
    time_points = sorted(true_pseudobulk.keys())
    n_tps = len(time_points)

    cell_type_metrics = {}

    fig_dir = "./figs/differential_expression/gsea/"
    os.makedirs(fig_dir, exist_ok=True)

    # Create figure for overlap with 4x5 grid
    fig_overlap, axes_overlap = plt.subplots(4, 5, figsize=(20, 16))
    axes_overlap_flat = axes_overlap.flatten()

    # Create figure for Spearman correlation with 4x5 grid
    fig_corr, axes_corr = plt.subplots(4, 5, figsize=(20, 16))
    axes_corr_flat = axes_corr.flatten()

    for idx, cell_type in enumerate(cell_types):
        metrics = {
            "accuracy": [],
            "spearman_corr": [],
            "time_pairs": [],
        }
        for i in range(n_tps - 1):
            tp1 = time_points[i]
            tp2 = time_points[i + 1]

            if (
                cell_type not in true_pseudobulk[tp1]
                or cell_type not in true_pseudobulk[tp2]
                or cell_type not in pred_pseudobulk[tp1]
                or cell_type not in pred_pseudobulk[tp2]
            ):
                continue

            expr_t1 = pd.Series(true_pseudobulk[tp1][cell_type], index=gene_names)
            expr_t2 = pd.Series(true_pseudobulk[tp2][cell_type], index=gene_names)

            pred_expr_t1 = pd.Series(pred_pseudobulk[tp1][cell_type], index=gene_names)
            pred_expr_t2 = pd.Series(pred_pseudobulk[tp2][cell_type], index=gene_names)

            log2fc = np.log2((expr_t2 + 1e-8) / (expr_t1 + 1e-8))
            pred_log2fc = np.log2((pred_expr_t2 + 1e-8) / (pred_expr_t1 + 1e-8))
            log2fc.sort_values(ascending=False, inplace=True)
            pred_log2fc.sort_values(ascending=False, inplace=True)

            # now let's do GSEA using gseapy
            true_preranks = gp.prerank(
                rnk=log2fc,
                gene_sets="MSigDB_Hallmark_2020",
                outdir=None,
            )
            pred_preranks = gp.prerank(
                rnk=pred_log2fc,
                gene_sets="MSigDB_Hallmark_2020",
                outdir=None,
            )

            # now let's compare the results, by doing the following:
            # 1) calculating the accuracy of the NES sign
            # 2) calculating the spearman correlation of the gene sets

            nes_count = 0
            threshold = 0.5
            for gene_set in true_preranks.res2d.index:
                if gene_set in pred_preranks.res2d.index:
                    true_nes = true_preranks.res2d.loc[gene_set, "NES"]
                    pred_nes = pred_preranks.res2d.loc[gene_set, "NES"]
                    if (true_nes >= threshold and pred_nes >= threshold) or (
                        true_nes < -threshold and pred_nes < -threshold
                    ):
                        nes_count += 1

            accuracy = nes_count / len(true_preranks.res2d.index)
            metrics["accuracy"].append(accuracy)

            # now let's calculate the spearman correlation
            spearman = spearmanr(
                true_preranks.res2d.sort_values(by="Term")["NES"],
                pred_preranks.res2d.sort_values(by="Term")["NES"],
            )
            metrics["spearman_corr"].append(spearman.statistic)
            metrics["time_pairs"].append(f"{tp1}->{tp2}")

        metrics_df = pd.DataFrame(metrics)

        time_pairs = metrics_df["time_pairs"].values
        x_pos = range(len(time_pairs))
        # Plot overlap as line graph
        ax_overlap = axes_overlap_flat[idx]
        ax_overlap.plot(
            x_pos,
            metrics_df["accuracy"].values,
            marker="o",
            color="steelblue",
            linewidth=2,
            markersize=6,
        )
        ax_overlap.set_title(f"{cell_type}", fontsize=10)
        ax_overlap.set_xlabel("Time Transition")
        ax_overlap.set_ylabel("NES Sign Accuracy")
        ax_overlap.set_xticks(x_pos)
        ax_overlap.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
        ax_overlap.set_ylim(0, 100)
        ax_overlap.grid(True, alpha=0.3)

        # Plot Spearman correlation as line graph
        ax_corr = axes_corr_flat[idx]
        ax_corr.plot(
            x_pos,
            metrics_df["spearman_corr"].values,
            marker="s",
            color="coral",
            linewidth=2,
            markersize=6,
        )
        ax_corr.set_title(f"{cell_type}", fontsize=10)
        ax_corr.set_xlabel("Time Transition")
        ax_corr.set_ylabel("Spearman r")
        ax_corr.set_xticks(x_pos)
        ax_corr.set_xticklabels(time_pairs, rotation=45, ha="right", fontsize=8)
        ax_corr.set_ylim(-1, 1)
        ax_corr.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax_corr.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(cell_types), 20):
        axes_overlap_flat[idx].set_visible(False)
        axes_corr_flat[idx].set_visible(False)

    # Save overlap figure
    fig_overlap.suptitle("Accuracy of NES Sign Agreement", fontsize=14, y=0.995)
    plt.figure(fig_overlap.number)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "accuracy.png"), dpi=300, bbox_inches="tight")
    print(f"Saved overlap plot to {fig_dir}")
    plt.close(fig_overlap)

    # Save Spearman correlation figure
    fig_corr.suptitle(
        "Spearman Correlation between True and Predicted DE Gene Sets",
        fontsize=14,
        y=0.995,
    )
    plt.figure(fig_corr.number)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "spearman.png"), dpi=300, bbox_inches="tight")
    print(f"Saved Spearman correlation plot to {fig_dir}")
    plt.close(fig_corr)


def diff_gene_analysis(
    args,
    pred_embeds,
    true_embeds,
    inferred_cell_types,
    true_cell_types,
    model,
    gene_names,
):
    """
    Perform differential gene analysis on the inferred trajectories.

    This is done through the following steps:
    1. Reconstruct the gene expression profiles from the embeddings.
    2. Pseudo-bulk the gene expression profiles by cell type at each time point
    3. For each cell type, perform differential gene analysis over time.
    4. Compare the DE genes to the true DE genes.

    Then we can do differential gene analysis over:
    a. time for each cell type
    b. between cell types at each time point
    """

    # Step 1: Reconstruct gene expression profiles from embeddings
    print(f"Shape of pred_embeds: {[embed.shape for embed in pred_embeds]}")
    print(f"Shape of true_embeds: {[embed.shape for embed in true_embeds]}")
    pred_recon = reconstruct_gene_expression_from_embeddings(pred_embeds, model)
    true_recon = reconstruct_gene_expression_from_embeddings(true_embeds, model)
    print(f"Shape of pred_recon: {[recon.shape for recon in pred_recon]}")
    print(f"Shape of true_recon: {[recon.shape for recon in true_recon]}")

    # Step 2: Pseudo-bulk the gene expression profiles by cell type at each time point
    pred_pseudo_bulk = pseudo_bulk_by_cell_type(
        pred_recon,
        [
            soft_labels_to_cell_types(inferred_cell_types[t])
            for t in range(len(inferred_cell_types))
        ],
    )
    true_pseudo_bulk = pseudo_bulk_by_cell_type(true_recon, true_cell_types)

    # For the next section, where we measure different metrics, we need to create three
    # different tasks per metric.
    # Task 1: Per cell type, per time point. (done previously)

    # let's create these different pseudobulks!
    # Task 2: Per cell type, aggregated over time points.
    pred_time_aggregate_pseudo_bulk = create_time_aggregate_pseudo_bulk(
        pred_pseudo_bulk
    )
    true_time_aggregate_pseudo_bulk = create_time_aggregate_pseudo_bulk(
        true_pseudo_bulk
    )

    # Task 3: Per time point, aggregated over cell types.
    pred_celltype_aggregate_pseudo_bulk = create_celltype_aggregate_pseudo_bulk(
        pred_pseudo_bulk
    )
    true_celltype_aggregate_pseudo_bulk = create_celltype_aggregate_pseudo_bulk(
        true_pseudo_bulk
    )

    # create the cell types we want!
    cell_types = set()
    for tp in pred_pseudo_bulk:
        cell_types.update(pred_pseudo_bulk[tp].keys())
    for tp in true_pseudo_bulk:
        cell_types.update(true_pseudo_bulk[tp].keys())

    def run_fn(fn):
        """
        Given a function, run it on all three types of tasks
        """
        # Task 1: Per cell type, per time point
        finegrain_metric = {}
        for cell_type in cell_types:
            for tp in range(len(times_sorted)):
                if (
                    cell_type in pred_pseudo_bulk[tp]
                    and cell_type in true_pseudo_bulk[tp]
                ):
                    finegrain_metric[f"{(cell_type, tp)}"] = fn(
                        pred_pseudo_bulk[tp][cell_type], true_pseudo_bulk[tp][cell_type]
                    )
                elif (
                    cell_type not in pred_pseudo_bulk[tp]
                    and cell_type in true_pseudo_bulk[tp]
                ):
                    print(
                        f"Cell type {cell_type} missing in one of the pseudo-bulks at time point {tp}"
                    )
                    finegrain_metric[f"{(cell_type, tp)}"] = None

        # Task 2: Per cell type, aggregated over time points
        time_agg_metric = {}
        for cell_type in cell_types:
            if (
                cell_type in pred_time_aggregate_pseudo_bulk
                and cell_type in true_time_aggregate_pseudo_bulk
            ):
                time_agg_metric[cell_type] = fn(
                    pred_time_aggregate_pseudo_bulk[cell_type],
                    true_time_aggregate_pseudo_bulk[cell_type],
                )
            elif (
                cell_type not in pred_time_aggregate_pseudo_bulk
                and cell_type in true_time_aggregate_pseudo_bulk
            ):
                print(f"Cell type {cell_type} missing in time aggregated pseudo-bulks")
                time_agg_metric[cell_type] = None

        # Task 3: Per time point, aggregated over cell types
        celltype_agg_metric = {}
        for tp in range(len(times_sorted)):
            celltype_agg_metric[tp] = fn(
                pred_celltype_aggregate_pseudo_bulk[tp],
                true_celltype_aggregate_pseudo_bulk[tp],
            )

        return finegrain_metric, time_agg_metric, celltype_agg_metric

    # Step 3: Compute MSE between pseudo-bulks
    finegrain_mse, time_agg_mse, celltype_agg_mse = run_fn(mse_between_pseudo_bulks)

    with open("./logs/diff_gene/mse.txt", "w") as f:
        f.write("Fine-grain MSE per cell type and time point:\n")
        pprint(finegrain_mse, stream=f)
        f.write("Time-aggregated MSE per cell type:\n")
        pprint(time_agg_mse, stream=f)
        f.write("Cell type-aggregated MSE per time point:\n")
        pprint(celltype_agg_mse, stream=f)

    # Step 4: For each cell type, perform differential gene analysis over time
    true_de_set = {}
    pred_de_set = {}
    for cell_type in cell_types:
        true_de_set[cell_type] = time_de_analysis(true_pseudo_bulk, cell_type)
        pred_de_set[cell_type] = time_de_analysis(pred_pseudo_bulk, cell_type)

    plot_time_de(true_de_set, pred_de_set, cell_types)

    # Step 5: Overall differential gene analysis over time (not per cell type)
    true_de_set = time_de_analysis(true_celltype_aggregate_pseudo_bulk, None)
    pred_de_set = time_de_analysis(pred_celltype_aggregate_pseudo_bulk, None)
    plot_time_de_all(true_de_set, pred_de_set)

    # TODO: fix this part here. Likely not the best to use atm...
    # Step 6: Now let's do GSEA analysis, based on the gene names that I have
    # gsea_analysis(
    #     true_pseudo_bulk, pred_pseudo_bulk, gene_names, cell_types
    # )


def prepare_cells(args, ann_data, latent_ode_model):
    """
    Based on the ann_data, prepare the following:
    1. True embeddings at each time point
    2. True cell types at each time point
    3. Predicted embeddings at each time point
    4. Inferred cell types at each time point
    5. Sorted times
    """
    traj_data, tps, times_sorted = prep_traj_data(ann_data)
    tps = tps_to_continuous(tps, times_sorted)

    if args.use_sequential_pred:
        pred_embeds = get_cell_pred_embeds_sequential(
            latent_ode_model, traj_data, tps, args
        )
    else:
        pred_embeds = get_cell_pred_embeds_joint(latent_ode_model, traj_data, tps)

    # ** Note cell prediction embeds are starting from time point 1, not time point 0 **
    true_embeds, true_cell_types = get_cell_embed_by_timepoint(
        ann_data, times_sorted, latent_ode_model
    )

    # now we use these true_embeds to infer the cell labels
    if args.use_knn:
        inferred_cell_types = infer_cell_types_knn(
            true_embeds, pred_embeds, true_cell_types, args
        )
    else:
        inferred_cell_types = infer_cell_types_ot(
            true_embeds, pred_embeds, true_cell_types, args
        )

    return true_embeds, true_cell_types, pred_embeds, inferred_cell_types, times_sorted


if __name__ == "__main__":
    parser = create_parser()
    add_args_to_parser(parser)
    parser.add_argument(
        "--sankey_plot",
        action="store_true",
        help="Whether to plot sankey diagrams for cell type trajectories.",
    )
    parser.add_argument(
        "--diff_genes",
        action="store_true",
        help="Whether to perform differential gene analysis.",
    )
    args = parser.parse_args()

    data_name = args.dataset
    split_type = args.split_type.value

    n_genes = 2000

    latent_ode_model = load_model(n_genes, split_type, args)
    print(f"Successfully loaded model")

    # let's save all of this information if needed
    if not os.path.exists("./logs/embeds.pkl"):
        # 154000 cells by 2000 genes (HVGs) if true
        ann_data, cell_tps, cell_types, n_genes, n_tps = loadSCData(
            data_name,
            split_type,
            path_to_dir="../",
            use_hvgs=args.hvgs,
            normalize_data=args.normalize,
        )

        (
            true_embeds,
            true_cell_types,
            pred_embeds,
            inferred_cell_types,
            times_sorted,
        ) = prepare_cells(args, ann_data, latent_ode_model)
        print(f"Successfully prepared cell embeddings and cell types")

        # finally, let's save a mapping of the genes and its names
        gene_names = ann_data.var_names.tolist()

        torch.save(
            {
                "true_embeds": true_embeds,
                "true_cell_types": true_cell_types,
                "pred_embeds": pred_embeds,
                "inferred_cell_types": inferred_cell_types,
                "times_sorted": times_sorted,
                "gene_names": gene_names,
            },
            "./logs/embeds.pkl",
        )
        print(f"Saved embeddings and cell types to ./logs/embeds.pkl")
    else:
        print(f"Loading pre-saved embeddings and cell types")
        data = torch.load("./logs/embeds.pkl", weights_only=False)
        true_embeds = data["true_embeds"]
        true_cell_types = data["true_cell_types"]
        pred_embeds = data["pred_embeds"]
        inferred_cell_types = data["inferred_cell_types"]
        times_sorted = data["times_sorted"]
        gene_names = data["gene_names"]

    if args.sankey_plot:
        # now we can use these inferred cell types to create trajectories
        trajectories = create_trajectory(inferred_cell_types)
        plot_trajectory_per_cell_type(trajectories, times_sorted, args)
        print(f"Plotted cell type trajectories")

        plot_switch_rate(trajectories, args)
        print(f"Plotted cell type switch rates")

        plot_entropy_over_time(trajectories, args)
        print(f"Plotted cell type entropy over time")
        exit()

    if args.diff_genes:
        print("Diff. gene analysis to be implemented.")
        diff_gene_analysis(
            args,
            pred_embeds,
            true_embeds,
            inferred_cell_types,
            true_cell_types,
            latent_ode_model,
            gene_names,
        )
        exit()
