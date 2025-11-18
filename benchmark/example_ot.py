import torch
from geomloss import SamplesLoss
from pykeops.torch import generic_sum
import os
import ot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


# this uses two different libraries, Geomloss and POT
# Geomloss: is differentiable, and is actually used in scNODE
# POT: is not differentiable, and can't be used in scNODE, BUT does give the proper OT plan
def ot_example(a, b, name, use_euclidean=False, uniform=False):
    ot_solver = SamplesLoss(
        "sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True, backend="tensorized"
    )

    # note: we assume uniform weighting to all of them, which is the default!
    a = a.cpu().numpy()
    b = b.cpu().numpy()

    # An example of what the weights should be:
    # a_weight = np.ones(len(a)) / len(a)
    # b_weight = np.ones(len(b)) / len(b)

    if use_euclidean:
        ot_result = ot.solve_sample(
            a, b, reg=0.05, metric="euclidean"
        )  # returns full transport plan, should be the same as blur=0.05
    elif uniform:
        a_weight = ot.utils.unif(len(a))
        b_weight = ot.utils.unif(len(b))
        ot_result = ot.solve_sample(a, b, a_weight, b_weight)
    else:
        ot_result = ot.solve_sample(
            a, b, reg=0.05
        )  # returns full transport plan, should be the same as blur=0.05
    print(f"Transport plan: {ot_result.plan} with a value of: {ot_result.value}")

    a_labels = [f"{a[i]}" for i in range(len(a))]
    b_labels = [f"{b[j]}" for j in range(len(b))]

    # Create dataframe
    df = pd.DataFrame(ot_result.plan, index=a_labels, columns=b_labels)

    # --- Print and visualize ---
    print("Transport Plan (P):")
    print(df.round(3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"Transport Plan (ε={0.05})")
    plt.xlabel("Target points (b)")
    plt.ylabel("Source points (a)")

    a = torch.FloatTensor(a)
    b = torch.FloatTensor(b)

    png_name = "euclidean" if use_euclidean else ("uniform" if uniform else "W2")
    os.makedirs(f"./ot_figs/{name}/plan", exist_ok=True)
    with open(f"./ot_figs/{name}/results.txt", "w") as f:
        f.write(f"Geomloss Loss ({png_name}): {ot_solver(a, b)}\n")
        f.write(f"OT Loss ({png_name}): {ot_result.value}\n")

    plt.savefig(f"./ot_figs/{name}/plan/{png_name}.png")
    return ot_result


def plot_transport_same(
    a,
    b,
    P=None,
    reg=None,
    show_flows=True,
    threshold=0.3,
    name="",
    use_euclidean=False,
    uniform=False,
):
    """
    Plot source (a) and target (b) in the same coordinate space.
    'x' = quadrant III (negative), 'o' = quadrant I (positive).
    Lines represent the optimal transport plan weights (P).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot source points
    for i, (x, y) in enumerate(a):
        marker = "x" if x < 0 else "o"
        ax.scatter(
            x, y, color="tab:blue", marker=marker, s=120, label="a" if i == 0 else ""
        )
        ax.text(x + 0.2, y, f"a{i+1}", color="tab:blue", fontsize=10)

    # Plot target points
    for j, (x, y) in enumerate(b):
        marker = "x" if x < 0 else "o"
        ax.scatter(
            x, y, color="tab:orange", marker=marker, s=120, label="b" if j == 0 else ""
        )
        ax.text(x + 0.2, y, f"b{j+1}", color="tab:orange", fontsize=10)

    # Draw flow lines
    if show_flows and P is not None:
        max_P = P.max()
        for i in range(len(a)):
            for j in range(len(b)):
                weight = float(P[i, j]) / max_P
                if weight > threshold:
                    ax.plot(
                        [a[i, 0], b[j, 0]],
                        [a[i, 1], b[j, 1]],
                        color="gray",
                        alpha=weight,
                        linewidth=2.5 * weight,
                    )

    # Axes and decorations
    ax.axhline(0, color="k", linewidth=1)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_title(
        f"Optimal Transport Plan (ε={reg if reg else 'N/A'})\n'x' = QIII, 'o' = QI",
        fontsize=14,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="upper left")

    png_name = "euclidean" if use_euclidean else ("uniform" if uniform else "W2")
    os.makedirs(f"./ot_figs/{name}/vis", exist_ok=True)
    plt.savefig(f"./ot_figs/{name}/vis/{png_name}.png")


def get_transport_plan(a, b, reg=0.05, scaling=0.99, debias=False, reach=None):
    """Uses geomloss to get the potentials and then return the transport plan."""
    ot_solver = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=reg,
        debias=False,
        backend="tensorized",
        scaling=scaling,
        potentials=True,
        reach=reach,
    )
    ot_solver_loss_only = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=reg,
        debias=False,
        backend="tensorized",
        scaling=scaling,
        potentials=False,
        reach=reach,
    )
    ot_solver_debiased = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=reg,
        debias=True,
        backend="tensorized",
        scaling=scaling,
        potentials=True,
        reach=reach,
    )

    F, G = ot_solver(a, b)
    F = F[0]
    G = G[0]

    alpha = torch.tensor(1.0 / len(a))
    beta = torch.tensor(1.0 / len(b))
    total_cost = torch.sum(F) * alpha + torch.sum(G) * beta

    F_unbias, G_unbias = ot_solver_debiased(a, b)
    F_unbias = F_unbias[0]
    G_unbias = G_unbias[0]
    unbiased_total_cost = torch.sum(F_unbias) * alpha + torch.sum(G_unbias) * beta

    print(F, G)
    print(f"Total OT cost, calculated from potentials: {total_cost}")
    print(f"Total Geomloss: {ot_solver_loss_only(a, b)}")
    print(f"Total OT unbiased: {unbiased_total_cost}")

    if debias:
        F = F_unbias
        G = G_unbias

    # Compute transport plan if desired:
    C = torch.cdist(a, b, p=2) ** 2 / 2

    print(f"Cost matrix: {C}")

    Pi = torch.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            Pi[i, j] = torch.exp((F[i] + G[j] - C[i, j]) / (reg**2)) * alpha * beta

    # let's add labels for the data as [1, 0] or [0, 1] depending on the data in b
    l_j = torch.zeros((len(b), 2))
    for j in range(len(b)):
        if b[j][0] > 0 and b[j][1] > 0:
            l_j[j][0] = 1
        else:
            l_j[j][1] = 1
    print(f"Dest: {b}, Labels: {l_j}")

    # now let's also calculate the transferred labels!
    # Define our KeOps CUDA kernel:
    transfer = generic_sum(
        "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",  # See the formula above
        "Lab = Vi(2)",  # Output:  one vector of size 3 per line
        "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
        "X_i = Vi(2)",  # 2nd arg: one 2d-point per line
        "Y_j = Vj(2)",  # 3rd arg: one 2d-point per column
        "F_i = Vi(1)",  # 4th arg: one scalar value per line
        "G_j = Vj(1)",  # 5th arg: one scalar value per column
        "L_j = Vj(2)",
    )  # 6th arg: one vector of size 3 per column

    # And apply it on the data (KeOps is pretty picky on the input shapes...):
    labels_i = transfer(
        torch.Tensor([args.reg**2]).type(torch.FloatTensor),
        a,
        b,
        F.view(-1, 1),
        G.view(-1, 1),
        l_j,
    ) / len(b)
    print(f"Source: {a}, Inferred Labels: {labels_i}")

    return Pi


def simple_example():
    # so first off do a balanced cell type
    balanced_a = torch.FloatTensor(
        [
            # 3rd quadrant, 4 points
            [-5, -5],
            [-5, -7],
            [-7, -5],
            [-7, -7],
            # 1st quadrant, 4 points
            [5, 5],
            [5, 7],
            [7, 5],
            [7, 7],
        ]
    )

    balanced_b = torch.FloatTensor(
        [
            # 3rd quadrant, 4 points, move the square to a diamond
            [-6, -5],
            [-6, -7],
            [-5, -6],
            [-7, -6],
            # 1st quadrant, 4 points, move the square to a diamond
            [6, 5],
            [6, 7],
            [5, 6],
            [7, 6],
        ]
    )

    balanced_result = ot_example(
        balanced_a,
        balanced_b,
        "balanced",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )
    plot_transport_same(
        balanced_a,
        balanced_b,
        P=balanced_result.plan,
        reg=0.05,
        name="balanced",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )

    # next, do a unequal version of this
    unequal_a = torch.FloatTensor(
        [
            # 3rd quadrant, 2 points
            [-5, -6],
            [-7, -6],
            # 1st quadrant, 6 points
            [5, 5],
            [4, 6],
            [5, 7],
            [7, 5],
            [8, 6],
            [7, 7],
        ]
    )

    unequal_b = torch.FloatTensor(
        [
            # 3rd quadrant, 6 points
            [-7, -5],
            [-8, -6],
            [-7, -7],
            [-5, -5],
            [-4, -6],
            [-5, -7],
            # 1st quadrant, 2 points
            [5, 6],
            [7, 6],
        ]
    )

    unequal_result = ot_example(
        unequal_a,
        unequal_b,
        "unequal",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )
    plot_transport_same(
        unequal_a,
        unequal_b,
        P=unequal_result.plan,
        reg=0.05,
        name="unequal",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )

    # finally, we do a version where the number is higher in a than b
    less_holes_a = torch.FloatTensor(
        [
            # 3rd quadrant, 2 points
            [-5, -6],
            [-7, -6],
            # 1st quadrant, 6 points
            [5, 5],
            [4, 6],
            [5, 7],
            [7, 5],
            [8, 6],
            [7, 7],
        ]
    )

    less_holes_b = torch.FloatTensor(
        [
            # 3rd quadrant, 2 points
            [-7, -5],
            [-5, -7],
            # 1st quadrant, 4 points
            [5, 6],
            [7, 6],
        ]
    )

    less_holes_result = ot_example(
        less_holes_a,
        less_holes_b,
        "less_holes",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )
    plot_transport_same(
        less_holes_a,
        less_holes_b,
        P=less_holes_result.plan,
        reg=0.05,
        name="less_holes",
        use_euclidean=args.euclidean,
        uniform=args.uniform,
    )


def test_transport(args):
    # so first off do a balanced cell type
    balanced_a = torch.FloatTensor(
        [
            # 3rd quadrant, 4 points
            [-5, -5],
            [-5, -7],
            [-7, -5],
            [-7, -7],
            # 1st quadrant, 4 points
            [5, 5],
            [5, 7],
            [7, 5],
            [7, 7],
        ]
    )

    balanced_b = torch.FloatTensor(
        [
            # 3rd quadrant, 4 points, move the square to a diamond
            [-6, -5],
            [-6, -7],
            [-5, -6],
            [-7, -6],
            # 1st quadrant, 4 points, move the square to a diamond
            [6, 5],
            [6, 7],
            [5, 6],
            [7, 6],
        ]
    )

    # next, do a unequal version of this
    unequal_a = torch.FloatTensor(
        [
            # 3rd quadrant, 2 points
            [-5, -6],
            [-7, -6],
            # 1st quadrant, 6 points
            [5, 5],
            [4, 6],
            [5, 7],
            [7, 5],
            [8, 6],
            [7, 7],
        ]
    )
    unequal_b = torch.FloatTensor(
        [
            # 3rd quadrant, 6 points
            [-7, -5],
            [-8, -6],
            [-7, -7],
            [-5, -5],
            [-4, -6],
            [-5, -7],
            # 1st quadrant, 2 points
            [5, 6],
            [7, 6],
        ]
    )

    scaling = args.scaling
    reg = args.reg
    debias = args.debias
    reach = args.reach

    def plot_transport(a, b, name):
        a_labels = [f"{a[i].numpy()}" for i in range(len(a))]
        b_labels = [f"{b[j].numpy()}" for j in range(len(b))]
        plan = get_transport_plan(
            a, b, reg=reg, scaling=scaling, debias=debias, reach=reach
        ).numpy()
        print(f"Plan: {plan}")
        df = pd.DataFrame(plan, index=a_labels, columns=b_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Transport Plan (ε={reg})")
        plt.xlabel("Target points (b)")
        plt.ylabel("Source points (a)")
        dir = f"{name}/geom_loss_plan/debias_{debias}/scaling_{scaling}/reg_{reg}/reach_{reach}"
        os.makedirs(f"./ot_figs/{dir}", exist_ok=True)
        plt.savefig(f"./ot_figs/{dir}/plan.png")
        plot_transport_same(
            a.numpy(),
            b.numpy(),
            P=plan,
            reg=reg,
            name=dir,
        )

    plot_transport(balanced_a, balanced_b, "balanced")
    plot_transport(unequal_a, unequal_b, "unequal")


def test_label_transfer(a, b, b_labels, args):
    """
    Given source points a and target points b, test the label transfer functionality.
    """
    ot_solver = SamplesLoss(
        "sinkhorn",
        p=2,
        blur=args.reg,
        debias=True,
        backend="tensorized",
        scaling=args.scaling,
        potentials=True,
        reach=args.reach,
    )

    F, G = ot_solver(a, b)

    transfer = generic_sum(
        "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",  # See the formula above
        f"Lab = Vi({b_labels.shape[1]})",  # Output:  one vector of size one_hot_labels per line
        "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
        f"X_i = Vi({a.shape[1]})",  # 2nd arg: one 2d-point per line
        f"Y_j = Vj({b.shape[1]})",  # 3rd arg: one 2d-point per column
        "F_i = Vi(1)",  # 4th arg: one scalar value per line
        "G_j = Vj(1)",  # 5th arg: one scalar value per column
        f"L_j = Vj({b_labels.shape[1]})",
    )  # 6th arg: one vector of size 3 per column

    labels_i = transfer(
        torch.Tensor([args.reg**2]).type(torch.FloatTensor),
        a,
        b,
        F.view(-1, 1),
        G.view(-1, 1),
        b_labels,
    ) / len(b)

    true_labels = np.argmax(b_labels, axis=1)
    infer_labels = np.argmax(labels_i, axis=1)

    # now let's print these labels one by one:
    with open("ot_figs/label_transfer_results.txt", "w") as f:
        correct = 0
        for i in range(len(a)):
            f.write(
                f"Point {i} out of {len(a)}, True Label: {true_labels[i]}, Inferred Label: {infer_labels[i]}"
            )
            f.write(f", Inferred One Hot Vector: {labels_i[i].tolist()}\n")
            if true_labels[i] == infer_labels[i]:
                correct += 1
        accuracy = correct / len(a)
        f.write(f"\nOverall Label Transfer Accuracy: {accuracy * 100:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--euclidean", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--debias", action="store_true")
    parser.add_argument(
        "--reg", type=float, default=0.05, help="Weight of OT regularization"
    )
    parser.add_argument(
        "--scaling", type=float, default=0.5, help="Weight of OT regularization"
    )
    parser.add_argument(
        "--reach", type=float, default=None, help="Weight of OT regularization"
    )
    args = parser.parse_args()

    a = torch.load("ot_figs/true_embed.pt")
    true_test_labels = torch.load("ot_figs/one_hot_labels.pt")

    # print(test_labels.shape)

    # let's give dummy labels for testing
    # k = 17
    k = true_test_labels.shape[1]
    test_labels = torch.zeros((a.shape[0], k))

    for i in range(a.shape[0]):
        test_labels[i][np.argmax(true_test_labels[i])] = 1

    # check to see if every index in true_test_labels match test_labels
    for i in range(a.shape[0]):
        assert np.argmax(true_test_labels[i]) == np.argmax(
            test_labels[i]
        ), f"Mismatch at index {i}"

    # assert torch.sum(true_test_labels, dim=1).eq(1).all(), "Each test label should be one-hot encoded."

    # print(true_test_labels.shape, test_labels.shape)

    # let's have the classes be split into k of them
    # for i in range(k):
    #     test_labels[a.shape[0] // k * i : a.shape[0] // k * (i + 1), i] = 1

    # let's print out what the test labels look like
    # test_labels = torch.load("ot_figs/one_hot_labels.pt")
    # with open("ot_figs/true_labels.txt", "w") as f:
    #     for i in range(test_labels.shape[0]):
    #         label = torch.argmax(test_labels[i]).item()
    #         f.write(f"Point {i}: True Label: {label}")
    #         # print out the entire one hot vector as well
    #         f.write(f", One Hot Vector: {test_labels[i].tolist()}\n")

    test_label_transfer(a, a, test_labels, args)
