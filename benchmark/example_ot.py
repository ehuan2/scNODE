import torch
from geomloss import SamplesLoss
import os
import ot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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
    with open(f"./ot_figs/{name}/results.txt", "a") as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--euclidean", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    args = parser.parse_args()

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
