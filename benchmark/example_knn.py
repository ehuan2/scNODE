"""
example_knn.py

Example script on doing label transfer using kNN.
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np


def example_knn(a, b, b_labels):
    """
    Given two datasets a and b with known cell type labels in b,
    infer cell type labels in a using kNN from b.
    """
    print("Dataset a:", a)
    print("Dataset b:", list(zip(b, b_labels)))

    k = 3

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(b)

    _, indices = nn.kneighbors(a)

    print(indices, b_labels)

    for i, a_i in enumerate(a):
        print(f"Label for {a_i} is {[b_labels[n_idx] for n_idx in indices[i]]}")


if __name__ == "__main__":
    example_knn(
        np.array([[5, 5], [-5, -5]]),
        np.array(
            [
                [start + i, start + j]
                for start in [-5, 5]
                for i in range(-1, 2)
                for j in range(-1, 2)
            ]
        ),
        np.array([[1, 0]] * 9 + [[0, 1]] * 9),
    )
