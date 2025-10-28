"""
Test out an example of a sinkhorn loss, and how it might not properly capture
the cell types that we want.
"""

import torch
import geomloss

ot_solver = geomloss.SamplesLoss("sinkhorn")
t_est = torch.tensor(
    [
        [0, 10],  # A
        [10, 0],  # A
        [20, 10],  # A
        [0, 0],  # B
        [10, 10],  # B
        [20, 0],  # B
    ],
    dtype=torch.float,
)
t_true = torch.tensor(
    [
        [0, 10],  # A
        [10, 10],  # A
        [20, 10],  # A
        [0, 0],  # B
        [10, 0],  # B
        [20, 0],  # B
    ],
    dtype=torch.float,
)
print(f"Sinkhorn loss: {ot_solver(t_true, t_est)}")
