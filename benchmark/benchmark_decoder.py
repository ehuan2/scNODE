# test_dataset.py
# used to test the data and examine the dataset
import argparse

from benchmark.BenchmarkUtils import Dataset, SplitType, loadSCData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset_sel = [dataset.value for dataset in list(Dataset)]
    parser.add_argument(
        "-d",
        "--dataset",
        type=Dataset,
        choices=list(Dataset),
        metavar=f"{dataset_sel}",
        default=Dataset.HERRING_GABA,
        help="The dataset to evaluate from",
    )
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--traj_view", action="store_true")
    parser.add_argument("--hvgs", action="store_true")
    parser.add_argument("--per_cell_type", action="store_true")

    split_type_sel = [split_type.value for split_type in list(SplitType)]
    parser.add_argument(
        "-s",
        "--split_type",
        type=SplitType,
        choices=list(SplitType),
        metavar=f"{split_type_sel}",
        default=SplitType.THREE_INTERPOLATION,
        help="split type to choose from",
    )
    parser.add_argument("-n", "--normalize", action="store_true")

    # so we add an argument to train a specific cell type, if it doesn't exist
    # then we train all cell types
    parser.add_argument("--cell_type_to_train", type=str, default="")
    parser.add_argument("--cell_type_to_vis", type=str, default="")

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

    major_clust = ann_data.obs["major_clust"].unique().tolist()
    print(f"Cell types: {major_clust}, num: {len(major_clust)}")
    # GABA: 27500 cells x 22500 genes
    # Full: 154000 cells x 26700 genes -- way too many probably...
    # Think about the total number of cells per timepoint
    # now we want to split it per cell-type

    cell_type = args.cell_type_to_vis
    cell_type_data = ann_data[ann_data.obs["major_clust"] == cell_type].copy()
