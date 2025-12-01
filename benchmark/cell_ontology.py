import rdflib
import json


def read_cell_ontology(file_name):
    g = rdflib.Graph()
    g.parse(file_name, format="xml")
    ct_graph_edge_list, ct_graph_vocab_clid2idx = [], {}
    clid_list = []
    for s, p, o in g:
        if "CL" in s and "CL" in o:
            u_clid, fa_clid = s.split("/")[-1], o.split("/")[-1]
            u_clid = u_clid.replace("_", ":")
            fa_clid = fa_clid.replace("_", ":")
            if u_clid == fa_clid:
                continue
            if not (
                u_clid.startswith("CL:") and fa_clid.startswith("CL:")
            ):  # CL:xxxxxxx
                continue

            ct_graph_edge_list.append((u_clid, fa_clid))
            clid_list.append(u_clid)
            clid_list.append(fa_clid)
    clid_list = sorted(list(set(clid_list)))
    ct_graph_vocab_clid2idx = {v: k for k, v in enumerate(clid_list)}
    ct_graph_edge_list = [
        (ct_graph_vocab_clid2idx[u], ct_graph_vocab_clid2idx[v])
        for u, v in ct_graph_edge_list
    ]
    return ct_graph_edge_list, ct_graph_vocab_clid2idx


def get_cell_ontology_subgraph(edges, vocab_clid2idx, relevant_clids):
    """
    Given the edges of the full cell ontology graph and a list of relevant CL IDs,
    return the subgraph that includes only the relevant cell types and their ancestors.
    """
    relevant_indices = set()
    clid_to_idx = vocab_clid2idx
    idx_to_clid = {v: k for k, v in clid_to_idx.items()}

    # First, find indices of relevant CL IDs
    for clid in relevant_clids:
        if clid in clid_to_idx:
            relevant_indices.add(clid_to_idx[clid])

    # Now, iteratively add ancestors
    added = True
    while added:
        added = False
        for u, v in edges:
            if u in relevant_indices and v not in relevant_indices:
                relevant_indices.add(v)
                added = True

    # Create subgraph edges
    subgraph_edges = []
    for u, v in edges:
        if u in relevant_indices and v in relevant_indices:
            subgraph_edges.append((u, v))

    return subgraph_edges, {idx_to_clid[idx]: idx for idx in relevant_indices}


def get_clids(cell_types_file, cell_type_to_clid_file):
    """
    Given a file with cell types and a mapping file from cell types to CL IDs,
    return a list of relevant CL IDs.
    """
    # Read cell types
    with open(cell_types_file, "r") as f:
        cell_types = [line.strip() for line in f]

    # Read mapping from cell types to CL IDs, from a json file

    with open(cell_type_to_clid_file, "r") as f:
        cell_type_to_clid = json.load(f)

    found_cell_types = {}

    # iterate over the json cell types
    for cell_type_dict in cell_type_to_clid:
        cl_id = cell_type_dict["id"]
        name = cell_type_dict["name"]
        if name in cell_types:
            found_cell_types[name] = cl_id

    return found_cell_types


if __name__ == "__main__":
    # now let's read from the file containing cell types used in our datasets
    # use the json file to map them to CL IDs
    # and finally prune the ontology to only include relevant cell types
    found_cell_types = get_clids(
        "../misc_data/onto_cell_types.txt", "../misc_data/celltype_relationship.json"
    )
    print("Found cell types and their CL IDs:", found_cell_types)

    exit()
    file_name = "../misc_data/cl.owl"
    edges, vocab = read_cell_ontology(file_name)
    # edges represent subtype of
    print("Number of edges:", len(edges))
    print("Vocabulary size:", len(vocab))
