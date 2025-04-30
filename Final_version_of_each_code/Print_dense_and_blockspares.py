import numpy as np


data1 = np.load("h2o_mps_complete.npy", allow_pickle=True).item()

with open("mps_data_output_1.txt", "w") as f:
    f.write(f"n_sites:   {data1['n_sites']}\n")
    f.write(f"bond_dims: {data1['bond_dims']}\n")
    f.write(f"energy:    {data1['energy']:.12f}\n\n")

    dense_all = data1['dense_tensors']        # <─ added line

    for i, (tensor, qlbl, dense) in enumerate(          # <─ add dense here
            zip(data1['tensors'], data1['q_labels'], dense_all)):

        f.write(f"Tensor {i}  (block-sparse flat shape {tensor.shape}):\n")
        f.write(f"{tensor}\n\n")
        f.write(f"q_labels {i} (rows={len(qlbl)}):\n{qlbl}\n\n")
        f.write(f"Dense Tensor {i} (shape {dense.shape}):\n")
        f.write(np.array2string(dense, precision=8, threshold=1_000_000))
        f.write("\n" + "-"*80 + "\n\n")



