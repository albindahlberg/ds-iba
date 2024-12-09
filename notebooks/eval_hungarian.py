import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os

def load_in_file(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["x", "y"])
    return data

def load_cut_files(cut_dir):
    cut_files = [file for file in os.listdir(cut_dir) if file.endswith(".cut")]
    cut_data = []
    for cut_file in cut_files:
        cut_file_path = os.path.join(cut_dir, cut_file)
        with open(cut_file_path, 'r') as f:
            lines = f.readlines()
        metadata = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
            else:
                break
        data_start_index = len(metadata)
        data = pd.read_csv(cut_file_path, delim_whitespace=True, skiprows=data_start_index, header=None)
        if data.shape[1] >= 3:
            data = data.iloc[:, :3]
            data.columns = ["ToF", "Energy", "Event_number"]
        else:
            raise ValueError(f"Unexpected column structure in file: {cut_file_path}")
        data["Cluster"] = cut_file.split(".")[1]
        data['ToF'] = pd.to_numeric(data['ToF'], errors='coerce')
        data['Energy'] = pd.to_numeric(data['Energy'], errors='coerce')
        data = data.dropna(subset=["ToF", "Energy"])
        cut_data.append(data[["ToF", "Energy", "Cluster"]])
    return pd.concat(cut_data, ignore_index=True)

#placeholder, byt till riktig modell
def simple_model(in_data):
    n_clusters = 6  
    cluster_labels = np.random.choice(range(1, n_clusters + 1), len(in_data))
    in_data["Cluster"] = cluster_labels
    return in_data

def evaluate_matching_with_hungarian(model_data, ground_truth):
    model_clusters = model_data["Cluster"].unique()
    ground_truth_clusters = ground_truth["Cluster"].unique()

    cost_matrix = np.zeros((len(model_clusters), len(ground_truth_clusters)))

    for i, model_cluster in enumerate(model_clusters):
        model_points = model_data[model_data["Cluster"] == model_cluster][["x", "y"]].values
        for j, ground_truth_cluster in enumerate(ground_truth_clusters):
            ground_truth_points = ground_truth[ground_truth["Cluster"] == ground_truth_cluster][["ToF", "Energy"]].values
            if len(model_points) > 0 and len(ground_truth_points) > 0:
                distance_matrix = pairwise_distances(model_points, ground_truth_points)
                cost_matrix[i, j] = distance_matrix.min()

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    results = []
    for i, j in zip(row_ind, col_ind):
        results.append({
            "Model Cluster": model_clusters[i],
            "Ground Truth Cluster": ground_truth_clusters[j],
            "Distance": cost_matrix[i, j]
        })

    results_df = pd.DataFrame(results)
    print("\nCluster Matching Results:")
    print(results_df)
    print(f"\nAverage Matching Distance: {results_df['Distance'].mean()}")

    return results_df

in_data = load_in_file("../cut_eval/in_1/I127_36MeV_ScN-11_pos11.asc")
cut_data = load_cut_files("../cut_eval/cut_1")
model_predictions = simple_model(in_data)  
results = evaluate_matching_with_hungarian(model_predictions, cut_data)

plt.figure(figsize=(10, 8))
for cluster in model_predictions["Cluster"].unique():
    cluster_data = model_predictions[model_predictions["Cluster"] == cluster]
    plt.scatter(cluster_data["x"], cluster_data["y"], label=f"Model Cluster {cluster}", alpha=0.6, s=5)

for cluster in cut_data["Cluster"].unique():
    cluster_data = cut_data[cut_data["Cluster"] == cluster]
    plt.scatter(cluster_data["ToF"], cluster_data["Energy"], label=f"GT Cluster {cluster}", marker='x', alpha=0.6, s=20)

plt.xlabel("Energy")
plt.ylabel("ToF")
plt.title("Model vs Ground Truth Clusters")
plt.legend()
plt.tight_layout()
plt.show()
