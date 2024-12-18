import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.preprocess import Phi
from image_processing import process

from models.mixture import MixIRLS

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

def mixirls(in_data):
    nr_clusters = 2
    in_data_complete = in_data.to_numpy()
    in_data_unique = np.unique(in_data_complete, axis=0)
    xraw = in_data_unique[:,1]
    yraw = in_data_unique[:,0]
    X, y = process(xraw, yraw, strength=0.1)

    mask = X != 0
    X, y = X[mask], y[mask]
    sort_idx = np.argsort(X)
    X, y = X[sort_idx], y[sort_idx]
    exponents = [-1/2, -1]
    phi = Phi(X, exponents)
    K = nr_clusters
    w_th=0.9

    model = MixIRLS(K=K, w_th=w_th)
    model.train(phi, y)
    
    xraw = in_data_complete[:,1]
    yraw = in_data_complete[:,0]
    phi_raw = Phi(xraw, exponents=exponents)
    result = model.assign_cluster(phi_raw, yraw)
    in_data["Cluster"] = result
    return in_data

def evaluate_matching(model_data, ground_truth):
    cluster_matches = []
    correct_assignments = 0
    total_points = len(model_data)
    clusters = model_data["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}).unique()
    ground_truth_clusters = ground_truth["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}).unique()

    confusion_matrix = pd.DataFrame(0, index=ground_truth_clusters, columns=clusters)
    
    for model_cluster in clusters:
        model_points = model_data[model_data["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}) == model_cluster][["x", "y"]].values
        best_match, min_distance = None, float("inf")
        match_counts = {}
        
        for ground_truth_cluster in ground_truth_clusters:
            ground_truth_points = ground_truth[ground_truth["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}) == ground_truth_cluster][["ToF", "Energy"]].values
            if len(model_points) > 0 and len(ground_truth_points) > 0:
                distance_matrix = pairwise_distances(model_points, ground_truth_points)
                distance = distance_matrix.min(axis=1).mean()
                if distance < min_distance:
                    best_match, min_distance = ground_truth_cluster, distance
            
            match_count = len(set(map(tuple, model_points)).intersection(set(map(tuple, ground_truth_points))))
            match_counts[ground_truth_cluster] = match_count
            confusion_matrix.at[ground_truth_cluster, model_cluster] += match_count
        
        cluster_matches.append({
            "Model Cluster": model_cluster,
            "Ground Truth Cluster": best_match,
            "Distance": min_distance,
            "Accuracy": match_counts.get(best_match, 0) / len(model_points) if len(model_points) > 0 else 0
        })
        correct_assignments += match_counts.get(best_match, 0)
    
    results = pd.DataFrame(cluster_matches)
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0).fillna(0)
    accuracy = correct_assignments / total_points

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Percentage")
    plt.xticks(range(len(confusion_matrix.columns)), confusion_matrix.columns, rotation=45)
    plt.yticks(range(len(confusion_matrix.index)), confusion_matrix.index)
    plt.xlabel("Predicted Clusters")
    plt.ylabel("Ground Truth Clusters")
    plt.title("Confusion Matrix (Percentages)")
    plt.tight_layout()
    plt.show()

    print("\nCluster Matching Results:")
    print(results)
    print(f"\nAverage Matching Distance: {results['Distance'].mean()}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return results, accuracy, confusion_matrix

in_data = load_in_file("../cut_eval/in_3/I127_36MeV_ref-TiN_pos02.asc")
cut_data = load_cut_files("../cut_eval/cut_3")
model_predictions = mixirls(in_data)
results = evaluate_matching(model_predictions, cut_data)
clusters = model_predictions["Cluster"].unique()

plt.figure(figsize=(10, 8))
for cluster in cut_data["Cluster"].unique():
    cluster_data = cut_data[cut_data["Cluster"] == cluster]
    plt.scatter(
        cluster_data["ToF"], cluster_data["Energy"], 
        label=f"GT Cluster {cluster}", 
        color='grey' if cluster == "N" else 'cyan', 
        alpha=0.5, marker='o', s=20
    )

cluster_names = {0: "N", 1: "Ti"}
for cluster in clusters[1:]:
    cluster_data = model_predictions[model_predictions["Cluster"] == cluster]
    cluster_label = cluster_names.get(cluster, cluster)  
    plt.scatter(
        cluster_data["x"], cluster_data["y"], 
        label=f"Predicted cluster {cluster_label}", 
        color='red' if cluster == 0 else 'blue', 
        alpha=0.9, s=5
    )

plt.xlabel("Energy")
plt.ylabel("ToF")
plt.title("Model vs Ground Truth Clusters")
plt.legend()
plt.tight_layout()
plt.show()
