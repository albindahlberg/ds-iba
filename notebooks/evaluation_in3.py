import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from sklearn import metrics

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.preprocess import Phi
from image_processing import process

from models.mixture import MixIRLS

def load_in_file(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None, names=["ToF", "E"])
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
        data = pd.read_csv(cut_file_path, sep='\s+', skiprows=data_start_index, header=None)
        if data.shape[1] >= 3:
            data = data.iloc[:, :3]
            data.columns = ["ToF", "E", "Event_number"]
        else:
            raise ValueError(f"Unexpected column structure in file: {cut_file_path}")
        data["Cluster"] = cut_file.split(".")[1]
        data['ToF'] = pd.to_numeric(data['ToF'], errors='coerce')
        data['E'] = pd.to_numeric(data['E'], errors='coerce')
        data = data.dropna(subset=["ToF", "E"])
        cut_data.append(data[["ToF", "E", "Cluster"]])
    return pd.concat(cut_data, ignore_index=True)

def assign_noise(in_data, cut_data):
    result_data = in_data.copy()
    result_data['Cluster'] = 'Noise'

    for _, row in cut_data.iterrows():
        matching_rows = result_data[(result_data['ToF'] == row['ToF']) & (result_data['E'] == row['E'])]
        result_data.loc[matching_rows.index, 'Cluster'] = row['Cluster']

    return result_data


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
    model_clusters = model_data["Cluster"]
    ground_truth_clusters = ground_truth["Cluster"]
    
    confusion_matrix = metrics.confusion_matrix(ground_truth_clusters, model_clusters)
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    accuracy = metrics.accuracy_score(ground_truth_clusters, model_clusters)

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, 
                annot=True, 
                cmap='YlGnBu', 
                fmt='.2f',
                xticklabels=np.unique(np.concatenate((ground_truth_clusters.unique(), model_clusters.unique()))),
                yticklabels=np.unique(np.concatenate((ground_truth_clusters.unique(), model_clusters.unique()))))
    plt.title(f'Normalized Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show(block=False)
    
    print("\nCluster Matching Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy, confusion_matrix

in_data = load_in_file("../cut_eval/in_3/I127_36MeV_ref-TiN_pos02.asc")
cut_data = load_cut_files("../cut_eval/cut_3")
ground_truth = assign_noise(in_data, cut_data)

color_dict = {'N': 'blue', 'Ti': 'red', 'Noise': 'gray'}

model_predictions = mixirls(in_data)

cluster_names = {-1: 'Noise', 0: "N", 1: "Ti"}
model_predictions["Cluster"] = model_predictions["Cluster"].replace(cluster_names)

results = evaluate_matching(model_predictions, ground_truth)
# Create subplots with horizontal layout
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for Ground Truth clusters
for cluster in ['Noise']:
    cluster_data = ground_truth.loc[ground_truth["Cluster"] == cluster]
    axes[0].scatter(
        cluster_data["ToF"], cluster_data["E"], 
        label=f"GT Cluster {cluster}", 
        color=color_dict[cluster], 
        alpha=0.5, marker='o', s=20
    )

axes[0].set_xlabel("E")
axes[0].set_ylabel("ToF")
axes[0].set_title("Ground Truth Clusters")
axes[0].legend()

# Plot for Predicted clusters
cmap = {'Noise': 'grey', 'N': 'red', 'Ti': 'blue'}

clusters = model_predictions["Cluster"].unique()
for cluster in clusters:
    cluster_data = model_predictions.loc[model_predictions["Cluster"] == cluster]
    axes[1].scatter(
        cluster_data["ToF"], cluster_data["E"], 
        label=f"Predicted cluster {cluster}", 
        color=cmap[cluster], 
        alpha=0.9,
        s=1
    )

axes[1].set_xlabel("E")
axes[1].set_ylabel("ToF")
axes[1].set_title("Predicted Clusters")
axes[1].legend()

# Adjust layout for better visualization
plt.tight_layout()
plt.show()