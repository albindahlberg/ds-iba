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
    data = pd.read_csv(file_path, sep='\s+', header=None, names=["x", "y"])
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
            data.columns = ["ToF", "Energy", "Event_number"]
        else:
            raise ValueError(f"Unexpected column structure in file: {cut_file_path}")
        data["Cluster"] = cut_file.split(".")[1]
        data['ToF'] = pd.to_numeric(data['ToF'], errors='coerce')
        data['Energy'] = pd.to_numeric(data['Energy'], errors='coerce')
        data = data.dropna(subset=["ToF", "Energy"])
        cut_data.append(data[["ToF", "Energy", "Cluster"]])
    return pd.concat(cut_data, ignore_index=True)

def assign_noise(in_data, cut_data):
    # Copy the original in_data to result_data
    result_data = in_data.copy()
    
    # Initialize the 'Cluster' column with 'Noise'
    result_data['Cluster'] = 'Noise'
    
    # Iterate over the rows in cut_data
    for _, row in cut_data.iterrows():
        # Find matching rows in result_data based on x, y, ToF, and E
        matching_rows = result_data[(result_data['x'] == row['ToF']) & (result_data['y'] == row['Energy'])]
        
        # Update the 'Cluster' of the matching rows with the value from cut_data
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
    cluster_matches = []
    correct_assignments = 0
    model_clusters = model_data["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"})#.unique()
    ground_truth_clusters = ground_truth["Cluster"]#.replace({-1: "Noise", 0: "N", 1: "Ti"}).unique()

    
    for model_cluster in model_clusters.unique():
        model_points = model_data[model_data["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}) == model_cluster][["x", "y"]].values
        best_match, min_distance = None, float("inf")
        match_counts = {}
        
        for ground_truth_cluster in ground_truth_clusters:
            ground_truth_points = ground_truth[ground_truth["Cluster"].replace({-1: "Noise", 0: "N", 1: "Ti"}) == ground_truth_cluster][["x", "y"]].values
            if len(model_points) > 0 and len(ground_truth_points) > 0:
                distance_matrix = pairwise_distances(model_points, ground_truth_points)
                distance = distance_matrix.min(axis=1).mean()
                if distance < min_distance:
                    best_match, min_distance = ground_truth_cluster, distance
            
            match_count = len(set(map(tuple, model_points)).intersection(set(map(tuple, ground_truth_points))))
            match_counts[ground_truth_cluster] = match_count
        
        cluster_matches.append({
            "Model Cluster": model_cluster,
            "Ground Truth Cluster": best_match,
            "Distance": min_distance
        })
        correct_assignments += match_counts.get(best_match, 0)
    confusion_matrix = metrics.confusion_matrix(ground_truth_clusters, model_clusters)
    normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
    accuracy = metrics.accuracy_score(ground_truth_clusters, model_clusters)
    

    plt.figure(figsize=(10, 7))
    sns.heatmap(normalized_confusion_matrix, 
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
    print(results)
    print(f"\nAverage Matching Distance: {results['Distance'].mean()}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return results, accuracy, confusion_matrix

in_data = load_in_file("../cut_eval/in_3/I127_36MeV_ref-TiN_pos02.asc")
cut_data = load_cut_files("../cut_eval/cut_3")
ground_truth = assign_noise(in_data, cut_data)

color_dict = {'N': 'blue', 'Ti': 'red', 'Noise': 'gray'}

model_predictions = mixirls(in_data)

results = evaluate_matching(model_predictions, ground_truth)
clusters = model_predictions["Cluster"].unique()

plt.figure(figsize=(10, 8))
for cluster in cut_data["Cluster"].unique():
    cluster_data = cut_data[cut_data["Cluster"] == cluster]
    plt.scatter(
        cluster_data["ToF"], cluster_data["Energy"], 
        label=f"GT Cluster {cluster}", 
        color='yellow' if cluster == "N" else 'cyan', 
        alpha=0.5, marker='o', s=20
    )

cluster_names = {-1:'Noise', 0: "N", 1: "Ti"}
cmap = {-1: 'grey', 0:'red', 1:'blue'}

for cluster in clusters:
    cluster_data = model_predictions[model_predictions["Cluster"] == cluster]
    cluster_label = cluster_names.get(cluster, cluster)  
    plt.scatter(
        cluster_data["x"], cluster_data["y"], 
        label=f"Predicted cluster {cluster_label}", 
        color=cmap[cluster], 
        alpha=0.9, s=5
    )

plt.xlabel("Energy")
plt.ylabel("ToF")
plt.title("Model vs Ground Truth Clusters")
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
for cluster in ground_truth['Cluster'].unique():
    plt.scatter(ground_truth.loc[ground_truth['Cluster'] == cluster]['x'], ground_truth.loc[ground_truth['Cluster'] == cluster]['y'], 
            c=color_dict[cluster],
            label=cluster,
            alpha=0.6, 
            s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot by Cluster Type')
plt.legend()
plt.show()