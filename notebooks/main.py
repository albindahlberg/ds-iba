# Folke Hilding, Albin Åberg Dahlberg
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from sklearn import metrics

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.data import Phi
from notebooks.image_processing_filter import process

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
    X, y = process(xraw, yraw, strength=0.5)

    mask = X != 0
    X, y = X[mask], y[mask]
    sort_idx = np.argsort(X)
    X, y = X[sort_idx], y[sort_idx]
    exponents = [-1/2, -1]
    phi = Phi(X, exponents)
    K = nr_clusters
    w_th=0.4

    model = MixIRLS(K=K, w_th=w_th, plot=True)
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
    
    accuracy = metrics.accuracy_score(ground_truth_clusters, model_clusters)
    confusion_matrix = metrics.confusion_matrix(ground_truth_clusters, model_clusters)
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()
    sns.heatmap(confusion_matrix, 
                annot=True, 
                cmap='YlGnBu', 
                fmt='.2f',
                xticklabels=np.unique(np.concatenate((ground_truth_clusters.unique(), model_clusters.unique()))),
                yticklabels=np.unique(np.concatenate((ground_truth_clusters.unique(), model_clusters.unique()))))
    plt.title(f'Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show(block=False)
    
    print("\nCluster Matching Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy, confusion_matrix

def plot_ground_truth(ground_truth,cmap):
    plt.figure()
    # Hardcoded for nicer plot
    for cluster in ['Noise', 'N', 'Ti']:
        cluster_data = ground_truth.loc[ground_truth["Cluster"] == cluster]
        plt.scatter(
            cluster_data["ToF"], cluster_data["E"], 
            label=f"{cluster}", 
            color=cmap[cluster], 
            alpha=0.5, marker='o', s=1
        )

    plt.xlabel("E (channel)")
    plt.ylabel("ToF (channel)")
    plt.title("Ground Truth Clusters")
    plt.legend()
    plt.show()
    

def plot_model_predictions(model_predictions, cmap):
    plt.figure()
    for cluster in ['Noise', 'N', 'Ti']:
        cluster_data = model_predictions.loc[model_predictions["Cluster"] == cluster]
        plt.scatter(
            cluster_data["ToF"], cluster_data["E"], 
            label=f"{cluster}", 
            color=cmap[cluster], 
            alpha=0.9,
            s=1
        )

    plt.xlabel("E (channel)")
    plt.ylabel("ToF (channel)")
    plt.title("Predicted Clusters")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load in data
    in_data = load_in_file("../cut_eval/in_3/I127_36MeV_ref-TiN_pos02.asc")
    cut_data = load_cut_files("../cut_eval/cut_3")
    # Fill in full data with noise labels
    ground_truth = assign_noise(in_data, cut_data)
    # Train model
    model_predictions = mixirls(in_data)
    cluster_names = {-1: 'Noise', 0: "N", 1: "Ti"}
    # Assing labels to appropriate label name
    model_predictions["Cluster"] = model_predictions["Cluster"].replace(cluster_names)
    # Evaluate
    evaluate_matching(model_predictions, ground_truth)


    ##### Plotting #####
    cmap = {'Noise': 'grey', 'N': 'red', 'Ti': 'blue'}
    plot_ground_truth(ground_truth, cmap)
    plot_model_predictions(model_predictions, cmap)
    

    plt.figure()
    non_noise = ['N', 'Ti']
    for cluster in non_noise:
        cluster_data = model_predictions.loc[model_predictions["Cluster"] == cluster]
        plt.scatter(
            cluster_data["ToF"], cluster_data["E"], 
            label=f"Predicted cluster {cluster}", 
            color=cmap[cluster], 
            alpha=0.9,
            s=1
        )
        
