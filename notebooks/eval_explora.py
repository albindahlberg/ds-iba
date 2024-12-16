import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "../cut_eval/"
cut_dir = os.path.join(base_dir, "cut_1")

def load_cut_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    metadata = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
        else:
            break
    data_start_index = len(metadata)
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=data_start_index,
        header=None
    )
    if data.shape[1] >= 3:
        data = data.iloc[:, :3]
        data.columns = ["ToF", "Energy", "Event_number"] 
    else:
        raise ValueError(f"Unexpected column structure in file: {file_path}")
    data['ToF'] = pd.to_numeric(data['ToF'], errors='coerce')
    data['Energy'] = pd.to_numeric(data['Energy'], errors='coerce')
    data['Event_number'] = pd.to_numeric(data['Event_number'], errors='coerce')
    data = data.dropna(subset=['ToF', 'Energy'])
    return metadata, data

cut_files = [file for file in os.listdir(cut_dir) if file.endswith(".cut")]

plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'olive', 'gray']
for i, cut_file in enumerate(cut_files):
    cut_file_path = os.path.join(cut_dir, cut_file)
    metadata, cut_data = load_cut_file(cut_file_path)
    element = cut_file.split(".")[1]
    plt.scatter(cut_data['Energy'], cut_data['ToF'], s=0.5, label=element, alpha=0.7, color=colors[i % len(colors)])

plt.legend(title="Elements", fontsize=10)
plt.xlabel('Energy', fontsize=12)
plt.ylabel('ToF', fontsize=12)
plt.title('Combined Visualization of All Cut Files', fontsize=14)
plt.tight_layout()
plt.show()
