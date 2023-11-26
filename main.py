import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog


def load_data():
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        return pd.read_csv(file_path)
    else:
        return None


def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    Y = kmeans.fit_predict(X)
    return Y, kmeans.cluster_centers_

# Function to plot clusters using Seaborn
def plot_clusters(X, Y, centroids):
    plt.figure(figsize=(8, 8))
    for cluster_label in range(len(centroids)):
        plt.scatter(X[Y==cluster_label, 0], X[Y==cluster_label, 1], s=50, label=f'Cluster {cluster_label + 1}')


    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='pink', label='Centroids')
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()

def create_gui():
    root = tk.Tk()
    root.title("Customer Segmentation GUI")

    def load_data_button():
        global c_data
        c_data = load_data()
        if c_data is not None:
            info_label.config(text=f"Data loaded. Shape: {c_data.shape}")

    def perform_clustering_button():
        if 'c_data' in globals():
            X = torch.tensor(c_data.iloc[:, [3, 4]].values, dtype=torch.float32)
            n_clusters = int(cluster_entry.get())
            Y, centroids = perform_kmeans(X, n_clusters)
            plot_clusters(X.numpy(), Y, centroids)
        else:
            info_label.config(text="Load data first!")

    load_button = tk.Button(root, text="Load Data", command=load_data_button)
    load_button.pack(pady=10)

    cluster_label = tk.Label(root, text="Enter Number of Clusters:")
    cluster_label.pack()

    cluster_entry = tk.Entry(root)
    cluster_entry.pack()

    perform_button = tk.Button(root, text="Perform Clustering", command=perform_clustering_button)
    perform_button.pack(pady=10)

    info_label = tk.Label(root, text="")
    info_label.pack()

    root.mainloop()

# Run the GUI
create_gui()
