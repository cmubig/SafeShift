import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time
import pandas as pd

from matplotlib import cm
from operator import itemgetter
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

import shutil

from utils.common import (
    load_infos, load_features, VISUAL_OUT_DIR, CACHE_DIR, HEADING_IDX, VEL_XY_IDX, POS_XYZ_IDX,
)
from utils.visualization import (
    plot_static_map_infos, plot_dynamic_map_infos, get_color_map, plot_cluster_overlap,
    AGENT_COLOR
)

np.set_printoptions(suppress=True, precision=3)

def feature_scaling(features):
    return (features - features.mean(axis=0).reshape(1, -1)) / features.std(axis=0).reshape(1, -1)

def unpack(features: dict, scale: bool = True):
    N = len(features)
    agent_features = np.asarray([features[i]['agent_type'].sum(axis=-1) for i in range(N)]).reshape(-1, 1)
    lane_features = np.asarray([features[i]['centerline'] for i in range(N)]) 
    full_trajectory_features = np.asarray([features[i]['full_trajectory'] for i in range(N)]) 
    hist_trajectory_features = np.asarray([features[i]['hist_trajectory'] for i in range(N)])
    
    full_features = np.concatenate((agent_features, lane_features, full_trajectory_features), axis=1)
    hist_features = np.concatenate((agent_features, lane_features, hist_trajectory_features), axis=1)

    if scale:
        full_features = feature_scaling(full_features)
        hist_features = feature_scaling(hist_features)

    return full_features, hist_features

def run_pca_analysis(
    features, shards: list, min_desired_explained_variance: float = 0.90, tag: str = 'temp'
):
    output = {}
    N, D = features.shape

    # PCA explained variance 
    pca = PCA()
    pca.fit(features)

    num_components = np.arange(1, D+1)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    idx = np.where(explained_variance >= min_desired_explained_variance)[0]

    plt.figure(figsize=(10, 5))
    plt.ylim(0.0, 1.1)
    plt.plot(num_components, explained_variance, marker='o', linestyle='--', color='blue', alpha=0.8)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('PCA Explained Variance')

    plt.axhline(y=min_desired_explained_variance, color='red', linestyle='-', alpha=0.5)
    plt.xticks(np.arange(min(num_components), max(num_components)+1, 1.0))
    plt.grid(axis='x', alpha=0.5)
    plt.subplots_adjust(wspace=0.05)

    plt.savefig(
        f"{tag}pca_analyis_shards{shards[0]}-{shards[-1]}.png", dpi=500, bbox_inches='tight')
    plt.close()

    print(f"PCA")
    print(f"Explained variance {pca.explained_variance_}")
    print(f"Explained variance ratio {pca.explained_variance_ratio_}")
    print(f"Number of components for desired variance of {min_desired_explained_variance} is {num_components[idx[0]]}")
    
    output = {
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'num_components': num_components,
        'min_num_components': num_components[idx[0]]
    }
    return pca, output

def run_kmeans_analysis(
    scores, shards: list, num_clusters: list,  max_components: int, tag: str = 'temp'
):
    # Plot K-Means clusters
    print("K-Means")
    output = {}
    components = range(2, max_components+1)
    num_clusters = range(num_clusters[0], num_clusters[1])
    for nc in components:
        wcss = []
        silhouette_scores = []
        for k in num_clusters:
            print(f"Running K-Means with {k} clusters using {nc} components", end='\r')
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
            kmeans.fit(scores[:, :nc])
            wcss.append(kmeans.inertia_)

            silhouette_scores.append(silhouette_score(scores[:, :nc], kmeans.labels_))

            sample_silhouette_values = silhouette_samples(scores[:, :nc], kmeans.labels_)
            y_lower = 10

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10 

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_scores[-1], color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(kmeans.labels_.astype(float) / k)
            ax2.scatter(
                scores[:, 0], scores[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
            centers = kmeans.cluster_centers_

            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k",)

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % k,
                fontsize=14,
                fontweight="bold",
            )
            plt.savefig(
                f"{tag}pca{nc}_kmeans_k{k}_shards{shards[0]}-{shards[-1]}.png", 
                dpi=300, bbox_inches='tight')

        plt.figure(figsize=(10, 5))
        plt.plot(num_clusters, wcss, color='blue', marker='o', linestyle='--', alpha=0.8)
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('K-Means Elbow')
        plt.xticks(np.arange(min(num_clusters), max(num_clusters)+1, 1.0))
        plt.grid(axis='x', alpha=0.5)
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(
            f"{tag}kmeans_wcss_pca-{nc}_shards{shards[0]}-{shards[-1]}.png", 
            dpi=300, bbox_inches='tight')
        plt.close()

def clustering_analysis(
    features: np.array, shards: list, num_clusters: int = [2, 10], tag: str = 'temp', 
    min_desired_variance: float = 0.90
):
    output = {}
    
    # Unpack all features
    full_features, hist_features = unpack(features)

    N, D = full_features.shape

    full_pca, full_output = run_pca_analysis(
        full_features, shards, min_desired_variance, tag=f"{tag}_full_")
    full_scores = full_pca.transform(full_features)
    output['pca_full'] = full_output
    run_kmeans_analysis(
        full_scores, shards, num_clusters, output['pca_full']['min_num_components'], tag=f"{tag}_full_")

    hist_pca, hist_output = run_pca_analysis(
        hist_features, shards, min_desired_variance, tag=f"{tag}_hist_")
    hist_scores = hist_pca.transform(hist_features)
    output['pca_hist'] = hist_output
    run_kmeans_analysis(
        hist_scores, shards, num_clusters, output['pca_hist']['min_num_components'], tag=f"{tag}_hist_")
    
    # output['num_clusters'] = num_clusters
    # output['num_components'] = num_components
    # output['scenario_labels'] = kmeans.labels_

    # if plot:
    #     for label in range(num_clusters):
    #         cluster_dir = f"{tag}/cluster_{label}_shards{shards[0]}-{shards[-1]}"
    #         if os.path.exists(cluster_dir):
    #             shutil.rmtree(cluster_dir)
    #         os.makedirs(cluster_dir, exist_ok=True)
        
    #     i = 0
    #     for (s, path), label in tqdm(zip(inputs, kmeans.labels_)):
    #         if i > 100: break
    #         i += 1
    #         with open(path, 'rb') as f:
    #             scenario = pkl.load(f)

    #         num_windows = 2
    #         fig, ax = plt.subplots(1, num_windows, figsize=(5 * num_windows, 5 * 1))
    #         plot_static_map_infos(scenario['map_infos'], ax)
    #         plot_dynamic_map_infos(scenario['dynamic_map_infos'], ax)

    #         track_infos = scenario['track_infos']
    #         trajectories = track_infos['trajs'][:, :, :-1]
    #         valid_masks = track_infos['trajs'][:, :, -1] > 0
    #         object_types = track_infos['object_type']

    #         num_agents, _, _ = trajectories.shape

    #         for n in range(num_agents):
    #             mask = np.where(valid_masks[n])[0]
    #             pos = trajectories[n, mask][:, POS_XYZ_IDX]
        
    #             ax[0].plot(pos[:, 0], pos[:, 1], AGENT_COLOR[object_types[n]])
                
    #             for a in ax.reshape(-1):
    #                 a.set_xticks([])
    #                 a.set_yticks([])

    #         extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #         plt.subplots_adjust(wspace=0.05)
    #         plt.savefig(
    #             os.path.join(f"{tag}", f"cluster_{label}_shards{shards[0]}-{shards[-1]}", f"{s.split('.')[0]}.png"), 
    #             dpi=500, bbox_inches=extent)
    #         plt.show()
    #         plt.close()
    
    return output

def fit(feats, num_components, num_clusters, shards, tag):
    pca = PCA(n_components=num_components)
    pca.fit(feats)
    scores = pca.transform(feats)

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(scores)
    plot_cluster_overlap(num_clusters, num_components, kmeans.labels_, scores, shards, f"{tag}")
    return pca, kmeans

def fit_wrapper(features, inputs, num_components, num_clusters, shards, cache_filepath, tag):
    full_features, hist_features = unpack(features=features)
    
    # Full trajectories (interp region)
    full_pca, full_kmeans = fit(full_features, num_components, num_clusters, shards, f'{tag}_full')
    # Hist trajectories (interp region)
    hist_pca, hist_kmeans = fit(hist_features, num_components, num_clusters, shards, f'{tag}_hist')

    labels = {
        'full_traj_labels': {}, 'hist_traj_labels': {},
    }

    zipped = zip(inputs, full_kmeans.labels_, hist_kmeans.labels_)
    for (s, _), full_label, hist_label in zipped:
        scenario_id = f"{s.split('.')[0]}"
        labels['full_traj_labels'][scenario_id] = full_label
        labels['hist_traj_labels'][scenario_id] = hist_label
    
    with open(cache_filepath, 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

    return (full_pca, full_kmeans), (hist_pca, hist_kmeans)

def predict(model, features):
    pca, kmeans = model
    features = pca.transform(features)
    labels = kmeans.predict(features)
    return labels

def predict_wrapper(full_model, hist_model, inputs, features, cache_filepath):
    full_features, hist_features = unpack(features)
    
    full_labels = predict(full_model, full_features)
    hist_labels = predict(hist_model, hist_features)
    
    labels = {
        'full_traj_labels': {}, 'hist_traj_labels': {}
    }

    zipped = zip(inputs, full_labels, hist_labels)
    for (s, _), full_label, hist_label in zipped:
        scenario_id = f"{s.split('.')[0]}"
        labels['full_traj_labels'][scenario_id] = full_label
        labels['hist_traj_labels'][scenario_id] = hist_label

    with open(cache_filepath, 'wb') as f:
        pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='~/monet_shared/shared/mtr_process')
    parser.add_argument(
        '--split', type=str, default='training', choices=['training', 'validation', 'testing'])
    parser.add_argument('--prob_threshold', type=float, default=0.5)
    parser.add_argument('--dist_threshold', type=float, default=2)
    parser.add_argument('--num_scenarios', type=int, default=-1)
    parser.add_argument('--thresh_iters', type=int, default=9)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--shard_idx', type=int, nargs='+',required=True)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--num_components', type=int, default=2)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--hist_only', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use_interp', action='store_true')
    parser.add_argument('--run_analysis', action='store_true')
    parser.add_argument('--scale_features', action='store_true')
    args = parser.parse_args()

    VISUAL_OUT_SUBDIR = os.path.join(VISUAL_OUT_DIR, args.split, f"{__file__.split('.')[0]}")
    os.makedirs(VISUAL_OUT_SUBDIR, exist_ok=True)
    
    cache_path = os.path.join(CACHE_DIR, args.split)
    infos = load_infos(
        base_path=args.base_path, cache_path=cache_path, split=args.split, use_extrap=args.hist_only, 
        shard_idx=args.shard_idx, shards=args.shards, num_scenarios=args.num_scenarios,
        load_lane_cache=False, load_conflict_points=False, load_interp_trajs=False
    )
    metas, inputs = infos['metas'], infos['inputs']

    cache_path = os.path.join(CACHE_DIR, args.split, 'features')
    features = load_features(
        base_path=cache_path, load_frenetp=True, hist_only=args.hist_only, use_interp=args.use_interp, 
        shard_idxs=args.shard_idx, shards=args.shards, 
    )
    prefix = ('interp' if args.use_interp else '') + ('hist' if args.hist_only else '')
    
    if args.run_analysis:
        clustering_analysis(
            features['frenetp_features'], shards=args.shard_idx, tag=f"{VISUAL_OUT_SUBDIR}/{prefix}")
    else:
        # Fit train features 
        print(f"Fitting model")
        shard_str = (f'_shard{args.shard_idx[0]}-{args.shard_idx[-1]}_{args.shards}.pkl')
        cache_subdir = os.path.join(CACHE_DIR, args.split, f"{__file__.split('.')[0]}")
        os.makedirs(cache_subdir, exist_ok=True)
        suffix = ('_interp' if args.use_interp else '') + ('_hist' if args.hist_only else '') + shard_str
        cache_filepath = f"{cache_subdir}/labels_k-{args.num_clusters}_c-{args.num_components}{suffix}"
        
        full_model, hist_model = fit_wrapper(
            features['frenetp_features'], inputs, args.num_components, args.num_clusters, 
            args.shard_idx, cache_filepath=cache_filepath, tag=f"{VISUAL_OUT_SUBDIR}/{prefix}")

        # Predict labels on validation set  
        print(f"Predicting labels on val set")
        val_infos = load_infos(
            base_path=args.base_path, cache_path=os.path.join(CACHE_DIR, 'validation'), 
            split='validation', use_extrap=args.hist_only, shard_idx=[0, args.shards-1], 
            shards=args.shards, num_scenarios=-1, load_lane_cache=False, load_conflict_points=False, 
            load_interp_trajs=False
        )
        val_inputs = val_infos['inputs']

        val_features = load_features(
            base_path=os.path.join(CACHE_DIR, 'validation/features'), load_frenetp=True, 
            hist_only=args.hist_only, use_interp=args.use_interp, shard_idxs=[0, args.shards-1], 
            shards=args.shards, 
        )
        
        cache_subdir = os.path.join(CACHE_DIR, 'validation', f"{__file__.split('.')[0]}")
        os.makedirs(cache_subdir, exist_ok=True)
        cache_filepath = f"{cache_subdir}/labels_k-{args.num_clusters}_c-{args.num_components}{suffix}"
        predict_wrapper(
            full_model, hist_model, val_inputs, val_features['frenetp_features'], cache_filepath)

        # Predict labels on validation set  
        print(f"Predicting labels on test set")
        test_infos = load_infos(
            base_path=args.base_path, cache_path=os.path.join(CACHE_DIR, 'testing'), 
            split='testing', use_extrap=args.hist_only, shard_idx=[0, args.shards-1], 
            shards=args.shards, num_scenarios=-1, load_lane_cache=False, load_conflict_points=False, 
            load_interp_trajs=False
        )
        test_inputs = test_infos['inputs']

        test_features = load_features(
            base_path=os.path.join(CACHE_DIR, 'testing/features'), load_frenetp=True, 
            hist_only=args.hist_only, use_interp=args.use_interp, shard_idxs=[0, args.shards-1], 
            shards=args.shards, 
        )
        
        cache_subdir = os.path.join(CACHE_DIR, 'testing', f"{__file__.split('.')[0]}")
        os.makedirs(cache_subdir, exist_ok=True)
        cache_filepath = f"{cache_subdir}/labels_k-{args.num_clusters}_c-{args.num_components}{suffix}"
        predict_wrapper(
            full_model, hist_model, test_inputs, test_features['frenetp_features'], cache_filepath)
