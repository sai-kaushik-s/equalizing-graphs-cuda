import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def parse_time(t_str):
    if pd.isna(t_str) or t_str == '-':
        return np.nan
    t_str = str(t_str).strip()
    if t_str.endswith('µs'):
        return float(t_str.replace('µs', '').strip()) / 1000.0
    elif t_str.endswith('ms'):
        return float(t_str.replace('ms', '').strip())
    elif t_str.endswith('s'):
        return float(t_str.replace('s', '').strip()) * 1000.0
    return float(t_str) if t_str else np.nan

def main():
    os.makedirs('plots', exist_ok=True)

    COLOR_MONO_LIGHT = '#bdc3c7'
    COLOR_MONO_MID = '#95a5a6'
    COLOR_MONO_DARK = '#2c3e50'
    COLOR_GREEN_ACCENT = '#27ae60'

    df = pd.read_csv('benchmark_results.csv')
    df['gpu_time_ms'] = df['gpu_time'].apply(parse_time)
    df['cpu_time_ms'] = df['cpu_time'].apply(parse_time)
    df['t'] = df['t'].replace('-', np.nan).astype(float)

    # ---------------------------------------------------------
    # Plot 1: GPU Execution Time vs Point Cloud Size (N)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_knn_n = df[(df['k'] == 16) & (df['precision'] == 'float') & (df['algo'].isin(['knn', 'approx_knn']))]
    palette_knn = {'approx_knn': COLOR_GREEN_ACCENT, 'knn': COLOR_MONO_MID}

    sns.lineplot(data=df_knn_n, x='n', y='gpu_time_ms', hue='algo', marker='o', linewidth=2.5, palette=palette_knn)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('GPU Execution Time vs Point Cloud Size (N)\nExact KNN vs Approximate KNN (K=16, Float)', fontsize=14)
    plt.xlabel('Number of Points (N)', fontsize=12)
    plt.ylabel('GPU Time (ms) [Log Scale]', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Approximate KNN (Accent)', 'Exact KNN'], title='Algorithm')
    plt.tight_layout()
    plt.savefig('plots/knn_scaling_N.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Hardware Speedup (CPU vs GPU) for k-Means
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_kmeans = df[(df['algo'] == 'kmeans') & (df['precision'] == 'float') & (df['k'] == 32) & (df['t'] == 50.0)].copy()
    df_kmeans['actual_speedup'] = df_kmeans['cpu_time_ms'] / df_kmeans['gpu_time_ms']
    df_kmeans = df_kmeans.dropna(subset=['actual_speedup'])

    colors = [COLOR_MONO_MID if i < len(df_kmeans)-1 else COLOR_GREEN_ACCENT for i in range(len(df_kmeans))]
    sns.barplot(data=df_kmeans, x='n', y='actual_speedup', palette=colors)
    plt.title('Hardware Speedup: CPU vs GPU\nk-Means (K=32, 50 Iterations, Float)', fontsize=14)
    plt.xlabel('Number of Points (N)', fontsize=12)
    plt.ylabel('Speedup Factor (CPU Time / GPU Time)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/kmeans_cpu_gpu_speedup.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Algorithm Sensitivity to K
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_k_scale = df[(df['n'] == 100000) & (df['precision'] == 'float')].copy()
    df_k_scale = df_k_scale[((df_k_scale['algo'] != 'kmeans') | (df_k_scale['t'] == 50.0))]
    palette_algos = {'approx_knn': COLOR_GREEN_ACCENT, 'knn': COLOR_MONO_MID, 'kmeans': COLOR_MONO_DARK}

    sns.lineplot(data=df_k_scale, x='k', y='gpu_time_ms', hue='algo', marker='s', linewidth=2.5, palette=palette_algos)
    plt.yscale('log')
    plt.title('Algorithm Sensitivity to K\n(N=100,000 Points, Float)', fontsize=14)
    plt.xlabel('Number of Neighbors/Clusters (K)', fontsize=12)
    plt.ylabel('GPU Time (ms) [Log Scale]', fontsize=12)
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig('plots/scaling_with_K.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 4: Precision Impact (Int vs Float)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_prec = df[(df['n'] == 100000) & (df['k'].isin([4, 16, 32, 64])) & (df['algo'] == 'approx_knn')].copy()
    palette_prec = {'int': COLOR_MONO_MID, 'float': COLOR_GREEN_ACCENT}

    sns.barplot(data=df_prec, x='k', y='gpu_time_ms', hue='precision', palette=palette_prec)
    plt.title('Precision Impact: Int vs Float GPU Execution Time\nApproximate KNN (N=100,000)', fontsize=14)
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('GPU Time (ms)', fontsize=12)
    plt.legend(title='Precision')
    plt.tight_layout()
    plt.savefig('plots/precision_int_vs_float.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 5: Algorithmic Speedup (Exact vs Approx KNN)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df_algo_speedup = df_knn_n.pivot_table(index='n', columns='algo', values='gpu_time_ms').reset_index()
    df_algo_speedup['algorithmic_speedup'] = df_algo_speedup['knn'] / df_algo_speedup['approx_knn']

    sns.lineplot(data=df_algo_speedup, x='n', y='algorithmic_speedup', marker='D', color=COLOR_GREEN_ACCENT, linewidth=2.5)
    plt.xscale('log')
    plt.title('Algorithmic Speedup: Approximate vs Exact KNN\n(K=16, Float)', fontsize=14)
    plt.xlabel('Number of Points (N) [Log Scale]', fontsize=12)
    plt.ylabel('Speedup Factor (Exact Time / Approx Time)', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/algorithmic_speedup.png', dpi=300)
    plt.close()

    print("Successfully generated all plots in 'plots/' directory with monotone/green-accent color palette.")

if __name__ == "__main__":
    main()