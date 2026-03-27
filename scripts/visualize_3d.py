import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import sys
import os

def load_input_file(filename):
    """Loads input file with n, k, T header and returns (x, y, z, intensity)."""
    try:
        data = np.loadtxt(filename, skiprows=3)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    except Exception as e:
        print(f"Error loading input file {filename}: {e}")
        return None

def load_output_file(filename):
    """Loads output file with x, y, z, intensity and returns (x, y, z, intensity)."""
    try:
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    except Exception as e:
        print(f"Error loading output file {filename}: {e}")
        return None

def create_visualization(input_file, knn_file, approx_file, kmeans_file, output_html):
    # Load data
    orig = load_input_file(input_file)
    knn = load_output_file(knn_file)
    approx = load_output_file(approx_file)
    kmeans = load_output_file(kmeans_file)

    if orig is None:
        print("Failed to load original data. Exiting.")
        return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Original Data", "KNN Equalized", 
                        "Approx-KNN Equalized", "KMeans Equalized")
    )

    def add_trace(data, name, row, col):
        if data is not None:
            x, y, z, intensity = data
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=intensity,
                        colorscale=[[0, 'black'], [0.5, 'green'], [1, 'white']],
                        opacity=0.8,
                        colorbar=dict(thickness=10, x=1.1 if col == 2 else -0.1)
                    ),
                    name=name
                ),
                row=row, col=col
            )

    add_trace(orig, "Original", 1, 1)
    add_trace(knn, "KNN", 1, 2)
    add_trace(approx, "Approx-KNN", 2, 1)
    add_trace(kmeans, "KMeans", 2, 2)

    # Update layout to be more premium
    fig.update_layout(
        title_text="Intensity Equalization Comparison (3D Point Cloud)",
        title_x=0.5,
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        scene3=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        scene4=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        height=900,
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark"
    )

    # Add custom JavaScript to synchronize the 3D scenes
    import plotly.io as pio
    
    # We use a fixed div_id to easily target the plot in JS
    plot_div_id = '3d-visualization-plot'
    
    # Custom JS to sync cameras
    sync_js = """
    var gd = document.getElementById('%PLOT_ID%');
    var scenes = ['scene', 'scene2', 'scene3', 'scene4'];
    var relayouting = false;
    
    gd.on('plotly_relayout', function(edata) {
        if (relayouting) return;
        
        var update = {};
        var foundCamera = false;
        
        scenes.forEach(function(s) {
            var camKey = s + '.camera';
            if (edata[camKey]) {
                foundCamera = true;
                scenes.forEach(function(targetS) {
                    if (s !== targetS) {
                        update[targetS + '.camera'] = edata[camKey];
                    }
                });
            }
        });
        
        if (foundCamera) {
            relayouting = true;
            Plotly.relayout(gd, update).then(() => { relayouting = false; });
        }
    });
    """.replace('%PLOT_ID%', plot_div_id)

    html_str = pio.to_html(fig, full_html=True, include_plotlyjs='cdn', 
                           post_script=sync_js, div_id=plot_div_id)
    
    with open(output_html, "w") as f:
        f.write(html_str)
        
    print(f"Visualization saved to {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D point cloud intensity equalization.")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--knn", required=True, help="Path to KNN output file")
    parser.add_argument("--approx", required=True, help="Path to Approx-KNN output file")
    parser.add_argument("--kmeans", required=True, help="Path to KMeans output file")
    parser.add_argument("--output", default="visualization.html", help="Path to save output HTML")

    args = parser.parse_args()

    # Check file existence
    for f in [args.input, args.knn, args.approx, args.kmeans]:
        if not os.path.exists(f):
            print(f"Error: File {f} does not exist.")
            sys.exit(1)

    create_visualization(args.input, args.knn, args.approx, args.kmeans, args.output)
