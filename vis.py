import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib
matplotlib.use('Agg')

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from PIL import Image
import io


def render_flame_mesh_plotly(faces, vertices, per_vertex_color=None, color_as_intensity=False, distance_to_face=5.7, resolution=256):
    '''
    Method for specifically rendering FLAME meshes with a proper frontal camera view.
    faces: [N, 3], vertices: [K, 3]
    Outputs an image (np.array) of the rendered mesh.
    '''
    # Calculate the centroid and maximum extent to adjust the camera
    centroid = np.mean(vertices, axis=0)
    max_extent = np.max(vertices, axis=0)
    min_extent = np.min(vertices, axis=0)
    size = max_extent - min_extent
    max_size = np.max(size)
    
    # Set camera distance to ensure the entire mesh is visible
    camera_distance = distance_to_face * max_size
    
    # Configure camera for frontal view
    camera = dict(
        eye=dict(x=centroid[0], y=centroid[1], z=centroid[2] + camera_distance),
        up=dict(x=0, y=1, z=0),
        center=dict(x=centroid[0], y=centroid[1], z=centroid[2]),
    )
    
    if per_vertex_color is not None or not color_as_intensity:
        lighting = dict(
                ambient=0.3,  # Brighter ambient light to reduce harsh shadows
                diffuse=0.9,  # Stronger direct light for surface contrast
                specular=0.8,  # Increased highlights to emphasize curves
                roughness=0.5,  # Smoother highlights (lower = shinier)
                fresnel=0.2  # Enhances edges/contours
            )
        lightposition = dict(x=100, y=200, z=100)
    else:
        lighting = None
        lightposition = None

    # Create the Mesh3d trace
    mesh_args = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=1.0,
        lighting=lighting,
        lightposition=lightposition,  # Light source position
        flatshading=False,
    )

    if per_vertex_color is not None:
        if not color_as_intensity:
            mesh_args['vertexcolor'] = per_vertex_color
        else:
            mesh_args['intensity'] = per_vertex_color
            mesh_args['colorscale'] = 'RdBu'
    else:
        mesh_args['color'] = '#4980b9'
        
    mesh_trace = go.Mesh3d(**mesh_args)
    
    # Create figure and configure layout
    fig = go.Figure(data=[mesh_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera,
            aspectmode='data',
            bgcolor='white'
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, b=0, t=0)
    )


    try:
        # Render the figure to an image array
        img_bytes = pio.to_image(fig, format='png', width=resolution, height=resolution)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
    except Exception as e:
        return fig
    return img_array


def plot_3d_point_cloud(x, y, z, title="3D Point Cloud", enumerate_points=False):
    """
    Plots a 3D point cloud using Plotly.
    
    Parameters:
    - x, y, z: Lists or arrays of coordinates.
    - title: Title of the plot.
    """
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+text',
        text=list(range(len(x))) if enumerate_points else '',   
        marker=dict(
            size=4,
            color=z,                # Color points based on z-values
            colorscale='Viridis',   # Choose a colorscale
            opacity=0.8
        )
    )])
    
    # Update layout with titles and axis labels
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    
    # Display the plot
    fig.show()


def render_mesh_plotly(faces, vertices, per_vertex_color=None, color_as_intensity=False, color='#4980b9', show=True, width=800, height=800, background='white', show_grid=False, grid_color='#222222', grid_width=1.0, grid_opacity=0.5):
    """
    Render a triangular mesh in Plotly with an automatically positioned camera.

    Parameters:
    - faces: (F, 3) np.ndarray of triangle indices (int)
    - vertices: (V, 3) np.ndarray of vertex coordinates (float)
    - per_vertex_color: optional per-vertex colors. If array of shape (V, 3 or 4), used as RGB(A) vertex colors.
      If 1D of length V and color_as_intensity=True, used as intensity with a colorscale.
    - color_as_intensity: if True and per_vertex_color is 1D, maps to colorscale instead of RGB.
    - color: fallback solid color when per_vertex_color is None.
    - show: whether to call fig.show() (default True)
    - width/height: figure size in pixels
    - background: scene and paper background color
    - show_grid: if True, overlays a wireframe grid of triangle edges
    - grid_color: color of the wireframe lines
    - grid_width: width of the wireframe lines
    - grid_opacity: opacity of the wireframe lines

    Returns:
    - Plotly Figure containing the mesh
    """
    faces = np.asarray(faces)
    vertices = np.asarray(vertices)

    # Compute a centered, padded cube range so the mesh is fully in view
    min_extent = np.min(vertices, axis=0)
    max_extent = np.max(vertices, axis=0)
    center = (min_extent + max_extent) / 2.0
    size = max_extent - min_extent
    max_size = float(np.max(size)) if np.all(np.isfinite(size)) else 1.0
    if not np.isfinite(max_size) or max_size == 0.0:
        max_size = 1.0
    pad = 0.05 * max_size
    half_range = 0.5 * max_size + pad

    x_range = [center[0] - half_range, center[0] + half_range]
    y_range = [center[1] - half_range, center[1] + half_range]
    z_range = [center[2] - half_range, center[2] + half_range]

    # Prepare mesh arguments
    mesh_args = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0].astype(int),
        j=faces[:, 1].astype(int),
        k=faces[:, 2].astype(int),
        opacity=1.0,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=200, z=100),
    )

    if per_vertex_color is not None:
        pvc = np.asarray(per_vertex_color)
        if pvc.ndim == 2 and pvc.shape[0] == vertices.shape[0] and pvc.shape[1] in (3, 4) and not color_as_intensity:
            mesh_args['vertexcolor'] = pvc
        elif pvc.ndim == 1 and pvc.shape[0] == vertices.shape[0] and color_as_intensity:
            mesh_args['intensity'] = pvc
            mesh_args['colorscale'] = 'Viridis'
        else:
            mesh_args['color'] = color
    else:
        mesh_args['color'] = color

    mesh_trace = go.Mesh3d(**mesh_args)

    # Default 3/4 view camera with distance scaled to mesh size
    distance = max(1.0, 2.5 * half_range)
    norm = np.sqrt(3.0)
    camera = dict(
        eye=dict(x=distance / norm, y=distance / norm, z=distance / norm),
        up=dict(x=0, y=1, z=0),
    )

    fig = go.Figure(data=[mesh_trace])

    # Optional wireframe grid (unique triangle edges)
    if show_grid:
        faces_int = faces.astype(int)
        unique_edges = set()
        for tri in faces_int:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                if a == b:
                    continue
                e = (a, b) if a < b else (b, a)
                unique_edges.add(e)

        xs, ys, zs = [], [], []
        for a, b in unique_edges:
            va = vertices[a]
            vb = vertices[b]
            xs.extend([va[0], vb[0], None])
            ys.extend([va[1], vb[1], None])
            zs.extend([va[2], vb[2], None])

        grid_trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=grid_color, width=grid_width),
            opacity=grid_opacity,
            hoverinfo='skip',
            showlegend=False,
        )
        fig.add_trace(grid_trace)
    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(visible=False, range=x_range),
            yaxis=dict(visible=False, range=y_range),
            zaxis=dict(visible=False, range=z_range),
            camera=camera,
            aspectmode='data',
            bgcolor=background,
        ),
        paper_bgcolor=background,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    if show:
        fig.show()
    return fig