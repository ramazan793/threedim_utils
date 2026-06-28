import open3d as o3d
import pickle
import numpy as np
import trimesh

from .convert import open3d_to_trimesh, trimesh_to_open3d

def subdivide_trimesh(vertices, faces, iterations=1):
    '''
        Subdivide the mesh by a given number of iterations.
        vertices: (N, 3)
        faces: (K, 3)
        iterations: int
    '''
    for _ in range(iterations):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    return vertices, faces

def subdivide_open3d(o3d_mesh, iterations=1):
    tri_mesh = open3d_to_trimesh(o3d_mesh)
    vert, faces = subdivide_trimesh(tri_mesh.vertices, tri_mesh.faces, iterations=iterations)
    subdivided_o3d_mesh = trimesh_to_open3d(vertices=vert, faces=faces)
    return subdivided_o3d_mesh
    