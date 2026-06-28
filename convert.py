import open3d as o3d
import pickle
import numpy as np
import trimesh


def np_to_open3d(xyz, faces):
    '''
        xyz: (N, 3)
        faces: (K, 3)
    '''
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

def open3d_to_trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    return tri_mesh

def trimesh_to_open3d(vertices=None, faces=None, tri_mesh=None):
    if tri_mesh is not None:
        vertices = tri_mesh.vertices
        faces = tri_mesh.faces

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh