from plyfile import PlyData, PlyElement
import numpy as np
from collections import defaultdict


def load_ply_mesh(path):
    plydata = PlyData.read(path)
    
    if len(plydata.elements) != 2:
        print("Ply contains more than 2 elements! Are you sure this is a simple mesh?")
        print(plydata.elements)
    
    points = plydata.elements[0]
    faces = plydata.elements[1]
    
    xyz = np.stack((np.asarray(points["x"]),
                    np.asarray(points["y"]),
                    np.asarray(points["z"])),  axis=1)
    faces = np.array([face for face in faces['vertex_indices']])

    return xyz, faces

def load_ply_points(path):
    plydata = PlyData.read(path)
    
    points = plydata['vertex']
    
    xyz = np.stack((np.asarray(points["x"]),
                    np.asarray(points["y"]),
                    np.asarray(points["z"])),  axis=1)

    return xyz


def load_flame_template(path, use_pytorch3d=False):
    if use_pytorch3d:
        from pytorch3d.io import load_ojb
        vert, faces, aux = load_obj(path)
        verts_uvs = aux.verts_uvs
        vert_faces = faces.verts_idx
        uv_faces = faces.textures_idx
    else:
        import torch
        vert, uv_faces, vert_faces, verts_uvs = torch.load(path.replace('.obj', '__torch.pt'))
    
    # make uv –> 3d vertex mapping
    texture_indices = uv_faces.view(-1)  
    vertex_indices = vert_faces.view(-1)      

    uv_to_vertices = defaultdict(set)

    for tex_idx, vert_idx in zip(texture_indices.tolist(), vertex_indices.tolist()):
        uv_to_vertices[tex_idx].add(vert_idx)
    
    uv_to_vertices = dict(uv_to_vertices)
    max_vertices_per_uv = max(len(vertices) for vertices in uv_to_vertices.values())

    if max_vertices_per_uv == 1:
        uv_to_vertices = {k:list(v)[0] for k, v in uv_to_vertices.items()}

        l = []
        for i in range(len(uv_to_vertices)) :
            l.append(uv_to_vertices[i])
        uv_to_vertices = l # return just a list
    else:
        uv_to_vertices = {k:list(v) for k, v in uv_to_vertices.items()}
    
    return vert, vert_faces, verts_uvs, uv_faces, uv_to_vertices