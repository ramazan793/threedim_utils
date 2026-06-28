import os
import open3d as o3d
import pickle
import numpy as np
import trimesh

import torch

# Common root for sibling repos / shared assets. Mirrors src/gghead/env.py so that a
# single environment variable can relocate everything when porting hosts.
_AGORA_COMMON_PREFIX = os.environ.get('AGORA_COMMON_PREFIX', '/data2/ramazan.fazylov/')
if not _AGORA_COMMON_PREFIX.endswith('/'):
    _AGORA_COMMON_PREFIX = _AGORA_COMMON_PREFIX + '/'

FLAME2020_MASK_PATH = f"{_AGORA_COMMON_PREFIX}code/flame/FLAME2020/FLAME2020_initial/FLAME_masks.pkl"
FLAME2020_TEMPLATE_PATH = f"{_AGORA_COMMON_PREFIX}code/flame/FLAME2020/head_template_mesh.obj"

def load_flame_masks(mask_path=FLAME2020_MASK_PATH):
    with open(mask_path, 'rb') as f:
        flame_masks = pickle.load(f, encoding='latin1')
    return flame_masks


def query_flame_by_mask(xyz, faces, mask_parts, flame_masks, method='exclude'):
    '''
        xyz: (N, 3)
        faces: (K, 3)
        mask_parts: list of parts to mask. i.e. 'boundary', 'neck', 'right_eyeball', 'left_eyeball', etc.
        method: 'exclude' or 'include'
    '''
    mask = set().union(*[flame_masks[part] for part in mask_parts])

    if method == 'exclude':
        masked_idx = list(set(range(xyz.shape[0])) - mask)
    elif method == 'include':
        masked_idx = list(mask)
    else:
        raise ValueError(f"Invalid method, allowed: exclude, include")
    
    masked_xyz = xyz[masked_idx]

    valid_faces_mask = np.all(np.isin(faces, masked_idx), axis=1)
    filtered_faces = faces[valid_faces_mask]

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(masked_idx)}
    filtered_faces_reindex = np.array([[old_to_new[idx] for idx in face] for face in filtered_faces], dtype=np.int32)

    return masked_xyz, filtered_faces_reindex, masked_idx

def save_template_as_torch(path):
    from pytorch3d import load_obj
    vert, faces, aux = load_obj('/home/r.fazylov/research_workspace/flame/FLAME2020/head_template_mesh.obj')
    torch.save((vert, faces.textures_idx, faces.verts_idx, aux.verts_uvs), '/home/r.fazylov/research_workspace/flame/FLAME2020/head_template_mesh__torch.pt')


def numpy_fix():
    import numpy as np_fixed
    np_fixed.bool = np_fixed.bool_
    np_fixed.int = np_fixed.int_
    np_fixed.float = np_fixed.float_
    np_fixed.complex = np_fixed.complex_
    np_fixed.object = np_fixed.object_
    np_fixed.unicode = np_fixed.unicode_
    np_fixed.str = np_fixed.str_
    return np_fixed


