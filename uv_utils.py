import torch
import numpy as np

def uv_sample_uniform(num_points, uv_bounds=[0, 1, 0, 1]):
    '''
        Naive sampling
    '''
    u_min, u_max, v_min, v_max = uv_bounds

    grid_size = int(np.sqrt(num_points))
    u_vals = torch.linspace(u_min, u_max, grid_size)
    v_vals = torch.linspace(v_min, v_max, grid_size)
    U, V = torch.meshgrid(u_vals, v_vals, indexing='ij')
    grid_points = torch.stack([U.flatten(), V.flatten()], axis=1)

    sampled_uvs = grid_points[:num_points]
    return sampled_uvs

def uv_sample_per_face(verts_uvs, faces_texture_idx, K):
    F = faces_texture_idx.shape[0]
    face_uvs = verts_uvs[faces_texture_idx]
    rand = torch.rand(F, K, 2)
    u = rand[:, :, 0]
    v = rand[:, :, 1]
    is_outside = u + v > 1
    u[is_outside] = 1 - u[is_outside]
    v[is_outside] = 1 - v[is_outside]
    sampled_points = (face_uvs[:, 0, :].unsqueeze(1) * (1 - u - v).unsqueeze(2) +
                      face_uvs[:, 1, :].unsqueeze(1) * u.unsqueeze(2) +
                      face_uvs[:, 2, :].unsqueeze(1) * v.unsqueeze(2))
    sampled_points = sampled_points.view(-1, 2)
    return sampled_points

def uv_to_faces(uv_verts, uv_bounds=[0, 1, 0, 1], face_lut=None, how='LUT'):
    '''
        uv_verts: [B, N, 2]
        
        output: [B, N]
    '''
    if how == 'LUT':
        def scale_uv_to_grid(uv, u_min, u_max, v_min, v_max, resolution):
            u_scaled = (uv[..., 0] - u_min) / (u_max - u_min) * (resolution - 1)
            v_scaled = (uv[..., 1] - v_min) / (v_max - v_min) * (resolution - 1)
            return torch.stack([u_scaled, v_scaled], axis=-1)

        res = face_lut.shape[0]

        if isinstance(uv_bounds, torch.Tensor):
            uv_bounds = uv_bounds.tolist()

        pts = scale_uv_to_grid(uv_verts, *uv_bounds, resolution=res)
        pts = torch.round(pts).long()
        pts = torch.clip(pts, 0, res - 1)

        face_indices = face_lut[pts[..., 1], pts[..., 0]]

        return face_indices
    else: 
        # also possible via vertical ray casting on uv_faces
        raise NotImplemented()


def get_barycentric(points, triangles, return_validity_mask=False):
    '''
        Accepts points within corresponding triangles
        points: [B, N, 2]
        trinagles: [B, N, 3, 2]

        output: [B, N, 3]
    '''
    A = triangles[..., 0, :]  
    B = triangles[..., 1, :]  
    C = triangles[..., 2, :]  
    v0 = B - A              
    v1 = C - A              
    v2 = points - A        
    denom = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]  

    # Avoid division by zero
    # denom = torch.where(denom == 0, 1e-8, denom)
    assert denom.abs().min() != 0 # assume that uv triangles are not degenerate

    u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / denom
    v = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / denom
    w = 1 - u - v

    result = torch.stack([u, v, w], axis=-1) 

    if return_validity_mask:
        eps = 1e-6
        validity_mask = (
            (u >= -eps) &
            (v >= -eps) &
            (w >= -eps) &
            (u <= 1 + eps) &
            (v <= 1 + eps) &
            (w <= 1 + eps)
        )
        return result, validity_mask
    return result