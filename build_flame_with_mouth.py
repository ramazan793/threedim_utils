"""Build assets/flame_with_mouth_no_backhead_v3/FLAME.pkl from the official FLAME 2020 generic_model.pkl:
adds a mouth-interior (cavity + teeth) mesh to FLAME and removes the back of the head.

    python build_flame_with_mouth.py [/path/to/FLAME2020/generic_model.pkl]
"""
import argparse
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BACKHEAD_FACES = f"{HERE}/assets/flame_with_mouth_no_backhead/head_template_mesh__without_backhead/flame_mask_faces.npy"
OUT = f"{HERE}/assets/flame_with_mouth_no_backhead_v3/FLAME.pkl"

# inner lip contours of the FLAME template, left corner -> right corner
UPPER = [1572, 1594, 1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731, 2730, 2708]
LOWER = [1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708]
CORNERS = [1572, 2708]
M, K, KT = 32, 16, 3            # samples along a lip / lip->throat depth / rows per teeth strip
THROAT_Z, TEETH_H = 0.02, 0.49  # throat line offset behind the lip corners / teeth height as fraction of lip gap
U_SLICE, L_SLICE = slice(8, -8), slice(4, -4)


def load_flame2020(path):
    for n in ("bool", "int", "float", "complex", "object", "unicode", "str"):  # chumpy needs the old numpy aliases
        setattr(np, n, getattr(np, n + "_"))
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    t = lambda x: torch.tensor(np.array(x.todense() if sp.issparse(x) else x, dtype=np.float32))
    sd = t(d["shapedirs"])
    parents = np.array(d["kintree_table"][0], dtype=np.int64); parents[0] = -1
    return dict(f=torch.tensor(np.array(d["f"], dtype=np.int64)), v=t(d["v_template"]), weights=t(d["weights"]),
                shapedirs=torch.cat([sd[:, :, :300], sd[:, :, 300:350]], 2),
                posedirs=t(d["posedirs"]), J=t(d["J_regressor"]), kintree=parents[None])


def resample(x, m):
    return F.interpolate(x.T[None], size=m, mode="linear", align_corners=True)[0].T


def between(a, b, k):
    r = torch.linspace(0, 1, steps=k).view(k, 1, 1)
    return a[None] * (1 - r) + b[None] * r


def chain(x, m):
    t = torch.linspace(0, x.shape[0] - 1, steps=m)
    i0 = torch.floor(t).long(); i1 = torch.clamp(i0 + 1, max=x.shape[0] - 1)
    a = (t - i0.float()).view(m, *[1] * (x.dim() - 1))
    return (1 - a) * x[i0] + a * x[i1]


def grid_mesh(g):
    Kg, Mg = g.shape[:2]
    r, c = torch.meshgrid(torch.arange(Kg - 1), torch.arange(Mg - 1), indexing="ij")
    v00 = (r * Mg + c).reshape(-1); v10, v01, v11 = v00 + Mg, v00 + 1, v00 + Mg + 1
    return g.reshape(-1, 3), torch.stack([torch.stack([v00, v10, v01], 1), torch.stack([v10, v11, v01], 1)], 1).reshape(-1, 3)


def blend(xu, xl):  # lip value at the lips -> 50/50 mix at the throat line
    au = torch.linspace(0, 0.5, steps=K).view(K, *[1] * xu.dim())
    al = torch.linspace(0, 0.5, steps=K - 1).view(K - 1, *[1] * xu.dim())
    up = (1 - au) * xu[None] + au * xl[None]
    lo = ((1 - al) * xl[None] + al * xu[None]).flip(0)
    return torch.cat([up, lo]).reshape(-1, *xu.shape[1:])


def ramp(xu, xl):  # lip value at the lips -> zero at the throat line
    ru = torch.linspace(1, 0, steps=K).view(K, *[1] * xu.dim())
    rl = torch.linspace(0, 1, steps=K - 1).view(K - 1, *[1] * xu.dim())
    return torch.cat([ru * xu[None], rl * xl[None]]).reshape(-1, *xu.shape[1:])


def tile(x, s):
    return x[s][None].expand(KT, *x[s].shape).reshape(-1, *x.shape[1:])


def build(flame):
    v, V0 = flame["v"], flame["v"].shape[0]

    up, lo, throat = v[UPPER], v[LOWER], v[CORNERS].clone()
    throat[:, 2] -= THROAT_Z
    up, lo, throat = resample(up, M), resample(lo, M), resample(throat, M)
    cavity = torch.cat([between(up, throat, K), between(throat, lo, K)[1:]])
    gap = (up[M // 2] - lo[M // 2])[1:2]
    up_teeth, lo_teeth = up.clone(), lo.clone()
    up_teeth[:, 1] -= gap * TEETH_H; lo_teeth[:, 1] += gap * TEETH_H
    grids = [cavity, between(up[U_SLICE], up_teeth[U_SLICE], KT), between(lo[L_SLICE], lo_teeth[L_SLICE], KT)]

    parts = [grid_mesh(g) for g in grids]
    offsets = np.cumsum([0] + [p[0].shape[0] for p in parts[:-1]])
    v_mouth = torch.cat([p[0] for p in parts])
    f_mouth = torch.cat([p[1] + int(o) for p, o in zip(parts, offsets)])
    f_plus = torch.cat([flame["f"], f_mouth + V0])
    v_plus = torch.cat([v, v_mouth])

    curves = lambda x: (chain(x[UPPER], M), chain(x[LOWER], M))
    mouth = lambda cav, xu, xl: torch.cat([cav, tile(xu, U_SLICE), tile(xl, L_SLICE)])
    wu, wl = [w / (w.sum(1, keepdim=True) + 1e-8) for w in curves(flame["weights"])]
    w_cav = blend(wu, wl); w_cav = w_cav / w_cav.sum(1, keepdim=True)
    su, sl = curves(flame["shapedirs"])
    pu, pl = curves(flame["posedirs"])
    w_plus = torch.cat([flame["weights"], mouth(w_cav, wu, wl)])
    sd_plus = torch.cat([flame["shapedirs"], mouth(ramp(su, sl), su, sl)])
    pd_plus = torch.cat([flame["posedirs"], mouth(blend(pu, pl), pu, pl)])
    J_plus = torch.cat([flame["J"], torch.zeros(flame["J"].shape[0], v_mouth.shape[0])], 1)

    F0, bh = flame["f"].shape[0], np.load(BACKHEAD_FACES)
    f_keep = torch.cat([f_plus[:F0][torch.from_numpy(np.setdiff1d(np.arange(F0), bh))], f_plus[F0:]])
    used = torch.unique(f_keep)
    old2new = torch.full((v_plus.shape[0],), -1, dtype=torch.long); old2new[used] = torch.arange(used.shape[0])
    removed = torch.from_numpy(np.setdiff1d(np.arange(v_plus.shape[0]), used.numpy()))

    return {
        "f": old2new[f_keep].numpy(),
        "v_template": v_plus[used].numpy(),
        "shapedirs": sd_plus[used].numpy(),
        "posedirs": pd_plus[used].numpy(),
        "J_regressor": sp.csr_matrix(J_plus[:, used].numpy()),
        "kintree_table": flame["kintree"],
        "weights": w_plus[used].numpy(),
        "bh_vert": v_plus[removed].numpy(),  # removed back-head verts, kept for the J_regressor
        "bh_vert_J_regressor": J_plus[:, removed].numpy(),
        "bh_vert_shapedirs": sd_plus[removed].numpy(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flame2020", nargs="?", default=f"{HERE}/../smirk/assets/FLAME2020/generic_model.pkl")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    model = build(load_flame2020(args.flame2020))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f, protocol=2)
    print(f"wrote {args.out}")
