# threedim_utils

Small utility library for FLAME-based 3D head processing: a FLAME 2020 linear-blend-skinning
implementation (`flame2020_lbs/`), mesh/UV helpers, and a few precomputed UV masks / vertex mappings.

It is a runtime companion of **[AGORA](https://github.com/ramazan793/AGORA)** — *Adversarial Generation
Of Real-time Animatable 3D Gaussian Head Avatars*. AGORA clones this repo into its
`dependencies/threedim_utils/` and imports:

- `flame2020_lbs.FLAME` — FLAME LBS forward model (shape / expression / pose → posed vertices)
- `flame.query_flame_by_mask`, `flame.load_flame_masks` — FLAME region masks
- assets under `assets/flame_with_mouth_no_backhead_v3/` — the with-mouth UV template, the
  `uv_to_3d__vert_idx_mapping.pt`, and the facial / mouth UV masks used by AGORA's deformation branch.

## Modules

- `flame2020_lbs/` — FLAME 2020 LBS (`FLAME.py`, `lbs.py`).
- `flame.py`, `load.py` — load FLAME models and region masks.
- `mesh_utils.py`, `uv_utils.py`, `convert.py`, `vis.py` — mesh / UV / OBJ helpers and visualization.

## FLAME assets (NOT included — license-restricted)

The actual FLAME 2020 model files are **not committed** here: the
[FLAME license](https://flame.is.tue.mpg.de/modellicense.html) forbids redistribution. Download FLAME
2020 from the official site — <https://flame.is.tue.mpg.de/> — then obtain/derive the following files and
place them at these paths inside the repo:

```
assets/
  flame_with_mouth_no_backhead_v3/FLAME.pkl    # required by AGORA (use_flame_template_with_mouth)
  flame_with_mouth_no_backhead_v2/FLAME.pkl    # v2 variant
  F_flame_plus.pt                              # FLAME faces / topology
  V_flame_plus.pt                              # FLAME template vertices
  W_flame_plus.pt                              # FLAME LBS skinning weights
```

Everything else (UV masks, `uv_to_3d__vert_idx_mapping.pt`, stretched-UV / no-backhead templates) is
already included in this repo.
