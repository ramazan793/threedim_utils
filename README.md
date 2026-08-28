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

`assets/flame_with_mouth_no_backhead_v3/FLAME.pkl` is FLAME 2020 with a mouth-interior mesh added and the
back of the head removed. It is derived from FLAME, so the
[FLAME license](https://flame.is.tue.mpg.de/modellicense.html) forbids redistributing it. Download FLAME 2020
from <https://flame.is.tue.mpg.de/> and build it:

```bash
python build_flame_with_mouth.py /path/to/FLAME2020/generic_model.pkl   # default: ../smirk/assets/FLAME2020/generic_model.pkl
```

This reproduces the pickle the AGORA checkpoints were trained with. All other assets are included.
