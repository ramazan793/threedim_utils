from argparse import Namespace

_cfg = {
    'flame_model_path' : '/home/r.fazylov/research_workspace/flame/FLAME2020/FLAME2020_initial/generic_model.pkl',
    'n_shape' : 300,
    'n_exp' : 100,
    'flame_lmk_embedding_path' : '/home/r.fazylov/research_workspace/flame/FLAME2020/landmark_embedding.npy'
}

DEFAULT_CONFIG = Namespace(**_cfg)