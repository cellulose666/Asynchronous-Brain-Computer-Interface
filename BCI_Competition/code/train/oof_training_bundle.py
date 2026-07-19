"""Bundle utilities were removed.

The simplified pipeline writes one OOF-ready NPZ file and trains directly from it.
Use:
  python BCI_Competition/code/preprocessing/build_oof_windows.py --subjects 1
  python BCI_Competition/code/train/train_hierarchical_oof.py --subjects 1
"""
