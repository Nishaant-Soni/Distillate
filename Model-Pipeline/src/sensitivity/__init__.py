"""Sensitivity analysis of the selected student: hyperparameters, tokens, length, obfuscation."""

import os

# PyTorch only. SHAP's explainers probe transformers' TensorFlow classes, which fails on images
# with TensorFlow and Keras 3. It must be set before transformers loads.
os.environ.setdefault("USE_TF", "0")
