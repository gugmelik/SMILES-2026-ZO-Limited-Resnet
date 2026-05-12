"""
head_init.py — Initialization of the new CIFAR100 classification head.

Strategy
--------
The downstream optimiser (see ``zo_optimizer.py``) replaces the head with
a closed-form linear-probe solution computed from cached features on its
very first ``.step()``. Consequently the role of this initialiser is
limited to providing a numerically benign starting point that:

  * keeps the initial cross-entropy near the uniform baseline log(K)≈4.61
    so that the first ``loss_fn`` evaluation does not explode, and
  * does not bias the optimiser toward any particular class.

We use **small-scale Xavier uniform** for the weight matrix and zero
initialisation for the bias. Xavier preserves the variance of the
activations entering the head, and the additional shrinkage factor of
0.01 ensures that initial logits are concentrated near the origin so that
``softmax`` produces a near-uniform predictive distribution — the
maximum-entropy starting point for a 100-class classifier.
"""

import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the final classification head in-place.

    Applies Xavier uniform initialisation to the weight matrix scaled by
    a conservative shrinkage factor, and zeroes the bias. This yields a
    near-uniform output distribution prior to any fine-tuning.

    Args:
        layer: The ``nn.Linear`` head appended to the backbone.
    """
    nn.init.xavier_uniform_(layer.weight)
    layer.weight.data.mul_(0.01)
    nn.init.zeros_(layer.bias)
