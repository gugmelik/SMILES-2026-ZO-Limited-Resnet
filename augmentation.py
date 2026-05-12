"""
augmentation.py — Image transform pipelines for CIFAR100.

Our zero-order fine-tuning approach refits the classification head by
caching the penultimate features extracted from training images and
solving a closed-form linear-probe problem on that cache. For this
strategy to be effective the **training-time features must match
test-time features** as closely as possible: any stochastic augmentation
would inject distribution shift between the cached training features and
the test-time features that the fitted head ultimately classifies.

We therefore align the training pipeline with the (fixed) validation
pipeline: ``Resize → ToTensor → Normalize`` with the per-channel
CIFAR100 statistics. No random crops, no flips, no colour jitter.

The validation pipeline is the original fixed pipeline and must not be
modified.
"""

import torchvision.transforms as T


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the image transform pipeline for CIFAR100.

    Args:
        train: If ``True``, returns the training pipeline (deterministic
               here, since our optimiser relies on feature consistency
               between train and test). If ``False``, returns the fixed
               validation pipeline.

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    if train:
        return T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(224),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
