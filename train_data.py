"""
train_data.py — Training dataset and dataloader construction.

Returns the CIFAR100 training split (50,000 images) wrapped in a
``DataLoader``. Because our linear-probe optimiser draws a *uniformly
random* mini-batch on every step, we keep ``shuffle=True`` and rely on
the seeded generator (set in ``validate.py``) for reproducibility. With
the default budget of 32 × 32 = 1024 samples, well under one epoch of the
50k training set, each ``.step()`` sees a fresh, non-repeating mini-batch.
"""

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from augmentation import get_transforms


USE_TRAIN_SUBSET_ONLY = True


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    """Build the CIFAR100 training dataset and a reproducibly-shuffled loader.

    Args:
        data_dir:        Root directory for the CIFAR100 archive (auto-downloaded).
        batch_size:      Mini-batch size used by the fine-tuning loop.
        generator_train: Seeded ``torch.Generator`` for reproducible shuffling.

    Returns:
        Tuple of (dataset, loader). The loader shuffles each epoch but is
        deterministic given a fixed generator seed.
    """
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,  # True → CIFAR100 train split (50k images)
        download=True,
        transform=get_transforms(train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator_train,
    )

    return train_dataset, train_loader
