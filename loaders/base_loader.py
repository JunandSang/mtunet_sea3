import torch
from torch.utils.data import DataLoader, Dataset
import loaders.datasets as datasets


def get_dataloader(args, split, shuffle=True, out_name=False, sample=None, selection=None, mode=None, seed=None):
    """
    Create a DataLoader for the given dataset and split.

    Args:
        args: Namespace object containing training arguments.
        split: The split of the dataset ('train', 'val', 'test').
        shuffle: Whether to shuffle the data.
        out_name: Whether to output the file name.
        sample: Few-shot sampling configuration.
        selection: Selected classes for the dataset.
        mode: Additional mode for dataset processing.
        seed: Seed for random sampling.

    Returns:
        DataLoader for the dataset.
    """
    # Determine transform mode
    if args.fsl:
        ts_condition = split
    else:
        ts_condition = mode if mode else split  # Default to split if mode is None

    transform = datasets.make_transform(args, ts_condition)
    sets = datasets.DatasetFolder(args.data_root, args.dataset, split, transform, out_name=out_name, cls_selction=selection, mode=mode)

    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample, seed)
        loader = DataLoader(sets, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        loader = DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    
    return loader

