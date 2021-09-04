
Dataset classes for using PyTorch with acoustics data.  Pytorch's
Dataloader uses a Dataset instance to read in data for training or
classification.

Acoustics data are converted from the original set of RAW files (as
produced by EK60 and EK80 echo sounders) to Python-friendly arrays in
zarr format.

Here we implement two versions: IterableDataset and TiledDataset.
Iterable is a generator that gives an infinite supply or rectangles
randomly chosen from the input data, while Tiled enumerates all
non-overlapping rectangles and returns them by index.



