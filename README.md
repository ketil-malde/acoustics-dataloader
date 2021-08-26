
A Dataloader class for using PyTorch with acoustics data.

Two versions: IterableDataset and TiledDataset

Initialize with a zarr-formatted directory, and iterate to generate
rectangular samples of a given size.  User PyTorch Dataloader to
return proper tensors.


