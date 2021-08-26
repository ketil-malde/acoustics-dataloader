# Example zarr file: /scratch/disk2/AzureMirror/cruise_data/2020/S2020819_PVENDLA_3206/ACOUSTIC/GRIDDED_EK_DATA/out.zarr

exfile = 'out.zarr'

# todo: benchmark performance

from acousticsdata import IterableDataset, TiledDataset
from torch.utils.data import DataLoader
from itertools import islice

def show(item):
    arr, coord = item
    print(arr.shape, coord)

print('Iterable')    
iterd = IterableDataset(exfile, 10, 10)

# cnt = 0
# for a, c in test.iterd:
#     print(a.shape, c)
#     cnt += 1
#     if cnt>5: break


loader = DataLoader(iterd, batch_size=2)
for batch in islice(loader, 3):
    print(batch)

print('Tiled')    
tiled = TiledDataset(exfile, 10, 10)

# for i in range(5):
#    show(tiled[i])

loader = DataLoader(tiled, batch_size=2)
for batch in islice(loader, 3):
    print(batch)
