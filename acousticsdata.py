
from torch.utils.data import Dataset, DataLoader, IterableDataset
from random import randrange
import zarr

class IterableDataset(IterableDataset):
    def __init__(self, datafiles, psize, rsize, type='zarr'):
        '''
        Args: 
           datafiles: input datafile (zarr, at least for now)
           samples: number of samples to extract
           psize, rsize: number of pings and ranges to extract - i.e. horizontal and vertical size

           Returns a stream of random rectangles with sv values
        '''

        zf = zarr.open(datafiles)
        self.sv = zf.sv
        self.shape = zf.sv.shape
        self.psize = psize
        self.rsize = rsize

    def __iter__(self):
        '''Select and yield a random rectangle'''
        while True:
            (_, pings, maxrng) = self.shape
        
            x = randrange(0, pings-self.psize)
            y = randrange(0, maxrng-self.rsize)
            
            rect = self.sv[:, x:x+self.psize, y:y+self.rsize]
            yield rect, (x, y)

class TiledDataset(Dataset):
    '''Splitting an acoustic data set into tiles of given size'''

    def __init__(self, datafiles, psize, rsize, type='zarr'):

        zf = zarr.open(datafiles)
        self.sv = zf.sv
        self.shape = zf.sv.shape
        self.psize = psize
        self.rsize = rsize

    def _num2coord(self, i):
        (_, _, maxrng) = self.shape
        colheight = maxrng//self.rsize

        p, r = divmod(i, colheight)
        return (p, r)

    def __len__(self):
        (_, pings, maxrng) = self.shape
        colheight = maxrng//self.rsize
        rowlength = pings//self.psize
        return(colheight*rowlength)

    def __getitem__(self, idx):        
        '''Extract tile number idx'''
        # tile number = depth, range
        p, r = self._num2coord(idx)

        x = p * self.psize
        y = r * self.rsize

        rect = self.sv[:, x:x+self.psize, y:y+self.rsize]
        return rect, (x, y)
