import numpy as np
import matplotlib.pyplot as plt
import gzip
import struct

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

images = read_idx('./data/MNIST/raw/t10k-images-idx3-ubyte.gz')

plt.imshow(images[0], cmap='gray')
plt.show()