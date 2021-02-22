import numpy as np
import matplotlib.pyplot as plt

from hilbert import decode

num_dims = 2

def draw_curve(ax, num_bits):

    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)

    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)

    # Compute the 2-dimensional locations.
    locs = decode(hilberts, num_dims, num_bits)
    locs = locs + 0.5

    # Draw
    ax.plot(locs[:, 0], locs[:, 1], '.-')
    ax.set_xlim([0, pow(2, num_bits)])
    ax.set_ylim([0, pow(2, num_bits)])
    if (num_bits % 2) == 0:
        ax.set_xticks(range(0, pow(2, num_bits)+1, num_bits))
        ax.set_yticks(range(0, pow(2, num_bits)+1, num_bits))
    else:
        ax.set_xticks(range(0, pow(2, num_bits)+1, num_bits-1))
        ax.set_yticks(range(0, pow(2, num_bits)+1, num_bits-1))
    ax.set_aspect('equal')
    ax.set_title('%d bits per dimension' % (num_bits), fontdict={ 'size': 16})
    ax.set_xlabel('xlabel', fontdict={'size': 16})
    ax.set_ylabel('ylabel', fontdict={'size': 16})


fig = plt.figure(figsize=(20, 5))
for ii, num_bits in enumerate([2, 3, 4, 5]):
    ax = fig.add_subplot(1, 4, ii+1)
    draw_curve(ax, num_bits)
plt.show()