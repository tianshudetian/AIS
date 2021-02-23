import numpy as np
from hilbert import decode
def hilbertGenerate(num_bits, num_dims):
    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)

    # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)

    # Compute the 2-dimensional locations.
    locs = decode(hilberts, num_dims, num_bits)
    return locs

def draw_curve(ax, num_bits, num_dims):

    locs = hilbertGenerate(num_bits, num_dims)

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
    ax.set_title('%d bits per dimension' % (num_bits))
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

def Knear(loc, K, num_bits):
    A = [-1, 0, 1]
    for k in range(K):
        if k > 0.1:
            a = [i * (k + 1) for i in A]
            A.extend(a)
    A = list(set(A))
    x = []
    y = []
    for i, element in enumerate(loc):
        for a in A:
            newElement = element + a
            if newElement >= 0 and newElement <= 2**num_bits:
                if i == 0:
                    x.append(newElement)
                else:
                    y.append(newElement)
    nearLoc = []
    for i in x:
        for j in y:
            B = [i, j]
            nearLoc.append(B)
    return nearLoc
