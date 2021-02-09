import numpy as np
from hilbert import decode, encode

# Turn an ndarray of Hilber integers into locations.
# 2 is the number of dimensions, 3 is the number of bits per dimension
locs = decode(np.array([0,1,2,3,4]), 2, 3)
print(locs)
# prints [[0 1]
#         [1 1]
#         [1 0]]

# You can go the other way also, of course.
H = encode(locs, 2, 3)

print(H)
# prints array([1, 2, 3], dtype=uint64)