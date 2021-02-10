from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
p = 2
n = 2
hilbert_curve = HilbertCurve(p, n)
num_points = 10_000
points = np.random.randint(
    low=0,
    high=hilbert_curve.max_x + 1,
    size=(num_points, hilbert_curve.n)
)
distances1 = hilbert_curve.distances_from_points(points)
distances2 = hilbert_curve.distances_from_points(points, match_type=True)
a=1