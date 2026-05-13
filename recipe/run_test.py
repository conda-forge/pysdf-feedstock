import numpy as np
from pysdf import SDF


vertices = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
faces = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ],
    dtype=np.uint32,
)

sdf = SDF(vertices, faces)
points = np.array(
    [
        [0.25, 0.25, 0.25],
        [2.0, 2.0, 2.0],
        [0.5, 0.0, 0.0],
    ],
    dtype=np.float64,
)

distances = sdf(points)
contains = sdf.contains(points)

assert distances.shape == (3,)
assert distances[0] > 0.0
assert distances[1] < 0.0
assert np.isclose(distances[2], 0.0)
np.testing.assert_array_equal(contains[:2], [True, False])
