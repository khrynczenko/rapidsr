import numpy as np
from typing import List, Tuple
import itertools


def count_edges_for_image(image: np.ndarray) -> int:
    """
    Cound amount of directional (but in both directions) edges that would
    for an image that we would like to turn into graph.

    :return: amount of bidirectional edges that would result in a
    graph corresponding to the input image
    """
    rows, cols = image.shape[:2]
    pixels_on_borders = (rows * 2) + (cols * 2) - 4
    pixels_inside = image.size - pixels_on_borders
    return pixels_inside


def _get_edges_for_image(image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Calculated edges in the graph but instead of using coorinated is uses
    idx as if the image was a one vector. It is done this way so its easier
    to put it into DGLGraph.
    """
    rows, cols = image.shape[:2]
    all_coordinates = list(itertools.product(range(rows), range(cols)))
    all_cordinates_with_idx = zip(all_coordinates, itertools.count())
    coordinates_to_id = {coor: idx for coor, idx in all_cordinates_with_idx}
    cid = coordinates_to_id
    edges = []
    for row, col in all_coordinates:
        # add edges for left upper corner
        if row == col == 0:
            edges.extend([(cid[(0, 0)], cid[(0, 1)]),
                          (cid[(0, 0)], cid[(1, 0)]),
                          (cid[(0, 0)], cid[(1, 1)])])
        # add edges for right upper corner
        elif row == 0 and col == cols - 1:
            edges.extend([(cid[(0, cols - 1)], cid[(0, cols - 2)]),
                          (cid[(0, cols - 1)], cid[(1, cols - 2)]),
                          (cid[(0, cols - 1)], cid[(1, cols - 1)])])
        # add edges for left lower corner
        elif row == rows - 1 and col == 0:
            edges.extend([(cid[(rows - 1, 0)], cid[(rows - 2, 0)]),
                          (cid[(rows - 1, 0)], cid[(rows - 2, 1)]),
                          (cid[(rows - 1, 0)], cid[(rows - 1, 1)])])
        # add edges for right lower corner
        elif row == rows - 1 and col == cols - 1:
            edges.extend(
                [(cid[(rows - 1, cols - 1)], cid[(rows - 1, cols - 2)]),
                 (cid[(rows - 1, cols - 1)], cid[(rows - 2, cols - 1)]),
                 (cid[(rows - 1, cols - 1)], cid[(rows - 2, cols - 2)])])
        elif row == 0:
            edges.extend([(cid[(row, col)], cid[(row, col - 1)]),
                          (cid[(row, col)], cid[(row, col + 1)]),
                          (cid[(row, col)], cid[(row + 1, col)]),
                          (cid[(row, col)], cid[(row + 1, col - 1)]),
                          (cid[(row, col)], cid[(row + 1, col + 1)])])

        elif row == rows - 1:
            edges.extend([(cid[(row, col)], cid[(row, col - 1)]),
                          (cid[(row, col)], cid[(row, col + 1)]),
                          (cid[(row, col)], cid[(row - 1, col)]),
                          (cid[(row, col)], cid[(row - 1, col - 1)]),
                          (cid[(row, col)], cid[(row - 1, col + 1)])])

        elif col == 0:
            edges.extend([(cid[(row, col)], cid[(row - 1, col)]),
                          (cid[(row, col)], cid[(row + 1, col)]),
                          (cid[(row, col)], cid[(row - 1, col + 1)]),
                          (cid[(row, col)], cid[(row, col + 1)]),
                          (cid[(row, col)], cid[(row + 1, col + 1)])])

        elif col == cols - 1:
            edges.extend([(cid[(row, col)], cid[(row - 1, col)]),
                          (cid[(row, col)], cid[(row + 1, col)]),
                          (cid[(row, col)], cid[(row - 1, col - 1)]),
                          (cid[(row, col)], cid[(row, col - 1)]),
                          (cid[(row, col)], cid[(row + 1, col - 1)])])

        else:
            edges.extend([(cid[(row, col)], cid[(row - 1, col + 1)]),
                          (cid[(row, col)], cid[(row - 1, col)]),
                          (cid[(row, col)], cid[(row - 1, col - 1)]),
                          (cid[(row, col)], cid[(row + 1, col + 1)]),
                          (cid[(row, col)], cid[(row + 1, col)]),
                          (cid[(row, col)], cid[(row + 1, col - 1)]),
                          (cid[(row, col)], cid[(row, col - 1)]),
                          (cid[(row, col)], cid[(row, col + 1)])])
    return edges
