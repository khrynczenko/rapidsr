from torchvision import transforms
import dgl
import numpy as np

from rapidsr import _internal


def convert_float_to_byte_image(image: np.ndarray) -> np.ndarray:
    return np.array(image * 255.0, np.uint8)


def image_to_graph(image: np.ndarray) -> dgl.DGLGraph:
    rows, cols = image.shape[:2]
    graph = dgl.DGLGraph()
    graph.rows = rows
    graph.cols = cols
    graph.size = image.size
    n_nodes = image.size
    graph.add_nodes(n_nodes)
    edge_list = _internal._get_edges_for_image(image)
    src, dst = tuple(zip(*edge_list))
    graph.add_edges(src, dst)
    tensor = transforms.ToTensor()(image)
    tensor = tensor.reshape((tensor.numpy().size, 1))
    graph.ndata["feat"] = tensor
    return graph


def graph_to_image(graph: dgl.DGLGraph) -> np.ndarray:
    image = convert_float_to_byte_image(
        graph.ndata["feat"].numpy().reshape(graph.rows, graph.cols))
    return image
