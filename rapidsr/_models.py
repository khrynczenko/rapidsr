from typing import Tuple
from torch import nn
import torch
import dgl
from dgl.nn.pytorch import conv as gconv


def _calculate_padding(kernel_size: int):
    """
    Calculates needed padding needed in order for the shape to remain unchanged
    after applying kernel.

    :param kernel_size: Size of a kernel that will be used.
    :return: Amount of padding needed.
    """
    if kernel_size % 2 == 0:
        raise RuntimeError("Padding cannot be calculated for kernel with "
                           "even size.")
    return kernel_size // 2


class SRCNN(nn.Module):
    """
    Implementation of Super-Resolution Convolutional Network based on paper
    titled "Image Super-Resolution Using Deep Convolutional Networks" by Dong
    et al.

    """

    def __init__(self, channels: int,
                 extraction_kernel_size: int = 9,
                 mapping_kernel_size: int = 1,
                 reconstruction_kernel_size: int = 5,
                 extraction_layer_kernels: int = 64,
                 mapping_layer_kernels: int = 32):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=extraction_layer_kernels,
                      kernel_size=extraction_kernel_size,
                      padding=_calculate_padding(extraction_kernel_size)),
            nn.ReLU())
        self.map = nn.Sequential(
            nn.Conv2d(in_channels=extraction_layer_kernels,
                      out_channels=mapping_layer_kernels,
                      kernel_size=mapping_kernel_size),
            nn.ReLU())
        self.reconstruct = nn.Sequential(
            nn.Conv2d(in_channels=mapping_layer_kernels, out_channels=channels,
                      kernel_size=reconstruction_kernel_size,
                      padding=_calculate_padding(reconstruction_kernel_size)),
            nn.ReLU())

    def forward(self, x):
        x = self.extract(x)
        x = self.map(x)
        x = self.reconstruct(x)
        return x


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Network based on paper titled
    "Accelerating the Super-Resolution Convolutional Neural Network"
    by Dong et al.

    """

    def __init__(self,
                 channels: int,
                 upscale_factor: int,
                 extraction_kernel_size: int = 5,
                 shrinkage_kernel_size: int = 1,
                 map_kernel_size: int = 3,
                 expansion_kernel_size: int = 1,
                 deconvolution_kernel_size: int = 9,
                 n_map_layers: int = 4,
                 n_dimension_filters: int = 56,
                 n_shrinkage_filters: int = 16):
        super().__init__()
        self.channels = channels
        self.upscale_factor = upscale_factor
        d = n_dimension_filters
        s = n_shrinkage_filters
        self.extract = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=d,
                      kernel_size=extraction_kernel_size,
                      padding=_calculate_padding(extraction_kernel_size)),
            nn.PReLU())
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=s,
                      kernel_size=shrinkage_kernel_size),
            nn.PReLU())
        self.map = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=s,
                      kernel_size=map_kernel_size,
                      padding=_calculate_padding(map_kernel_size)),
            nn.PReLU())
            for _ in range(n_map_layers)])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=s, out_channels=d,
                      kernel_size=expansion_kernel_size),
            nn.PReLU())

        pad, out_pad = self._calculate_deconvolution_padding(upscale_factor)
        self.deconvolve = nn.ConvTranspose2d(in_channels=d,
                                             out_channels=channels,
                                             kernel_size=
                                             deconvolution_kernel_size,
                                             stride=upscale_factor,
                                             padding=pad,
                                             output_padding=out_pad
                                             )

    def forward(self, x):
        x = self.extract(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconvolve(x)
        return x

    def _calculate_deconvolution_padding(self, factor: int) \
            -> Tuple[int, int]:
        """
        For different scale factor image must be differently padded so the
        output image shape will be scaled correctly i.e. without losing some
        pixels at borders.

        :param factor: upscale factor.
        :return: padding and output padding that must be applied at the
            deconvolution layer.
        """
        paddings = {
            2: (4, 1),
            3: (3, 0),
            4: (3, 1),
            8: (2, 3),
        }
        padding, output_padding = paddings.setdefault(factor, (3, 1))
        return padding, output_padding


class LapSRN(nn.Module):
    """
    LapSRN network as described in "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks" by
    Lai et al. Its variant is the one with shared-source skip connection, recursive blocks and pyramid level
    parameter sharing

    """

    class FeatureExtractionBlock(nn.Module):
        """
        Feature extraction block with shared source skip connection.

        """

        def __init__(self, depth: int, recursive_blocks_count: int):
            super().__init__()
            self._recursive_blocks_count = recursive_blocks_count
            self.recursive_block = nn.Sequential(*[nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                          padding=1),
                nn.LeakyReLU(negative_slope=0.2))
                for _ in range(depth)])
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                   kernel_size=4, stride=2,
                                   padding=1),
                nn.LeakyReLU(negative_slope=0.2)
            )

        def forward(self, x):
            y = self.recursive_block(x)
            y = y + x
            for _ in range(1, self._recursive_blocks_count):
                y = self.recursive_block(y)
                y = y + x
            return self.deconv(y)

    def __init__(self, channels: int, upscale_factor: int, depth: int = 5,
                 recursive_blocks_count: int = 8):
        super().__init__()
        self._possible_upscale_factors = [2, 4, 8, 16, 32]
        if upscale_factor not in self._possible_upscale_factors:
            raise ValueError("upscale factor should be power of 2 up to 2^5")
        self._channels = channels
        self._upscale_factor = upscale_factor
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.convt_i = nn.ConvTranspose2d(in_channels=channels,
                                          out_channels=channels,
                                          kernel_size=4, stride=2,
                                          padding=1)
        self.conv_r = nn.Conv2d(in_channels=64, out_channels=channels,
                                kernel_size=3, stride=1, padding=1)
        self.feature_extraction_block = \
            self.FeatureExtractionBlock(depth, recursive_blocks_count)

    def forward(self, x):
        conv0 = self.conv0(x)
        convt_f = self.feature_extraction_block(conv0)
        convt_i = self.convt_i(x)
        conv_r = self.conv_r(convt_f)
        out = convt_i + conv_r  # Upscaled X2
        for _ in range(len(list(filter(lambda x: x < self._upscale_factor,
                                       self._possible_upscale_factors)))):
            convt_f = self.feature_extraction_block(convt_f)
            convt_i = self.convt_i(out)
            conv_r = self.conv_r(convt_f)
            out = convt_i + conv_r
        return out


class SRGCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self._channels = channels
        self.first = gconv.GraphConv(1, 1, activation=nn.ReLU())
        self.second = gconv.GraphConv(1, 1, activation=nn.ReLU())
        self.third = gconv.GraphConv(1, 1, activation=nn.ReLU())

    def forward(self, graph: dgl.DGLGraph):
        graph.ndata['h'] = graph.ndata["feat"]
        graph.update_all(self._message, self._reduce)
        graph.ndata["h"] = self.first(graph, graph.ndata["h"])
        graph.update_all(self._message, self._reduce)
        graph.ndata["h"] = self.second(graph, graph.ndata["h"])
        graph.update_all(self._message, self._reduce)
        graph.ndata["h"] = self.third(graph, graph.ndata["h"])
        graph.ndata["feat"] = graph.ndata["h"]
        # return graph.ndata.pop('h')
        return graph

    @staticmethod
    def _message(edges):
        return {'m': edges.src['h']}

    @staticmethod
    def _reduce(nodes):
        accum = torch.mean(nodes.mailbox['m'], 1)
        return {'h': accum}
