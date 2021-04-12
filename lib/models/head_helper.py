# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x
