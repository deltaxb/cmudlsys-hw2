from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(axis=1,keepdims=True)
        sum = array_api.sum(array_api.exp(Z - maxz), 1, keepdims=True)
        return Z - array_api.log(sum) - maxz
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        num_c = node.inputs[0].shape[-1]
        Z = exp(logsoftmax(node.inputs[0]))
        def turn(a:Tensor, num_c):
            nshape = a.shape + (1,)
            ncshape = a.shape + (num_c,)
            return a.reshape(nshape).broadcast_to(ncshape)
        A = out_grad.sum(axes=(-1,))
        return out_grad - Z * turn(A, num_c)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxz = Z.max(axis=self.axes,keepdims=True)
        sum = array_api.sum(array_api.exp(Z - maxz), self.axes)
        return array_api.log(sum) + maxz.reshape(sum.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        def restore_shape(ori_shape, axes):
            axes = axes if axes is not None else tuple(range(len(ori_shape)))
            return tuple(1 if i in axes else v for i, v in enumerate(ori_shape))
        def turnshape(a:Tensor, tmp_shape, fin_shape) -> Tensor:
            return a.reshape(tmp_shape).broadcast_to(fin_shape)
        Z = node.inputs[0]
        ori_shape = Z.shape
        # print(out_grad.reshape(restore_shape(ori_shape, self.axes)).broadcast_to(ori_shape).shape)
        # print(ori_shape)
        tmp_shape = restore_shape(ori_shape, self.axes)
        return turnshape(out_grad, tmp_shape, ori_shape) * exp(Z - turnshape(logsumexp(Z,self.axes), tmp_shape, ori_shape))
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)