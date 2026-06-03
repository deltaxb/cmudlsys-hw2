"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype))
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=device, dtype=dtype).reshape((1, out_features)))
            # print(f"init: {self.bias.cached_data}")
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print(f"111: {self.bias.shape}")
        batch = X.shape[0]
        ops.broadcast_to(self.bias, (batch, self.out_features))
        # print(f"222: {self.bias.shape}")
        if self.bias is not None:
            return X @ self.weight + ops.broadcast_to(self.bias, (batch, self.out_features))
        return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print(f"x.shape: {X.shape}")
        size = X.cached_data.size
        # print(f"x.size: {size}")
        return X.reshape((X.shape[0], size // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for model in self.modules:
            x = model.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = logits.cached_data.size / logits.shape[-1]
        return (ops.logsumexp(logits,axes=(len(logits.shape) - 1,)).sum() - (logits * init.one_hot(logits.shape[-1], y, device=y.device, dtype=logits.dtype).broadcast_to(logits.shape)).sum()) / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight:Tensor = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias:Tensor = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean:Tensor = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        self.running_var:Tensor = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        xshape = x.shape
        batch = x.shape[0]
        eps = self.eps
        momentum = self.momentum
        assert len(xshape) == 2
        # def restore(a:Tensor):
        #     ori = a.shape
        #     a = a.reshape((1, ) + ori)
        #     a = a.broadcast_to(xshape)
        #     return a
        if self.training == True:
            mean = x.sum(axes=(0,)) / batch
            var = ((x - mean.broadcast_to(xshape)) ** 2).sum(axes=(0,)) / batch
            self.running_mean.data = (1.0 - momentum) * self.running_mean.data + momentum * mean.data
            self.running_var.data = (1.0 - momentum) * self.running_var.data + momentum * var.data
        else:
            mean = self.running_mean
            var = self.running_var
        mean = (mean).broadcast_to(xshape)
        var = (var).broadcast_to(xshape)
        weight = self.weight.broadcast_to(xshape)
        bias = self.bias.broadcast_to(xshape)
        return weight * (x - mean) * ((var + eps) ** (-0.5)) + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight:Tensor = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias:Tensor = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        xshape = x.shape
        dim = self.dim
        eps = self.eps
        assert x.shape[-1] == dim
        print(self.weight.shape, self.bias.shape)
        weight = self.weight.broadcast_to(xshape)
        bias = self.bias.broadcast_to(xshape)
        print(weight.shape, bias.shape)
        def restore(a:Tensor):
            ori = a.shape
            a = a.reshape(ori + (1,))
            a = a.broadcast_to(xshape)
            return a
        e_x = restore(x.sum(axes=(-1,)) / dim)
        var_x = restore(((x - e_x) ** 2).sum(axes=(-1,)) / dim)
        return weight * ((x - e_x) * ((var_x + eps) ** -0.5)) + bias
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTIO
        if self.training == True:
            drop = init.randb(*x.shape, p = (1 - self.p), device=x.device, dtype=x.dtype)
            return  drop / (1.0 - self.p) * x
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
