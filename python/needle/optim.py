"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for theta in self.params:
            if theta not in self.u:
                self.u[theta] = ndl.init.zeros(*theta.shape, device=theta.device, dtype=theta.dtype)
            grad = theta.grad.data + self.weight_decay * theta.data
            self.u[theta].data = self.momentum * self.u[theta].data + (1 - self.momentum) * grad.data
            theta.data -=  self.lr * self.u[theta].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for theta in self.params:
            if theta not in self.m:
                self.m[theta] = ndl.init.zeros(*theta.shape, device=theta.device, dtype=theta.dtype)
            if theta not in self.v:
                self.v[theta] = ndl.init.zeros(*theta.shape, device=theta.device, dtype=theta.dtype)
            grad = theta.grad.data + self.weight_decay * theta.data
            self.m[theta].data = self.beta1 * self.m[theta].data + (1.0 - self.beta1) * grad.data
            self.v[theta].data = self.beta2 * self.v[theta].data + (1.0 - self.beta2) * (grad.data ** 2)
            mm = self.m[theta].data / (1.0 - self.beta1 ** self.t)
            vv = self.v[theta].data / (1.0 - self.beta2 ** self.t)
            theta.data -= self.lr * mm.data / (vv.data ** 0.5 + self.eps)
        ### END YOUR SOLUTION
