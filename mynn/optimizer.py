from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        # 初始化动量缓存：字典嵌套，每一层每个参数一个 v
        self.velocity = {}
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocity[idx] = {}
                for key in layer.params.keys():
                    self.velocity[idx][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():

                    grad = layer.grads[key]
                    v = self.velocity[idx][key]

                    # update the moment
                    v = self.mu * v - self.init_lr * grad

                    # L2 regularization
                    if layer.weight_decay:
                        v -= self.init_lr * layer.weight_decay_lambda * layer.params[key]

                    # update the parameters
                    layer.params[key] += v

                    # save the updated moment
                    self.velocity[idx][key] = v

# class MomentumOriginal(Optimizer):
#     """
#     Implements momentum using the original formula from the project description:
#     w_{t+1} = w_t - α_t * ∇f(w_t) + β_t * (w_t - w_{t−1})
    
#     - α_t: learning rate
#     - β_t: momentum coefficient
#     """
#     def __init__(self, init_lr, model, beta=0.9):
#         super().__init__(init_lr, model)
#         self.beta = beta
#         self.prev_weights = {}  # stores w_{t-1} for each layer and parameter

#     def step(self):
#         for idx, layer in enumerate(self.model.layers):
#             if layer.optimizable:
#                 # Initialize previous weights for the layer if not present
#                 if idx not in self.prev_weights:
#                     self.prev_weights[idx] = {}
#                     for key in layer.params:
#                         self.prev_weights[idx][key] = layer.params[key].copy()

#                 for key in layer.params:
#                     w_t = layer.params[key]                    # current weight
#                     w_prev = self.prev_weights[idx][key]       # previous weight
#                     grad = layer.grads[key]                    # current gradient

#                     # Apply L2 regularization
#                     if layer.weight_decay:
#                         grad += layer.weight_decay_lambda * w_t

#                     # Compute update using project-specified momentum rule
#                     update = -self.init_lr * grad + self.beta * (w_t - w_prev)

#                     # Store current weight as previous for the next step
#                     self.prev_weights[idx][key] = w_t.copy()

#                     # save the update
#                     layer.params[key] = w_t + update

class MomentumOriginal(Optimizer):
    """
    Implements momentum using the original formula:
    w_{t+1} = w_t - α_t * ∇f(w_t) + β_t * (w_t - w_{t−1})
    """
    def __init__(self, init_lr, model, beta=0.9):
        super().__init__(init_lr, model)
        self.beta = beta
        self.prev_weights = {}  # stores w_{t-1} for each layer and parameter

    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                # Initialize prev_weights only once
                if idx not in self.prev_weights:
                    self.prev_weights[idx] = {}
                    for key in layer.params:
                        self.prev_weights[idx][key] = layer.params[key].copy()

                for key in layer.params:
                    w_t = layer.params[key]
                    w_prev = self.prev_weights[idx][key]
                    grad = layer.grads[key]

                    if grad is None:
                        continue  # skip if no gradient

                    # L2 regularization
                    if layer.weight_decay:
                        grad += layer.weight_decay_lambda * w_t

                    # Compute update
                    update = -self.init_lr * grad + self.beta * (w_t - w_prev)

                    # In-place update of weights
                    layer.params[key] -= self.init_lr * grad   # this is equivalent to w_t = w_t - α * grad
                    layer.params[key] += self.beta * (w_t - w_prev)  # still in-place, additive

                    # Update prev_weights for next step
                    self.prev_weights[idx][key][...] = w_t  # in-place copy

