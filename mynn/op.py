from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.dot(X, self.W) + self.b
        pass

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.input.T, grad) / self.input.shape[0]
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / self.input.shape[0]

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return np.dot(grad, self.W.T)
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

def im2col(X, kernel_size, stride, padding):
    """
    Transform input tensor into columns for vectorized convolution.
    """
    N, C, H, W = X.shape
    k = kernel_size
    s = stride

    out_H = (H - k + 2 * padding) // s + 1
    out_W = (W - k + 2 * padding) // s + 1

    if padding > 0:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    cols = np.zeros((N, C, k, k, out_H, out_W))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = X[:, :, i:i + s * out_H:s, j:j + s * out_W:s]

    cols = cols.reshape(N, C * k * k, out_H * out_W)
    return cols  # Shape: [N, C*k*k, out_H*out_W]

def col2im(cols, input_shape, kernel_size, stride, padding):
    N, C, H, W = input_shape
    k = kernel_size
    out_h = (H + 2 * padding - k) // stride + 1
    out_w = (W + 2 * padding - k) // stride + 1

    cols_reshaped = cols.reshape(N, C, k, k, out_h, out_w)
    cols_reshaped = cols_reshaped.transpose(0, 1, 4, 5, 2, 3)

    img = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

    for y in range(k):
        for x in range(k):
            img[:, :, y:y + stride * out_h:stride, x:x + stride * out_w:stride] += cols_reshaped[:, :, :, :, y, x]

    if padding == 0:
        return img
    return img[:, :, padding:-padding, padding:-padding]

initialize_method=lambda size: np.random.randn(*size) * 0.01

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=initialize_method, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    # def forward(self, X):
    #     """
    #     input X: [batch, channels, H, W]
    #     W : [1, out, in, k, k]
    #     no padding
    #     """
    #     self.input = X
    #     batch_size, in_channels, H, W = X.shape
    #     k = self.kernel_size
    #     s = self.stride
    #     p = self.padding

    #     # add padding
    #     if p > 0:
    #         X = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')

    #     H_p, W_p = X.shape[2], X.shape[3]
    #     out_H = (H_p - k) // s + 1
    #     out_W = (W_p - k) // s + 1
    #     output = np.zeros((batch_size, self.out_channels, out_H, out_W))

    #     # Convolution
    #     for n in range(batch_size):
    #         for oc in range(self.out_channels):
    #             for i in range(out_H):
    #                 for j in range(out_W):
    #                     region = X[n, :, i*s:i*s+k, j*s:j*s+k]  # shape: [in_channels, k, k]
    #                     output[n, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]
    #     return output
    #     # pass

    def forward(self, X):
        """
        Vectorized forward pass for 2D convolution.
        """
        self.input = X
        self.X_col = im2col(X, self.kernel_size, self.stride, self.padding)  # [N, C*k*k, OH*OW]
        N, _, L = self.X_col.shape
        W_col = self.W.reshape(self.out_channels, -1)  # [F, C*k*k]

        # Matrix multiplication to compute convolution result
        out = np.tensordot(W_col, self.X_col, axes=([1], [1]))  # [F, N, OH*OW]
        out = out.transpose(1, 0, 2) + self.b.reshape(1, -1, 1)  # Add bias

        out_H = (X.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_W = (X.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out.reshape(N, self.out_channels, out_H, out_W)


    def backward(self, grad_output):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        batch_size, in_channels, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # 如果 padding 了，说明 X 是原图，仍需要 pad
        if p > 0:
            X = np.pad(X, ((0,0), (0,0), (p,p), (p,p)), mode='constant')

        _, _, H_p, W_p = X.shape
        out_H = (H_p - k) // s + 1
        out_W = (W_p - k) // s + 1

        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for n in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        region = X[n, :, i*s:i*s+k, j*s:j*s+k]
                        dW[oc] += grad_output[n, oc, i, j] * region
                        dX[n, :, i*s:i*s+k, j*s:j*s+k] += grad_output[n, oc, i, j] * self.W[oc]
                db[oc] += np.sum(grad_output[n, oc])

        # average gradient
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size

        # L2 regularization
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        if p > 0:
            dX = dX[:, :, p:-p, p:-p]
        return dX
        # pass

    def backward(self, grad_output):
        """
        Vectorized backward pass for 2D convolution.
        grad_output: [N, out_channels, out_H, out_W]
        """
        X = self.input
        N, C, H, W = X.shape
        F = self.out_channels
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Compute output size
        out_H = (H + 2 * p - k) // s + 1
        out_W = (W + 2 * p - k) // s + 1

        # Reshape
        dout_reshaped = grad_output.reshape(N, F, -1)  # [N, F, OH*OW]
        W_col = self.W.reshape(F, -1)                  # [F, C*k*k]

        # Gradient w.r.t. weights
        dW = np.tensordot(dout_reshaped, self.X_col, axes=([0, 2], [0, 2]))  # [F, C*k*k]
        dW = dW.reshape(self.W.shape) / N

        # Gradient w.r.t. bias
        db = np.sum(dout_reshaped, axis=(0, 2)) / N  # [F]

        # Gradient w.r.t. input
        dX_col = np.tensordot(dout_reshaped, W_col, axes=([1], [0]))  # [N, OH*OW, C*k*k]
        dX_col = dX_col.transpose(0, 2, 1)  # [N, C*k*k, OH*OW]

        dX = col2im(dX_col, X.shape, k, s, p)  # [N, C, H, W]

        # Save grads
        self.grads['W'] = dW
        self.grads['b'] = db

        # Apply L2 regularization
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return dX

    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.preds = None
        self.labels = None 
        self.grads = None 
        self.optimizable = False
        self.has_softmax = True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels

        # softmax
        if self.has_softmax:
            self.preds = softmax(predicts)
        else:
            self.preds = predicts

        # Calculate the crossentropy
        m = predicts.shape[0]
        eps = 1e-9  # avoid log(0)
        log_likelihood = -np.log(self.preds[range(m), labels] + eps)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        m = self.preds.shape[0]
        grad = self.preds.copy()
        grad[range(m), self.labels] -= 1
        grad = grad / m
        self.grads = grad
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, params, weight_decay_lambda=1e-4) -> None:
        super().__init__()
        self.params = params
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizable = False

    def forward(self):
        """
        L2 loss：sum(W^2) / 2
        """
        reg_loss = 0.0
        for key, param in self.params.items():
            reg_loss += np.sum(param ** 2)
        return 0.5 * self.weight_decay_lambda * reg_loss

    def backward(self):
        grads = {}
        for key, param in self.params.items():
            grads[key] = self.weight_decay_lambda * param
        return grads
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    # def forward(self, X):
    #     self.input = X
    #     batch_size, C, H, W = X.shape
    #     k = self.kernel_size
    #     s = self.stride

    #     out_H = (H - k) // s + 1
    #     out_W = (W - k) // s + 1
    #     out = np.zeros((batch_size, C, out_H, out_W))
    #     self.mask = np.zeros_like(X)

    #     for n in range(batch_size):
    #         for c in range(C):
    #             for i in range(out_H):
    #                 for j in range(out_W):
    #                     h_start = i * s
    #                     w_start = j * s
    #                     region = X[n, c, h_start:h_start+k, w_start:w_start+k]
    #                     max_val = np.max(region)
    #                     out[n, c, i, j] = max_val
    #                     # 记录最大值位置
    #                     max_mask = (region == max_val)
    #                     self.mask[n, c, h_start:h_start+k, w_start:w_start+k] += max_mask
    #     return out

    # def forward(self, X):
    #     self.input = X
    #     N, C, H, W = X.shape
    #     k = self.kernel_size
    #     s = self.stride

    #     assert H % k == 0 and W % k == 0, "Height and width must be divisible by kernel size"

    #     out_H = (H - k) // s + 1
    #     out_W = (W - k) // s + 1

    #     X_reshaped = X.reshape(N, C, out_H, k, out_W, k)
    #     X_reshaped = X_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, out_H, out_W, k * k)
    #     out = np.max(X_reshaped, axis=-1)

    #     self.mask = (X_reshaped == out[..., None])
    #     return out

    def forward(self, X):
        self.input = X
        batch_size, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        out = np.zeros((batch_size, C, out_H, out_W))
        self.mask = np.zeros_like(X)

        for n in range(batch_size):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * s
                        w_start = j * s
                        region = X[n, c, h_start:h_start+k, w_start:w_start+k]
                        max_val = np.max(region)
                        out[n, c, i, j] = max_val
                        self.mask[n, c, h_start:h_start+k, w_start:w_start+k] += (region == max_val)
        return out


    def backward(self, grad):
        batch_size, C, H, W = self.input.shape
        dx = np.zeros_like(self.input)
        k = self.kernel_size
        s = self.stride
        out_H = grad.shape[2]
        out_W = grad.shape[3]

        for n in range(batch_size):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * s
                        w_start = j * s
                        dx[n, c, h_start:h_start+k, w_start:w_start+k] += grad[n, c, i, j] * self.mask[n, c, h_start:h_start+k, w_start:w_start+k]
        return dx
    
class DummyPool(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.params = {}
        self.grads = {}
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        return X
    def backward(self, grad):
        return grad


class Flatten(Layer):
    """
    将多维输入展平为二维，保持 batch_size 不变
    例如: [batch, channels, height, width] → [batch, channels * height * width]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        p: dropout 概率，表示有多少比例的神经元会被丢弃
        """
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(np.float32)
            # 为了保持期望值不变，需要除以保留的比例
            return (X * self.mask) / (1.0 - self.p)
        else:
            return X

    def backward(self, grad):
        if self.training:
            return (grad * self.mask) / (1.0 - self.p)
        else:
            return grad

    def eval(self):
        self.training = False

    def train(self):
        self.training = True
