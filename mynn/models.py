from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Model_MLP_N(Layer):
    def __init__(self, input_dim, nHidden, output_dim, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = [input_dim] + nHidden + [output_dim]
        self.act_func = act_func

        if self.size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Model_MLP_NEO(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, input_dim, nHidden, output_dim, act_func=None, lambda_list=None, dropout_rates=None):
        super().__init__()
        self.size_list = [input_dim] + nHidden + [output_dim]
        self.act_func = act_func
        self.layers = []

        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[i]
            self.layers.append(layer)

            # 对隐藏层添加ReLU和Dropout
            if i < len(self.size_list) - 2:
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError("Logistic not supported")

                # Apply Dropout Layer
                if dropout_rates is not None:
                    self.layers.append(Dropout(p=dropout_rates[i]))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    # def load_model(self, param_list):
    #     with open(param_list, 'rb') as f:
    #         param_list = pickle.load(f)
    #     self.size_list = param_list[0]
    #     self.act_func = param_list[1]

    #     for i in range(len(self.size_list) - 1):
    #         self.layers = []
    #         for i in range(len(self.size_list) - 1):
    #             layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
    #             layer.W = param_list[i + 2]['W']
    #             layer.b = param_list[i + 2]['b']
    #             layer.params['W'] = layer.W
    #             layer.params['b'] = layer.b
    #             layer.weight_decay = param_list[i + 2]['weight_decay']
    #             layer.weight_decay_lambda = param_list[i+2]['lambda']
    #             if self.act_func == 'Logistic':
    #                 raise NotImplemented
    #             elif self.act_func == 'ReLU':
    #                 layer_f = ReLU()
    #             self.layers.append(layer)
    #             if i < len(self.size_list) - 2:
    #                 self.layers.append(layer_f)

    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            param_list = pickle.load(f)

        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        param_idx = 0

        for i in range(len(self.size_list) - 1):
            # Linear layer
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[param_idx + 2]['W']
            layer.b = param_list[param_idx + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[param_idx + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[param_idx + 2]['lambda']
            self.layers.append(layer)
            param_idx += 1

            # ReLU
            if i < len(self.size_list) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError

                # Dropout 不需要加载参数，默认保留原模型结构时附带
                if hasattr(self, 'dropout_rates'):
                    self.layers.append(Dropout(p=self.dropout_rates[i]))

        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.optimizable:  # ✅ skip Dropout and others
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            with open(save_path, 'wb') as f:
                pickle.dump(param_list, f)

class Model_CNN0(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        pass

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, loss_grad):
        pass
    
    def load_model(self, param_list):
        pass
        
    def save_model(self, save_path):
        pass

class Model_CNN(Layer):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        # self.layers = [
        #     conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(kernel_size=2, stride=2),  # 28→14

        #     conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(kernel_size=2, stride=2),  # 14→7

        #     conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     ReLU(),
        #     MaxPool2D(kernel_size=2, stride=2),  # 7→3

        #     Flatten(),  # 32 × 3 × 3 = 288
        #     Linear(in_dim=32 * 3 * 3, out_dim=128),
        #     ReLU(),
        #     Dropout(p=dropout_rate),
        #     Linear(in_dim=128, out_dim=10)
        # ]
        self.layers = [
            conv2D(1, 8, 3, 1, 1),
            ReLU(),
            DummyPool(),
            # MaxPool2D(2, 2),   # 28 → 14
            Flatten(), 
            Linear(6272, 1024),
            ReLU(),
            Linear(1024, 128),
            ReLU(),   
            # Dropout(p=dropout_rate),
            Linear(128, 10)
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def load_model(self, path):
        with open(path, 'rb') as f:
            param_list = pickle.load(f)

        ptr = 0  # index in param_list
        for layer in self.layers:
            if layer.optimizable:
                layer.params['W'] = param_list[ptr]['W']
                layer.params['b'] = param_list[ptr]['b']
                layer.weight_decay = param_list[ptr]['weight_decay']
                layer.weight_decay_lambda = param_list[ptr]['lambda']
                ptr += 1

    def save_model(self, path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(path, 'wb') as f:
            pickle.dump(param_list, f)
