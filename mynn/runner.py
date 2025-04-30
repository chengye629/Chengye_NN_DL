import numpy as np
import os
from tqdm import tqdm
from .op import Dropout
from .augmentation import augment_batch

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, val_loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.val_loss_fn = val_loss_fn

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            # Set dropout layers to training mode
            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    layer.train()


            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                # pred_labels = np.argmax(logits, axis=1)
                # print("Batch class dist:", np.bincount(pred_labels))
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model15.pickle')
                self.save_model(save_path)
                print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        # Set dropout layers to eval mode
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                layer.eval()

        logits = self.model(X)
        loss = self.val_loss_fn.forward(logits, y)
        score = self.metric(logits, y)
        
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)


class RunnerEarlystop():
    """
    Modified Runner with Early Stopping but keeps the original RunnerM structure.
    """
    def __init__(self, model, optimizer, metric, loss_fn, val_loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.val_loss_fn = val_loss_fn

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        early_stop_patience = kwargs.get("early_stop_patience", True)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        early_stop_patience = 3
        best_score = 0
        no_improve_counter = 0

        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss:.4f}, score: {trn_score:.4f}")
                    print(f"[Dev] loss: {dev_loss:.4f}, score: {dev_score:.4f}")

            # Save best model
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model12.pickle')
                self.save_model(save_path)
                print(f"Best model updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                no_improve_counter = 0  # reset counter
            else:
                no_improve_counter += 1  # no improvement

            # Early stopping check
            if early_stop_patience is not None and no_improve_counter >= early_stop_patience:
                print(f"Early stopping triggered: no improvement for {early_stop_patience} evaluations.")
                return  # Exit train early

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.val_loss_fn.forward(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)

class RunnerM_Augmented():
    def __init__(self, model, optimizer, metric, loss_fn, val_loss_fn, batch_size=32, scheduler=None, use_augmentation=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.val_loss_fn = val_loss_fn

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

        self.use_augmentation = use_augmentation

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            if self.use_augmentation:  # <<< 开启增强
                X, y = augment_batch(X, y)

            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    layer.train()

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                if train_X.shape[0] == 0:
                    continue

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                self.loss_fn.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if iteration % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss:.4f}, score: {trn_score:.4f}")
                    print(f"[Dev] loss: {dev_loss:.4f}, score: {dev_score:.4f}")

            if dev_score > best_score:
                best_score = dev_score
                save_path = os.path.join(save_dir, 'best_model15.pickle')
                self.save_model(save_path)
                print(f"Best model updated: {best_score:.5f}")


        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.val_loss_fn.forward(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)
