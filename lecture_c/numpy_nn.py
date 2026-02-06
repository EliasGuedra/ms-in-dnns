import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class NPLinear():
    def __init__(self, input_size, output_size):

        bound = np.sqrt(2 / input_size)
        self.W = np.random.uniform(
            -bound, bound, size=(output_size, input_size)
        )
        self.b = np.random.uniform(
            -bound, bound, size=(output_size,)
        )

        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

        self.z_prev = None
        
        self.trainable = True

        self.name = f"Linear ({input_size}) --> ({output_size})"

    def forward(self, z):
        self.z_prev = z
        return z @ self.W.T + self.b

    def backward(self, loss_grad):
        self.W_grad = loss_grad.T @ self.z_prev
        self.b_grad = loss_grad.sum(axis=0)

        return loss_grad @ self.W

    def gd_update(self, lr):
        self.W -= self.W_grad * lr 
        self.b -= self.b_grad * lr 
        
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)


class NPReLU():
    def __init__(self):
        self.mask = None
        self.trainable = False
        self.name = f"ReLU"

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, loss_grad):
        return loss_grad * self.mask


class NPMSELoss():
    def __init__(self):
        self.diff = None
        self.trainable = False    
        self.name = f"MSE Loss"

    def forward(self, predictions, target):
        self.diff = predictions - target
        return np.mean(self.diff**2)

    def backward(self):
        return 2*self.diff / self.diff.size


class NPModel():
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_history = []
        self.val_loss_history = []
        self.name = "I have no name :("

    def forward(self, x):
        z = x.copy()
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def backward(self, predictions, targets):
        if self.loss_function is None:
            raise ValueError("Set a lossfunction before calculating backprop!")

        loss = self.loss_function.forward(predictions, targets)
        self.loss_history.append(loss)
        current_grad = self.loss_function.backward()
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
            
    def gd_update(self, lr):
        for layer in self.layers:
            if layer.trainable:
                layer.gd_update(lr)

    def validate(self, x, y):
        z = self.forward(x)
        loss = self.loss_function.forward(z, y)
        self.val_loss_history.append(loss)
        return self.val_loss_history[-1]

    def add(self, layer):
        self.layers.append(layer)




if __name__ == "__main__":

    N_TRAIN = 100
    N_TEST = 1000
    SIGMA_NOISE = 0.1

    np.random.seed(0xDEADBEEF)
    x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
    y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

    x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
    y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE

    N_EPOCHS = 100


    models = {}

    widths     = [2, 4, 8, 16]
    depths     = [1, 2, 4]
    start_lrs  = [0.1, 0.01, 0.001]
    lr_decays  = [0.9, 0.99, 0.999]


    for w, d, lr, lr_decay in product(widths, depths, start_lrs, lr_decays):

        MODEL_NAME = f"width-{w}-depth-{d}-lr-{lr}-decay-{lr_decay}"

        model = NPModel()
        model.name = MODEL_NAME

        model.add(NPLinear(1, w))
        model.add(NPReLU())

        for _ in range(d-1):
            model.add(NPLinear(w, w))
            model.add(NPReLU())  

        model.add(NPLinear(w, 1))
        model.loss_function = NPMSELoss()

        for epoch in range(N_EPOCHS):
            preds = model.forward(x_train)
            model.backward(preds, y_train)

            if (val_loss:=model.validate(x_test, y_test)) < 0.035:
                print(f"Reached {val_loss} in {epoch} epochs.")
                break

            model.gd_update(lr * (lr_decay**epoch))

        
        models[MODEL_NAME] = model
        print(MODEL_NAME, model.val_loss_history[-1])

    losses = []
    for name, model in models.items():
        losses.append(model.val_loss_history[-1])
        plt.plot(model.val_loss_history, label = name)
    plt.legend()
    plt.show()


    #Take the best one and try again to remove bias from lucky initialisation. 
    w = 16
    d = 2
    lr = 0.1
    le_decay = 0.99

    x = np.linspace(-np.pi, np.pi, 1000)[..., np.newaxis]

    MODEL_NAME = f"width-{w}-depth-{d}-lr-{lr}-decay-{lr_decay}"

    model = NPModel()
    model.name = MODEL_NAME

    model.add(NPLinear(1, w))
    model.add(NPReLU())

    for _ in range(d-1):
        model.add(NPLinear(w, w))
        model.add(NPReLU())  

    model.add(NPLinear(w, 1))
    model.loss_function = NPMSELoss()

    for epoch in range(N_EPOCHS):
        

        preds = model.forward(x_train)
        model.backward(preds, y_train)

        if (val_loss:=model.validate(x_test, y_test)) < 0.035:
            print(f"Reached {val_loss} in {epoch} epochs.")
            break

        model.gd_update(lr * (lr_decay**epoch))

        preds = model.forward(x)
        plt.plot(x , preds, c = "red", label = "Modelpredictions")
        plt.plot(x, np.sin(x), c = "black", label = "True sine")
        plt.title(f"Epoch: {epoch}, Loss: {model.val_loss_history[-1]}")
        plt.legend()
        #plt.savefig(f"plots/gif/epoch-{epoch}")
        plt.clf()


    

    model = models["width-16-depth-2-lr-0.1-decay-0.99"]

    x = np.linspace(-np.pi, np.pi, 1000)[..., np.newaxis]
    preds = model.forward(x)
    plt.scatter(x_train, y_train, c = "blue", label = "Training data")
    plt.plot(x, preds, c = "red", label = "Model predictions")
    plt.plot(x, np.sin(x), c = "black", label = "True sine")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Final model predictions ({model.name})")
    plt.savefig(f"plots/Final model predictions ({model.name}).png")
    plt.legend()
    plt.show()

    plt.plot(model.loss_history, label = "Training loss (MSE)")
    plt.plot(model.val_loss_history, label = "Validation loss (MSE)")
    plt.xlabel("EPOCH")
    plt.ylabel("MSE")
    plt.title(f"Final model loss ({model.name})")
    plt.legend()
    plt.savefig(f"plots/Final model loss ({model.name}).png")
    plt.show()