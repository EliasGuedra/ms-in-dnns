import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def poly(x, W):
    k = W.shape[-1]
    X = create_desingmatrix(x.T, k-1)
    return W@X.T

#Function has to wrok on numpy array
def generate_data(function = np.sin, n = 25, standard_deviation = 0.1, intervall = (0, 2*np.pi)):
    x = np.random.uniform(intervall[0], intervall[1], n)
    pure_signal = function(x)
    noise = np.random.normal(0, standard_deviation, n)
    return (x, pure_signal+noise)


def create_desingmatrix(x_train, k):
    X = np.zeros((x_train.shape[0], k+1))

    for i in range(k+1):
        X[:, i] = x_train**i
    return X

    
def fit_poly(x_train, y_train, k = 5):
    X = create_desingmatrix(x_train, k)
    Y = y_train

    #W = np.linalg.lstsq(X, Y, rcond=None)[0]
    W = Y.T@X@np.linalg.inv(X.T@X)
    return W


def ridge_fit_poly(x_train, y_train, k = 5, lamb = 1):
    X = create_desingmatrix(x_train, k)
    Y = y_train

    W = Y.T@X@np.linalg.inv(X.T@X + lamb*np.identity(k+1))
    return W


def mse_poly(x, y, W):
    Y = poly(x, W)
    mse = ((y - Y)**2).mean()
    return mse


def perform_cv(x, y, k=5, lamb=1, folds=10):
    if len(y) % folds != 0:
        raise ValueError(
            f"Number of samples ({len(y)}) must be divisible by folds ({folds})"
        )
    
    fold_size = int(len(y)/folds)

    total_mse = 0

    for fold in range(folds):
        x_train = np.concatenate([x[:fold*fold_size], x[(fold+1)*fold_size:]])
        x_test  = x[fold*fold_size : (fold+1)*fold_size]
        y_train = np.concatenate([y[:fold*fold_size], y[(fold+1)*fold_size:]])
        y_test  = y[fold*fold_size : (fold+1)*fold_size]

        W = ridge_fit_poly(x_train, y_train, k=k, lamb=lamb)

        total_mse += mse_poly(x_test, y_test, W)

    return total_mse/folds


if __name__ == "__main__":

    # 2a)
    np.random.seed(42)

    x_train, y_train = generate_data(n = 15, intervall = (0, 2*np.pi))
    x_test,  y_test  = generate_data(n = 10, intervall = (0, 2*np.pi))

    plt.scatter(x_train, y_train, label = "Train data", c = "red")
    plt.scatter(x_test , y_test,  label = "Test data",  c = "green")

    x = np.linspace(0, 2*np.pi, 1000)
    plt.plot(x, np.sin(x), label = "Pure sine", c = "black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Noise samples from $sin(x)$")
    plt.legend()
    plt.show()


    #2b)

    plt.scatter(x_train, y_train, label = "Train data", c = "red")
    plt.scatter(x_test , y_test,  label = "Test data",  c = "green")

    plt.plot(x, np.sin(x), label = "Pure sine", c = "black")

    W = fit_poly(x_train, y_train, k=3)
    X = create_desingmatrix(x, k=3)
    Y = W@X.T

    MSE = mse_poly(x_test, y_test, W)

    plt.plot(x, Y, label = f"Degree {3} polynomial fit. $MSE = {MSE:.4f}$", c = "cyan", alpha = 1)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Degree {3} polynomial fit. $MSE = {MSE:.4f}$")

    plt.show()



    #2c)
    np.random.seed(42)

    x_train, y_train = generate_data(n = 15, intervall = (0, 4*np.pi))
    x_test,  y_test  = generate_data(n = 10, intervall = (0, 4*np.pi))

    plt.scatter(x_train, y_train, label = "Train data", c = "red")
    plt.scatter(x_test, y_test, label = "Test data", c = "green")

    x = np.linspace(0, 4*np.pi, 1000)
    plt.plot(x, np.sin(x), label = "Pure sine", c = "black")

    mses = []

    for d in range(1, 16):
        W = fit_poly(x_train, y_train, k=d)
        X = create_desingmatrix(x, k=d)

        Y = W@X.T

        mses.append(mse_poly(x_test, y_test, W))

        print(f"Polynomial of degree {d} had mse: {mses[-1]}")

        plt.plot(x, Y, label = f"Degree {d} polynomial fit.", alpha = 0.1)

    plt.legend()
    plt.show()

    plt.plot(range(1, 16), np.log(mses))
    plt.title("MSE for polynomials of degree 1-15")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("MSE")
    plt.show()


    # Choosed k = 7
    k = 7
    W = fit_poly(x_train, y_train, k=k)
    X = create_desingmatrix(x, k=k)

    Y = W@X.T

    MSE = mse_poly(x_test, y_test, W)

    plt.scatter(x_train, y_train, label = "Train data", c = "red")
    plt.scatter(x_test , y_test,  label = "Test data",  c = "green")

    plt.plot(x, np.sin(x), label = "Pure sine", c = "black")
    plt.plot(x, Y, label = f"Degree {k} polynomial fit.", alpha = 1, c = "cyan")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Degree {k} polynomial fit. $MSE = {MSE:.4f}$")

    plt.legend()
    plt.show()


    # 2d) Ridge
    np.random.seed(42)

    x_train, y_train = generate_data(n = 15, intervall = (0, 4*np.pi))
    x_test,  y_test  = generate_data(n = 10, intervall = (0, 4*np.pi))

    fig, axs = plt.subplots(1,3, sharex=True, sharey=True)

    degrees = list(range(1, 21))
    lambdas = (10 ** np.linspace(-5, 0, 20))

    mses = np.zeros((len(degrees), len(lambdas)))

    for i, k in enumerate(degrees):
        for j, lamb in enumerate(lambdas):

            W = ridge_fit_poly(x_train, y_train, k=k, lamb = lamb)
            X = create_desingmatrix(x, k=k)

            Y = W@X.T

            mses[i,j] += mse_poly(x_test, y_test, W)

    log_mses = np.log(mses)
    axs[0].imshow(log_mses, aspect = "auto", extent = [-5, 0, degrees[0], degrees[-1]])
    axs[0].set_title(f"$n_{{test}}={len(y_test)}, n_{{train}}={len(y_train)}$")

    #More test_data
    np.random.seed(43)

    x_train, y_train = generate_data(n = 15,   intervall = (0, 4*np.pi))
    x_test,  y_test  = generate_data(n = 1000, intervall = (0, 4*np.pi))

    mses = np.zeros((len(degrees), len(lambdas)))

    for i, k in enumerate(degrees):
        for j, lamb in enumerate(lambdas):
            W = ridge_fit_poly(x_train, y_train, k=k, lamb = lamb)
            X = create_desingmatrix(x, k=k)
            Y = W@X.T
            mses[i,j] += mse_poly(x_test, y_test, W)

    log_mses = np.log(mses)
    axs[1].imshow(log_mses, aspect = "auto", extent = [-5, 0, degrees[0], degrees[-1]])
    axs[1].set_title(f"$n_{{test}}={len(y_test)}, n_{{train}}={len(y_train)}$")

    #More train_data
    np.random.seed(44)

    x_train, y_train = generate_data(n = 1500,   intervall = (0, 4*np.pi))
    x_test,  y_test  = generate_data(n = 1000, intervall = (0, 4*np.pi))

    mses = np.zeros((len(degrees), len(lambdas)))

    for i, k in enumerate(degrees):
        for j, lamb in enumerate(lambdas):
            W = ridge_fit_poly(x_train, y_train, k=k, lamb = lamb)
            mses[i,j] += mse_poly(x_test, y_test, W)

    log_mses = np.log(mses)
    axs[2].imshow(log_mses, aspect = "auto", extent = [-5, 0, degrees[0], degrees[-1]])
    axs[2].set_title(f"$n_{{test}}={len(y_test)}, n_{{train}}={len(y_train)}$")


    fig.supxlabel("Exponent on $\lambda$")
    fig.supylabel("Degree $k$")
    plt.show()


    # 2e)

    #Best values
    np.random.seed(1000)

    x, y = generate_data(n = 120,   intervall = (0, 4*np.pi))

    mses = np.zeros((len(degrees), len(lambdas)))

    for i, k in enumerate(degrees):
        for j, lamb in enumerate(lambdas):
            W = ridge_fit_poly(x, y, k=k, lamb = lamb)
            mses[i,j] += mse_poly(x, y, W)
    
    a, b = np.unravel_index(mses.argmin(), mses.shape)

    degree = degrees[a]
    lam    = lambdas[b]

    list_of_folds = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]

    mses = np.zeros((100, len(list_of_folds)))

    for i in tqdm(range(100)):

        np.random.seed(i+100)
        x, y = generate_data(n = 120,   intervall = (0, 4*np.pi))

        for j, folds in enumerate(list_of_folds):
            mse = perform_cv(x, y, k=degree, lamb=lam, folds=folds)
            mses[i, j] = mse
    
    standard_deviations = np.std(mses, axis = 0)
    means = np.mean(mses, axis = 0)

    plt.plot(list_of_folds, means, c = "red", label = "Mean MSE")    
    upper_bound = means+standard_deviations
    lower_bound = means-standard_deviations
    lower_bound[lower_bound<0] = 0
    plt.fill_between(list_of_folds, lower_bound, upper_bound, color = "blue", alpha = 0.5, label = "$\mu \pm \sigma$")
    plt.legend()
    plt.xlabel("Number of folds")
    plt.ylabel("MSE")

    plt.title("Avergage MSE over number of folds")
    #plt.yscale("log") 
    plt.show()