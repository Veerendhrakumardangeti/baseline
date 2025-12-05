import numpy as np

def compute_difficulty(X):
    diff = np.var(X, axis=-1)   # (N,)
    return diff

if __name__ == "__main__":
    X = np.load("results/baseline/X_train.npy")
    y = np.load("results/baseline/y_train.npy")

    scores = compute_difficulty(X)
    np.save("results/baseline/difficulty.npy", scores)
    print("Saved difficulty scores.")
