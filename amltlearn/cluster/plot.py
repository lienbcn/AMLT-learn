
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np


def test_plot():
    X, y = make_blobs(n_samples=500, centers=5, random_state=0)
    aLabels = np.unique(y)
    import matplotlib.pyplot as plt
    for iLabel in aLabels:
        tColour = tuple(np.random.rand(3))
        iLength = X[y == iLabel, 0].shape[0]
        plt.scatter(X[y == iLabel, 0], X[y == iLabel, 1], c=tColour, alpha=0.5)
    plt.show()