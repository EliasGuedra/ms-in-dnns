import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
data = load_wine()
X = data.data
y = data.target

# Standardize
X = StandardScaler().fit_transform(X)

