#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Declare X_train And X_test
X_train = np.array([
    [0.876,0.0],[1.046,0.5],
    [1.01,0.5],[1.034,0.5],
    [1.444,0.0],[1.462,0.0],
    [1.039,0.0],[1.033,0.0],
    [1.293,0.5],[1.286,1.0]
])

X_test = np.array([[1.495,0.5],[0.883,0.0]])

#Compute RBF Kernel Vector
gamma = 0.5

def rbf_kernel_vector(x_test, X_train, gamma):
    kernel_vector = np.array([
        np.exp(-gamma * np.sum((x_test - x_train) ** 2))
        for x_train in X_train
    ])
    return kernel_vector

kernel_vectors = []
for x_test in X_test:
    kernel_vector = rbf_kernel_vector(x_test, X_train, gamma)
    print(f"RBF kernel vector for {x_test}:", kernel_vector)
    kernel_vectors.append(kernel_vector)

diff = np.abs(kernel_vector[0]-kernel_vector[1])
print("gamma=%s diff=%s" % (gamma, diff))

#Show Heatmap Of RBF Kernel Vector
plt.figure(figsize = (8, 6))
sns.heatmap(kernel_vectors, annot = True, cmap = 'viridis', square = True, cbar = True)
plt.title("RBF Kernel Vectors")
plt.xlabel("X_train")
plt.ylabel("X_test")
plt.show()