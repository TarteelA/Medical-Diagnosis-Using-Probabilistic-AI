#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Declare X_train And X_test
X_train = np.array([
    [0.284654,1],[0.368674,1],
    [0.332634,1],[0.368975,1],
    [0.410335,1],[0.357775,1],
    [0.211756,1],[0.163755,1],
    [0.231571,1],[0.271362,1]
])

X_test = np.array([[0.234589,1],[0.218164,1]])

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