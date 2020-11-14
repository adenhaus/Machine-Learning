import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_diabetes
from sklearn import linear_model
import numpy as np

# Styling
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['lines.linewidth'] = 2


# Function to calculate gradient and y intercept
def func1():
    m = (np.mean(dx_train) * np.mean(dy_train) - np.mean(dx_train * dy_train)) / ((np.mean(dx_train)) ** 2 - np.mean(dx_train ** 2))
    b = np.mean(dy_train) - m * np.mean(dx_train)
    return m, b


# Load data set
d = load_diabetes()

# Select which features to use
d_X = d.data[:, 2]

# Split data into training and testing
dx_train = d_X[:-20]
dy_train = d.target[:-20]
dx_test = d_X[-20:]
dy_test = d.target[-20:]

# Create linear regression object
m, b = func1()

# Calculate mean square error
mse = np.mean((((m * dx_test) + b) - dy_test) ** 2)

# Calculate variance
score = (1 - ((dy_test - ((m * dx_test) + b)) ** 2).sum() / ((dy_test - dy_test.mean()) ** 2).sum())

# Print results
print(m)
print(mse)
print(score)

# Draw scatter plot and graph

plt.axis([-0.1, 0.1, 0, 400])
#plt.scatter(dx_train, dy_train, c='#9fd1ce', label="Training data")
plt.scatter(dx_test, dy_test, c='#ccd19f', label="Testing data")
plt.plot(dx_test, ((m * dx_test) + b), c='0.2', label="Line of Best Fit")
plt.legend(loc="upper left")

plt.show()
