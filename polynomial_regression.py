import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read csv file and make pandas data
data = pd.read_csv('china_cases.csv')
print(data.head())

# x = data.iloc[:, 0:1].values
# y = data.iloc[:, 1].values

# Separate dataset into x and y columns
x = data['x'].values
y = data['y'].values

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# After the random split, the arrays are no longer sorted
# Sort y by x's indices then sort x
y_train = y_train[x_train[:,0].argsort()]
x_train = x_train[x_train[:, 0].argsort()]

# Create polynomial regression object and set degree
poly = PolynomialFeatures(degree=3)

# Fit training data to transform it
x_poly = poly.fit_transform(x_train)

# Create linear regression object
poly_reg = LinearRegression()

# Fit new x data and old y data to the model
poly_reg.fit(x_poly, y_train)

# R^2 score & coefficients
print(poly_reg.score(x_poly, y_train))
print(poly_reg.coef_)

# Styling
# rcParams['figure.figsize'] = 16, 8
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['lines.linewidth'] = 3

# Build our visualisation
plt.title('Disease cases regressed on days')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.scatter(x_train, y_train, c='#edbf6f', label='Testing data')
plt.scatter(x_test, y_test, c='#8acfd4', label='Training data')
plt.plot(x_train, poly_reg.predict(x_poly), c='#a3cfa3', label='Polynomial regression line')
# plt.plot(x, -101.88364142*x + 32.32990905*(x**2) - 0.96990761*(x**3))
plt.legend(loc="upper left")
plt.show()
