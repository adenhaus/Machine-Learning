import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter, defaultdict

# K-Means clustering implementation
# Get number of clusters from user
clust_num = int(input("Number of clusters: "))


# Function that reads data in from the csv files
def read_csv():
    x = []
    y = []
    countries = []
    x_label = ""
    y_label = ""
    with open('dataBoth.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = 0
        for row in reader:
            if lines >= 1:
                print(', '.join(row))
                x.append(float(row[1]))
                y.append(float(row[2]))
                countries.append(row[0])
                lines += 1
            else:
                x_label = row[1]
                y_label = row[2]
                print(', '.join(row))
                lines += 1
    return x, y, x_label, y_label, countries


# Declare variables for the data
x, y, x_label, y_label, countries = read_csv()

# Create 2D array from x and y values
X = np.vstack((x, y)).T


# Function to find clusters
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    # The main loop
    # This loop continues until convergence.
    # I could make it run a set number of times by changing
    # it to say while x > 5, for example, and removing the break
    print("\nConverging centres:")
    while True:
        # 2a. Assign labels based on closest center
        # I am using the pairwise_distances_argmin method to
        # calculate distances between points to centres
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

        # Print converging centres
        print(centers)
        print()

    return centers, labels


# Draw the scatter plot
centers, labels = find_clusters(X, clust_num)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('K-Means clustering of countries by birth rate vs life expectancy')
plt.xlabel(x_label)
plt.ylabel(y_label)

# Print number of countries in each cluster
print("\nNumber of countries in each cluster:")
print(Counter(labels))

# Get cluster indices
clusters_indices = defaultdict(list)
for index, c in enumerate(labels):
    clusters_indices[c].append(index)

# Print countries in each cluster and means
x = 0
while x < clust_num:
    print("\nCluster " + str(x + 1))
    print("----------")
    for i in clusters_indices[x]:
        print(countries[i])
    print("----------")
    print("Mean birth rate:")
    print(centers[x][0])
    print("Mean life expectancy:")
    print(centers[x][1])
    x+=1

# Show plot
plt.show()
