import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib

data = np.loadtxt('data_1.csv', delimiter=',')
pred = data[:, :-1]
y= data[:, -1].reshape(-1,1)

def column(matrix, p):
    return np.array([row[p] for row in matrix])

col_1 = 3#4
col_2 = 4#5

X1 = np.c_[column(pred, col_1), column(pred, col_2)]
X = StandardScaler().fit_transform(X1)

x_axis = column(X, 0)
y_axis = column(X, 1)

colors = []
for i in range(0, len(X)):
    if (y[i][0] == 0):
        colors.append('red')
    if (y[i][0] == 1):
        colors.append('blue')

plt.scatter(x_axis, y_axis, c=y.ravel(), alpha=0.9, cmap=matplotlib.colors.ListedColormap(colors), edgecolor='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
acc = []

for i in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Tacnost za dubinu {i}:", metrics.accuracy_score(y_test, y_pred) * 100)
    acc.append(metrics.accuracy_score(y_test, y_pred) * 100)

niz_x = np.linspace(1, 10, 10)
plt.stem(niz_x, acc)
plt.xlabel('Maximalna dubina stabla')
plt.ylabel('Tacnost[%]')

num_class = 2
plot_colors = "rb"

for i in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X, y)

    display = DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.RdYlBu, response_method="predict", xlabel='x1', ylabel='x2')

    plt.figure()
    display.plot(ax=plt.gca(), cmap=plt.cm.RdYlBu, alpha=0.7)

    for j, color in zip(range(num_class), plot_colors):
        idx = np.where(y == j)[0]
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor="black", s=15, label=f'Klasa {j+1}')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f"Dubina {i}")
    plt.legend()
    plt.show()