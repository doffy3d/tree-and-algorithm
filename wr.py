import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


data = np.loadtxt('data_1.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

lr = LogisticRegression(penalty='none', max_iter=1000, solver='lbfgs')
lr.fit(X, y)

max_predictors = X.shape[1]
accuracies = []
feature_rankings = []

for num_predictors in range(1, max_predictors + 1):
    selected_predictors = X[:, :num_predictors]
    #dodajem krosval zbog acc
    cv_scores = cross_val_score(lr, selected_predictors, y, cv=5)
    mean_acc = cv_scores.mean()
    accuracies.append(mean_acc)
    #treninranje na ovom modelu
    lr.fit(selected_predictors, y)

    # vaznost na osnovu koef
    coeff = lr.coef_[0]
    feature_indices = sorted(range(len(coeff)), key=lambda i: abs(coeff[i]), reverse=True)
    feature_rankings.append(feature_indices)


plt.figure()
plt.stem(range(1, max_predictors + 1), accuracies)
plt.xlabel('Broj prediktora')
plt.ylabel('Tacnost')
plt.title('Tacnost zavisnosti prediktora')
plt.show()

coeff = lr.coef_[0]
sorted_indices = np.argsort(np.abs(coeff))[::-1]
print("Prediktori na osnovu vaznosti:")
for i, id in enumerate(sorted_indices, 1):
    print(f"{i}. Prediktor kolone {id + 1}")
print()