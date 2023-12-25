from pandas import read_csv
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


data = np.loadtxt('data_1.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

num_of_predictors = 13

coeff = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
ind = np.argpartition(abs(coeff), -num_of_predictors)[-num_of_predictors:]
X_red = X[:, ind]

abs_coeff = abs(coeff)
sorted_indices = np.argsort(abs_coeff[ind])[::-1]
sorted_ind = ind[sorted_indices]

print("Redosled prediktora na osnovu vaznosti:")
for i, id in enumerate(sorted_ind):
    print(f"{i + 1}. Prediktor od kolone {id + 1}")

plt.figure(1)
plt.stem(range(len(coeff)), coeff)
plt.xlabel('Redni broj prediktora')
plt.ylabel('Koeficijent korelacije')
plt.title('Koeficijenti prediktora')
plt.show()

plt.figure(2)
plt.stem(range(len(abs_coeff)), abs_coeff)
plt.xlabel('Redni broj prediktora')
plt.ylabel('Absolutni koeficijent korelacije')
plt.title('Absolutne vrednosti prediktora')
plt.show()
