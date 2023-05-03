import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.model_selection import train_test_split

df = pd.read_csv('Heart.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

x = df[['RestBP','Chol']]
y = df['AHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Treina o modelo SVM com kernel linear, polinomial e radial
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Kernel linear
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Accuracy (linear):', accuracy_score(y_test, y_pred))

# # Kernel polinomial
# clf = SVC(kernel='poly')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print('Accuracy (poly):', accuracy_score(y_test, y_pred))

# # Kernel radial
# clf = SVC(kernel='rbf')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print('Accuracy (rbf):', accuracy_score(y_test, y_pred))

# Plotar resultados usando matplotlib
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    x,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()



