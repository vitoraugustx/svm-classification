import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv('Heart.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# drop NaN
df = df.dropna()

cols = ['ChestPain','Thal', 'AHD']

df[cols] = df[cols].apply(lambda x: pd.factorize(x)[0])

print(df.head())

choosenCols = ['MaxHR','Age']

x = df[choosenCols]
y = df['AHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



# we create 40 separable points
X, y = x_train.to_numpy(), y_train.to_numpy()

# fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel="", C = 0.0001)
clf = svm.SVC(kernel="rbf", gamma=0.01)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
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