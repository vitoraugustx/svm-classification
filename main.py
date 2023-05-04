import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.model_selection import train_test_split

def plot_graph(income_x,income_y, fig_title):

    global models
    global choosenCols

    # title for the plots
    titles = (
        "SVM com kernel linear",
        "SVM com kernel radial",
        "SVM com kernel polinomial (grau 3)",
    )

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = income_x[:, 0], income_x[:, 1]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            income_x,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=choosenCols[0],
            ylabel=choosenCols[1],
        )
        ax.scatter(X0, X1, c=income_y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    fig.canvas.manager.set_window_title(fig_title)
    plt.show()

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

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 0.0001  # SVM regularization parameter
models = (
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="rbf", gamma=0.01),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

#plot_graph(X,y,'Treinamento')
plot_graph(x_test.to_numpy(),y_test,'Teste')








