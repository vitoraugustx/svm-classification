import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import utils.data_provider as data_provider

def plot_max_margin(k, x_data, y_data):
    if k == 'linear':
        clf = svm.SVC(kernel=k, C = 0.0001)
    elif k == 'rbf':
        clf = svm.SVC(kernel=k, gamma=0.01)
    elif k == 'poly':
        clf = svm.SVC(kernel=k, degree=3, gamma=0.01)
    else:
        raise Exception('Invalid kernel')
    
    clf.fit(x_data, y_data)

    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x_data,
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

x_train, x_test, y_train, y_test = data_provider.get_data(numpy=True)

plot = plot_max_margin('linear', x_test, y_test)
plot = plot_max_margin('rbf', x_test, y_test)
plot = plot_max_margin('poly', x_test, y_test)


