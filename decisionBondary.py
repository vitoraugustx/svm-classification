from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import utils.data_provider as data_provider

def plot_graph(income_x,income_y, fig_title, models_list):

    chosenCols = data_provider.get_chosen_cols()

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

    for clf, title, ax in zip(models_list, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            income_x,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=chosenCols[0],
            ylabel=chosenCols[0],
        )
        ax.scatter(X0, X1, c=income_y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    fig.canvas.manager.set_window_title(fig_title)
    plt.show()
    plt.close()

x_train, x_test, y_train, y_test = data_provider.get_data(numpy=True)

models = (
    svm.SVC(kernel="linear", C=0.0001),
    svm.SVC(kernel="rbf", gamma=0.01),
    svm.SVC(kernel="poly", degree=3, gamma=0.01),
)

models = (clf.fit(x_train, y_train) for clf in models)

plot_graph(x_test,y_test,'Teste', models)
plot_graph(x_train,y_train,'Treinamento', models)








