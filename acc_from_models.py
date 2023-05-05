from sklearn import svm
import utils.data_provider as data_provider

x_train, x_test, y_train, y_test = data_provider.get_data(numpy=True)

models = (
    svm.SVC(kernel="linear", C = 0.0001),
    svm.SVC(kernel="rbf", gamma=0.01),
    svm.SVC(kernel="poly", degree=3, gamma=0.01),
)

models = (clf.fit(x_train, y_train) for clf in models)

for clf in models:
    print("Accuracy: ", round(clf.score(x_test, y_test) * 100, 2), "; Kernel: ", clf.kernel)
    