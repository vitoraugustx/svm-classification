import utils.data_provider as data_provider

x_train, x_test, y_train, y_test = data_provider.get_data()

# Plotting train and test data
import matplotlib.pyplot as plt

# plot train data
plt.scatter(x_train['MaxHR'], x_train['Age'], c=y_train, cmap='Accent_r')
plt.xlabel('MaxHR')
plt.ylabel('Age')
plt.title('Dados de treino')
plt.show()

# plot test data
plt.scatter(x_test['MaxHR'], x_test['Age'], c=y_test, s=30, cmap='Accent_r')
plt.xlabel('MaxHR')
plt.ylabel('Age')
plt.title('Dados de teste')
plt.show()

