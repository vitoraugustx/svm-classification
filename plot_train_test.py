import utils.data_provider as data_provider

x_train, x_test, y_train, y_test = data_provider.get_data()

# Plotting train and test data
import matplotlib.pyplot as plt

# Divide data by class
x_train_0 = x_train[y_train == 0]
x_train_1 = x_train[y_train == 1]

x_test_0 = x_test[y_test == 0]
x_test_1 = x_test[y_test == 1]

# plot train data
plt.scatter(x_train_0['MaxHR'], x_train_0['Age'], c='tab:blue', label='0 - No')
plt.scatter(x_train_1['MaxHR'], x_train_1['Age'], c='tab:red', label='1 - Yes')
plt.xlabel('MaxHR')
plt.ylabel('Age')
plt.title('Dados de treinamento')
plt.legend()
plt.show()

# plot test data
plt.scatter(x_test_0['MaxHR'], x_test_0['Age'], c='tab:blue', label='0 - No')
plt.scatter(x_test_1['MaxHR'], x_test_1['Age'], c='tab:red', label='1 - Yes')
plt.xlabel('MaxHR')
plt.ylabel('Age')
plt.title('Dados de teste')
plt.legend()
plt.show()


# # plot train data
# plt.scatter(x_train['MaxHR'], x_train['Age'], c=y_train, cmap='Accent_r')
# plt.xlabel('MaxHR')
# plt.ylabel('Age')
# plt.title('Dados de treino')
# plt.show()

# # plot test data
# plt.scatter(x_test['MaxHR'], x_test['Age'], c=y_test, s=30, cmap='Accent_r')
# plt.xlabel('MaxHR')
# plt.ylabel('Age')
# plt.title('Dados de teste')
# plt.show()

