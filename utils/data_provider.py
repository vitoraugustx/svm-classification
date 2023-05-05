import pandas as pd
from sklearn.model_selection import train_test_split

chosenCols = ['MaxHR','Age']

def get_chosen_cols():
    global chosenCols
    return chosenCols

def get_data(numpy=False):
    global chosenCols
    df = pd.read_csv('src/Heart.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # drop NaN
    df = df.dropna()
    cols = ['ChestPain','Thal', 'AHD']

    df[cols] = df[cols].apply(lambda x: pd.factorize(x)[0])

   
    x = df[chosenCols]
    y = df['AHD']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    if numpy:
        return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    else:
        return x_train, x_test, y_train, y_test
