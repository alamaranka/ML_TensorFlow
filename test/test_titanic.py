import pandas as pd
from neural_nets import neural_nets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from test.impute import impute_with_regression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# read data
train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')
passengerID = test['PassengerId']
# drop unnecessary predictors
train = train.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

# get the whole data
data = [train, test]
for df in data:
    # mapping 'Sex'
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # mapping 'Embarked'
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    # generate Title from Name
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = pd.Categorical(df['Title'], categories=df['Title'].unique()).codes.astype(int)

# drop 'Name'
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

# missing value imputation using logistic regression
print('Missing data imputation is being applied...')
train = impute_with_regression(train, LinearRegression())
test = impute_with_regression(test, LinearRegression())

x_train, y_train, x_test = train.drop('Survived', axis=1), train[['Survived']].values, test

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = neural_nets.NeuralNetsTensorFlow(hidden_layers=(10, 5), learning_rate=0.0035, epochs=5000,
                                         batch_size=30, n_steps=3, print_log=True)
model.train(x_train, y_train)
y_pred = (model.predict(x_test) >= .5).astype(int).squeeze()
print("Train accuracy score: {0:.2f}%".format(accuracy_score(y_train, (model.predict(x_train) >= .5).astype(int)) * 100))

# submission = pd.DataFrame({'PassengerId': passengerID, 'Survived': y_pred})
# file_name = 'titanic_submission.csv'
# submission.to_csv(file_name, sep=',', encoding='utf-8', index=False)

# gets 0.78947 on kaggle.com
