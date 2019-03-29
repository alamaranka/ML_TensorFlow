import pandas as pd
from neural_nets import neural_nets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# read data
train = pd.read_csv('datasets/mnist/train.csv')
test = pd.read_csv('datasets/mnist/test.csv')

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train, y_train = scaler.fit_transform(train.drop('label', axis=1)), train[['label']].values

model = neural_nets.NeuralNetsTensorFlow(hidden_layers=(50, 20, 50), learning_rate=0.0025, epochs=5000,
                                         batch_size=250, n_steps=1, print_log=True)
model.train(x_train, y_train)

y_pred = model.one_hot_encoder.inverse_transform(model.predict(x_train))
print("Train accuracy score: {0:.2f}%".format(accuracy_score(y_train, y_pred) * 100))

