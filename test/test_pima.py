from neural_nets import neural_nets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('datasets/pima-indians-diabetes.csv', header='infer')

encoder = LabelEncoder()
scaler = MinMaxScaler(feature_range=(-1, 1))

X = df.drop(['Class'], axis=1)
y = df[['Class']].values

X['Group'] = encoder.fit_transform(X['Group'])
X = scaler.fit_transform(X)

y_pred_cum = 0
run_time = 5

for i in range(run_time):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i*999)
    model = neural_nets.NeuralNetsTensorFlow(hidden_layers=(10,), learning_rate=0.001, epochs=3_000,
                                             batch_size=25, n_steps=2, print_log=False)
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_cum += accuracy_score(y_test, (y_pred >= .5).astype(int)) * 100

print("Accuracy score: {0:.2f}%".format(y_pred_cum / run_time))

