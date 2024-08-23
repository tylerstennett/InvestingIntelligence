import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LSTM_CONSIDERED_FEATURES = ['Close'] # We can add additional features like volume, open, etc...
LSTM_SEQUENCE_TIMEFRAME = 20 # Number of days to consider for each sequence

def swish_activation(x, beta=1):
    # Swish activation function from this paper: https://arxiv.org/abs/1710.05941
    return tf.sigmoid(beta * x) * x

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish_activation)})

def lstm_load_data(stock_name='AAPL', limit=None):
    """
    Load data from CSV file
    :param stock_name: Stock to load
    :param limit: Limit the number of rows to load
    :return: Pandas DataFrame of stock data
    """
    file_path = f'data/{stock_name}.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True) # Set Date as row

    if limit:
        data.head(limit)

    return data

def lstm_load_dataset(data):
    """
    Convert the pandas DataFrame to a numpy array for training
    :param data: The pandas DataFrame
    :return: Numpy array of data
    """

    dataset = data[LSTM_CONSIDERED_FEATURES].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    classifier_labels = np.where(scaled_data[1:,0] > scaled_data[:-1,0], 1, 0)

    return scaled_data, classifier_labels, scaler

def create_sequences_reg(dataset):
    """
    Create labelled sequences from the dataset for LSTM
    :param dataset: The numpy array of the dataset
    :param seq_length: The number of time steps (DAYS) to consider
    :return:
    """
    x, y = [], []
    for i in range(len(dataset) - LSTM_SEQUENCE_TIMEFRAME - 1):
        end_idx = i + LSTM_SEQUENCE_TIMEFRAME
        x.append(dataset[i:end_idx])
        y.append(dataset[end_idx])
    return np.array(x), np.array(y)

def lstm_split_data(dataset, test_size = 0.2):
    """
    Split the data into training and testing sets
    :param data: The numpy array of the dataset
    :param test_size: The proportion of the dataset to include in the test split
    :return: The training and testing sets
    """
    train, test = train_test_split(dataset, test_size=test_size, shuffle=False)
    return train, test

def split_data_dual(dataset, labels, test_size = 0.2):
    train_size = int(len(dataset) * (1 - test_size))
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    train_labels, test_labels = labels[0:train_size], labels[train_size:len(dataset)]
    return train, test, train_labels, test_labels

def create_arima():
    pass

def create_lstm_reg():
    num_features = len(LSTM_CONSIDERED_FEATURES)

    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(LSTM_SEQUENCE_TIMEFRAME,num_features)),
        Dropout(0.1),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(16, activation='relu', return_sequences=False),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    # Note: We use relu to reduce vanishing gradient and Dropout layers to reduce overfitting

    return model

def train_lstm_reg(model, train_dataset, epochs=100, batch_size=32, learning_rate=1e-5):

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    x_train, y_train = create_sequences_reg(train_dataset)
    y_train = y_train.reshape(-1, 1)

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              shuffle=False,
              verbose=1)

    return model

def get_rmse_metrics(y_test, predictions):
    """
    Calculate the error metrics for the model predictions
    :param y_test: The actual test data
    :param predictions: The model predictions
    :param scaler: The MinMaxScaler used to normalize the data
    :return: The error metrics
    """
    y_actual = y_test
    y_predicted = predictions
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    return {"rmse": rmse}

def test_lstm():
    stock_data = lstm_load_data("AAPL", 500)
    dataset = lstm_load_dataset(stock_data)

    train_data, test_data = lstm_split_data(dataset)

    model = create_lstm_reg()
    model = train_lstm_reg(model, train_data, epochs=50, batch_size=32)

    x_test, y_test = create_sequences_reg(test_data)
    y_test = y_test.reshape(-1, 1)

    predictions = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")

    print(f"Predictions: {predictions[:5]}")
    print(f"Actual: {y_test[:5]}")
