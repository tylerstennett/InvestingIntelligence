import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.lstm import create_sequences_reg

#LR_CONSIDERED_FEATURES = ['Open', 'Close']
LR_CONSIDERED_FEATURES = ['Close']

def linreg_load_data(stock_name='AAPL', limit=None):
    """
    Load data from CSV file
    :param stock_name: Stock to load
    :param limit: Limit the number of rows to load
    :return: Pandas DataFrame of stock data
    """
    file_path = f'data/{stock_name}.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)  # Set Date as row

    if limit:
        data.head(limit)

    return data

def linreg_load_dataset(data):
    dataset = data[LR_CONSIDERED_FEATURES].values.astype('float32')

    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    classifier_labels = np.where(scaled_data[1:, 0] > scaled_data[:-1, 0], 1, 0) # Find days where last day is greater than curr

    return scaled_data, classifier_labels, scaler


def train_linear_regression(x_train):
    """
    Train linear regression model
    :param x_train: Training features
    :param y_train: Training labels
    :return: Model
    """

    x_train, y_train = create_sequences_reg(x_train)
    x_train = x_train.reshape(x_train.shape[0], -1) # Remove the 1 feature dim for linreg
    y_train = y_train.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model

