import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import seaborn as sns


def plot_reg_predictions(stock_data, y_pred, y_actual, method:str):
    """
    Plot the actual stock prices and the predicted prices.

    :param stock_data: Original stock data DataFrame
    :param test_data: Test data used for predictions
    :param predictions: Model predictions
    :param scaler: The scaler used to normalize the data
    """

    prediction_length, _ = y_pred.shape


    test_dates = stock_data.index[-prediction_length:]

    #actual_prices = scaler.inverse_transform(y_actual)
    #predicted_prices = scaler.inverse_transform(y_pred)
    actual_prices = y_actual
    predicted_prices = y_pred

    plot_data = pd.DataFrame({
        'Actual': actual_prices.flatten(),
        'Predicted': predicted_prices.flatten()
    }, index=test_dates)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_data.index, plot_data['Actual'], label='Actual Price', color='blue')
    plt.plot(plot_data.index, plot_data['Predicted'], label='Predicted Price', color='red')

    plt.title(f'Stock Price Predictions for {method}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_pred, y_actual, method:str):
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    cm = cm[::-1] # Adjusting Axes for clarity
    class_names = ["Positive", "Negative"]

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', xticklabels=class_names[::-1], yticklabels=class_names)

    plt.xlabel("Predicted Upticks")
    plt.ylabel("Actual Upticks")
    plt.title(f"Confusion Matrix for {method}")
    plt.show()


def evaluate_class(y_pred, y_class):
    precision = precision_score(y_class, y_pred)
    accuracy = accuracy_score(y_class, y_pred)
    recall = recall_score(y_class, y_pred)
    f1 = f1_score(y_class, y_pred)

    return accuracy, precision, recall, f1


