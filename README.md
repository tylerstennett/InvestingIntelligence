# InvestingIntelligence

**InvestingIntelligence** is a quick and simple repository used for testing different machine learning models on 
stock market data with the purpose of predicting short-term stock price changes.

## Dataset

The dataset used in the project is the popular [S&P 500 stock data](https://www.kaggle.com/camnugent/sandp500) from 
Kaggle. To limit computational requirements, the following 10 stocks were chosen for the project: AAPL, AMZN, BRK-B,
DIS, GOOG, JPM, META, MSFT, NVDA, and TSLA. The dataset contains stock data from 2013 to 2018.

## Models

The following models were tested on the dataset:
- Linear Regression
- Support Vector Machine (SVM)
- Long Short-Term Memory (LSTM)

The range of models provides a sufficient variety of complexity. 

## Execution

The `models.ipynb` notebook contains the code used for executing the models. Each model is implemented in the `src` directory.

To use the Jupyter notebook and test the models, the following procedure can be followed:
1. Install the required packages (it is recommended to use the Conda environment in the `investing.yaml` file).
2. Run the Jupyter notebook cell-by-cell.

The notebook contains diagrams and success metrics (i.e., accuracy, precision, recall, and F1 score) for each model.
