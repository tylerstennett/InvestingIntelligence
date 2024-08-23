import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

#SVM_CONSIDERED_FEATURES = ['Open', 'Close']
SVM_CONSIDERED_FEATURES = ['Close']
NUM_PCA_COMPONENTS = len(SVM_CONSIDERED_FEATURES)
SVM_CONSIDERED_TIME_SEQ = 20

def svm_load_dataset(stock_name='AAPL'):
    file_path = f"data/{stock_name}.csv"
    data = pd.read_csv(file_path)

    dataset = data[SVM_CONSIDERED_FEATURES].values.astype('float32')

    new_data = []
    labels = []
    for i in range(len(dataset) - SVM_CONSIDERED_TIME_SEQ):
        new_data.append(dataset[i:i + SVM_CONSIDERED_TIME_SEQ])
        labels.append((dataset[i + SVM_CONSIDERED_TIME_SEQ] > dataset[i + SVM_CONSIDERED_TIME_SEQ - 1]).astype('int'))
    new_data = np.array(new_data)
    labels = np.array(labels)

    dataset = np.squeeze(new_data, axis=-1) # Should be (N,20)
    labels = np.squeeze(labels, axis=-1) # Should be (N,)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    return scaled_data, labels, scaler

def run_pca(data):
    N, D = data.shape
    pca = PCA(n_components=D)
    pca_data = pca.fit_transform(data)
    return pca_data

def svm_split_data(data, target, test_size=0.2):
    # Split data into training and validation sets.
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size, shuffle=False)
    return train_data, test_data, train_target, test_target

def create_svm(train_data, train_target, do_grid = False):
    if do_grid:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(train_data, train_target)
        return grid.best_estimator_
    else:
        svc = SVC(C=1, gamma=0.01, kernel='rbf')
        svc.fit(train_data, train_target)
        return svc

def evaluate_svm(predictions, test_target):
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions)
    recall = recall_score(test_target, predictions)
    f1 = f1_score(test_target, predictions)
    return accuracy, precision, recall, f1

# OLD ===============================================================================================================

def preprocess_stock_data(data):
    # Calculate means and standard deviations for open and close prices.
    means = data[['Open', 'Close']].mean()
    stds = data[['Open', 'Close']].std()

    # Normalize the data.
    scaler = StandardScaler()
    data[['Open', 'Close']] = scaler.fit_transform(data[['Open', 'Close']])

    return data, means, stds

def apply_pca_old(data, n_components=2):
    # Select features for PCA which are open and cose prices.
    features = data[['Open', 'Close']]
    
    # Apply PCA.
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    
    # Create a DataFrame with the principal components.
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    return pca_df

def create_train_test_split(data, target, test_size=0.2, random_state=42):
    # Split data into training and validation sets.
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return train_data, test_data, train_target, test_target

def train_and_evaluate_svm(train_data, train_target, test_data, test_target):
    # Hyperparameter tuning using GridSearchCV to ensure effective cross validation.
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(train_data, train_target)
    
    best_model = grid.best_estimator_
    predictions = best_model.predict(test_data)

    # Model results.
    conf_matrix = confusion_matrix(test_target, predictions)
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions)
    recall = recall_score(test_target, predictions)

    return conf_matrix, accuracy, precision, recall, best_model

def main(csv_file):
    # Read CSV file.
    data = pd.read_csv(csv_file)

    # Display first rows to ensure file is correct.
    print(data.head())

    # Data preprocessing.
    preprocessed_data, means, stds = preprocess_stock_data(data)

    # Normalization.
    print(preprocessed_data.head())
    print("Means:", means)
    print("Standard Deviations:", stds)

    # PCA.
    pca_data = apply_pca_old(preprocessed_data)

    # Display PCA results.
    print(pca_data.head())

    # Set up target variable: whether the closing price was higher than the opening price
    preprocessed_data['Target'] = (data['Close'] > data['Open']).astype(int)

    # Create train-test splits for PCA data.
    train_data, test_data, train_target, test_target = create_train_test_split(pca_data, preprocessed_data['Target'])

    # Display train and test splits results.
    print("Train Data:\n", train_data.head())
    print("Test Data:\n", test_data.head())

    # Train/Evaluate the SVC model.
    conf_matrix, accuracy, precision, recall, best_model = train_and_evaluate_svm(train_data, train_target, test_data, test_target)

    # Display results.
    print("Best Model:", best_model)
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Plotting the PCA data.
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(preprocessed_data['Open'], preprocessed_data['Close'], alpha=0.5)
    plt.title('Original Data: Open vs. Close')
    plt.xlabel('Open')
    plt.ylabel('Close')

    plt.subplot(1, 2, 2)
    plt.scatter(pca_data['PC1'], pca_data['PC2'], alpha=0.5)
    plt.title('PCA-Transformed Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.tight_layout()
    plt.show()

#if __name__ == "__main__":
#    csv_file = r"C:\Users\Gabri\Downloads\NVDA.csv"
    # Path to the CSV file that needs to be analyzed.
#    main(csv_file)

