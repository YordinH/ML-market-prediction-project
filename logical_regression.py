import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List

def get_preprocessed_data(path: str):
    market_df = pd.read_csv(path)

    features = [
        'Change_ES', 'Change_YM', 'SMT_NQ_ES', 'SMT_NQ_YM', 'RS',
        'FVG_Bullish', 'FVG_Bearish', 'Sweep_High', 'Sweep_Low',
        'Bullish_Setup', 'Bearish_Setup'
    ]

    X = market_df[features].values
    y = market_df['Target'].values.reshape(-1,1)

    X_trn, X_tst, y_trn, y_tst = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.transform(X_tst)

    X_trn = np.hstack([np.ones((X_trn.shape[0], 1)), X_trn])
    X_tst = np.hstack([np.ones((X_tst.shape[0], 1)), X_tst])

    return X_trn, X_tst, y_trn, y_tst

def nll_loss(y: np.ndarray, probs: np.ndarray) -> float:
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(np.negative(z)))

def get_batches(data_len: int, batch_size: int = 32,) -> List[np.ndarray]:
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    batches = [indices[i:i+batch_size] for i in range(0, data_len, batch_size)]

    return batches

class LogisticRegression():
    def __init__(self, alpha: float, batch_size: int, epochs: int = 1, seed: int = 0):
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        self.W = None
        self.trn_loss = []
        self.vld_loss = []

    def fit(self, X: np.ndarray, y: np.ndarray, X_vld: np.ndarray = None, y_vld: np.ndarray = None) -> object:
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))

        for epoch in range(self.epochs):
            batch_ids = get_batches(n_samples, self.batch_size)
            for batch in batch_ids:
                X_b = X[batch]
                y_b = y[batch]
                probs = sigmoid(X_b @ self.W)
                gradient = X_b.T @ (probs - y_b) / y_b.shape[0]
                self.W -= self.alpha * gradient

            # Track training and validation loss
            train_probs = sigmoid(X @ self.W)
            self.trn_loss.append(nll_loss(y, train_probs))

            if X_vld is not None and y_vld is not None:
                val_probs = sigmoid(X_vld @ self.W)
                self.vld_loss.append(nll_loss(y_vld, val_probs))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = sigmoid(X @ self.W)
        return (probs >= 0.5).astype(int)
    
X_trn, X_tst, y_trn, y_tst = get_preprocessed_data("./data/ict_dataset.csv")

model = LogisticRegression(alpha=0.1, batch_size=64, epochs=100, seed=42)
model.fit(X_trn, y_trn, X_tst, y_tst)

y_hat = model.predict(X_tst)
accuracy = (y_hat == y_tst).mean()
print("Test Accuracy:", accuracy)