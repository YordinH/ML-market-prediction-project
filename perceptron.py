import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(
    y: np.ndarray, 
    y_hat: np.ndarray, 
    class_name_key: Dict[int, str] = None
) -> pd.DataFrame:
    y =  y.flatten()
    y_hat = y_hat.flatten()
    cfm = confusion_matrix(y, y_hat)
    
    labels = np.sort(np.unique(y))
    if class_name_key is not None:
        classes = []
        for l in labels:
            class_name = class_name_key.get(l, l)
            classes.append(class_name)
        labels = classes
        
    columns, index = labels, labels
    cfm_df = pd.DataFrame(cfm, index=index, columns=columns)
    sns.heatmap(cfm_df, annot=True)

    return cfm_df

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:    
    y =  y.flatten() 
    y_hat = y_hat.flatten() 
    return np.sum(y==y_hat) / len(y)

def ppv(y: np.ndarray, y_hat: np.ndarray) -> float:
    y =  y.flatten() 
    y_hat = y_hat.flatten() 
    
    tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_hat).ravel()
    return tp / (tp+fp)

def tpr(y: np.ndarray, y_hat: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_hat).ravel()
    return tp / (tp + fn)

def tnr(y: np.ndarray, y_hat: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_hat).ravel()
    return tn / (tn + fp)

class Perceptron():
    def __init__(
        self, 
        alpha: float,
        seed: int = 0,
        epochs: int = 1,
    ):
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.w = None
        self.trn_acc = None
        self.vld_acc = None

    def fit(
         self, X: np.ndarray, 
         y: np.ndarray, 
         X_vld: np.ndarray=None, 
         y_vld: np.ndarray=None
     ) -> object:
        np.random.seed(self.seed) # Set seed for reproducibility
        self.trn_acc = []
        self.vld_acc = []
        self.w = np.random.rand(X.shape[1])
        y_bin = 2 * y.flatten() - 1
        for e in range(self.epochs):
            misclassified = 0
            for i in range(X.shape[0]):
                z = np.dot(self.w.T, X[i])
                y_hat = np.sign(z)
                if y_hat != y_bin[i]:
                    misclassified +=1
                    self.w += self.alpha * X[i] * y_bin[i]
            trn_preds = self.predict(X)
            trn_acc = accuracy(y, trn_preds)
            self.trn_acc.append(trn_acc)
            
            if X_vld is not None and y_vld is not None:
                vld_preds = self.predict(X_vld)
                vld_acc = accuracy(y_vld, vld_preds)
                self.vld_acc.append(vld_acc)
                
            if misclassified == 0:
                break;
            
        return self 
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.sign(np.dot(X, self.w))
        preds[preds <= 0] = 0 
        return preds.astype(int)
    
X_trn, X_tst, y_trn, y_tst = get_preprocessed_data("./data/ict_dataset.csv")

model = Perceptron(alpha=0.001, epochs=50, seed=42)
model.fit(X_trn, y_trn, X_tst, y_tst)

y_hat = model.predict(X_tst)
print("Perceptron Accuracy:", accuracy(y_tst, y_hat))
plot_confusion_matrix(y_tst, y_hat)

#Plot
plt.plot(model.trn_acc, label="Train Accuracy")
plt.plot(model.vld_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Perceptron Accuracy per Epoch")
plt.legend()
plt.show()

#Print results
print(f"TPR: {tpr(y_tst, y_hat)}")
print(f"TNR: {tnr(y_tst, y_hat)}")
print(f"PPV: {ppv(y_tst, y_hat)}")