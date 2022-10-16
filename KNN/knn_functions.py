"""Python file contains the functions required to return KNN strategy outputs."""

# Base Libraries
import pandas as pd
import numpy as np

# Financial Data
import yfinance as yf

# Trading Strategy
import pyfolio as pf

# Plotting
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score

# Classifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class KNNPricePredictor():
    def __init__(
        self,
        stock: str,
    ):
        self.stock = stock
        self.df = self.prepare_equity_data()
        self.X, self.y = self.predictors_and_labels()
        self.X_train, self.X_test, self.y_train, self.y_test, self.pipe = self.split_and_scale_data()
        self.best_params, self.best_score = self.cross_validation()

    def prepare_equity_data(self):
        """This downloads and prepares the equity data to be used by the KNN algorithm."""
        # Load data from yfinance
        df = yf.download(
                tickers = self.stock,
                period = "5y",
                interval = "1d",
                group_by = 'ticker',
                auto_adjust = False,
                prepost = True,
                threads = True,
                proxy = None
            )

        # Compute shifted return
        df['Forward Returns'] = np.log(df['Adj Close']).diff().shift(-1)

        # Remove any null values
        df = df.dropna()
        return df
    
    def predictors_and_labels(self):
        """Return predictors and target values."""
        df = self.df

        # Predictors
        df['O-C'] = df.Open - df.Close
        df['H-L'] = df.High - df.Low

        X = df[['O-C', 'H-L']].values
        
        # Labels
        y = np.where(df['Forward Returns']>=np.quantile(df['Forward Returns'], q=0.5), 1,-1)
        
        return X, y
    
    def split_and_scale_data(self):
        # Split data into testing and training
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
        
        # Scale and fit the model
        pipe = Pipeline([
            ("scaler", MinMaxScaler()), 
            ("classifier", KNeighborsClassifier())
        ]) 
        pipe.fit(X_train, y_train)
        
        return X_train, X_test, y_train, y_test, pipe
    
    def predict_model(self):
        # Predicting the test dataset
        self.init_y_pred = self.pipe.predict(self.X_test)
        self.acc_train = accuracy_score(self.y_train, self.pipe.predict(self.X_train))
        self.acc_test = accuracy_score(self.y_test, self.init_y_pred)
        return self.acc_train, self.acc_test, self.init_y_pred
    
    def cross_validation(self):
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5, gap=1)

        # Get parameters list
        self.pipe.get_params()

        # Perform Gridsearch and fit
        param_grid = {"classifier__n_neighbors": np.arange(1,51,1)}

        grid_search = GridSearchCV(self.pipe, param_grid, scoring='roc_auc', n_jobs=-1, cv=tscv, verbose=0)
        grid_search.fit(self.X_train, self.y_train)

        # Best Params
        best_params = grid_search.best_params_

        # Best Score
        best_score = grid_search.best_score_
        return best_params, best_score
    
    def fit_and_predict(self):
        # Instantiate KNN model with search param
        self.clf = KNeighborsClassifier(n_neighbors = self.best_params.get('classifier__n_neighbors'))

        # Fit the model
        self.clf.fit(self.X_train, self.y_train)

        # Predicting the test dataset
        self.cv_y_pred = self.clf.predict(self.X_test)

        # Measure Accuracy
        self.cv_acc_train = accuracy_score(self.y_train, self.clf.predict(self.X_train))
        self.cv_acc_test = accuracy_score(self.y_test, self.cv_y_pred)
        return self.cv_acc_train, self.cv_acc_test, self.clf, self.cv_y_pred

    def trading_strategy(self):
        df = self.df
        df['Signal'] = self.clf.predict(self.X)

        # Strategy Returns
        df['Strategy'] = df['Forward Returns'] * df['Signal'].fillna(0)

        # Localize index for pyfolio
        df.index = df.index.tz_convert('utc')
        return df
    
    def plot_confusion_matrix(self, y_pred):
        plot_confusion_matrix(self.pipe, self.X_test, self.y_test, cmap='Blues', values_format='.4g')
        plt.title('Confusion Matrix')
        plt.grid(False)
        
        print(classification_report(self.y_test, y_pred))
        
    def plot_random_prediction(self):
        # Random Prediction
        r_prob = [0 for _ in range(len(self.y_test))]
        r_fpr, r_tpr, _ = roc_curve(self.y_test, r_prob, pos_label=1)

        # Plot ROC Curve
        plot_roc_curve(self.pipe, self.X_test, self.y_test)
        plt.plot(r_fpr, r_tpr, linestyle='dashed', label='Random Prediction')
        plt.title('Receiver Operating Characteristic for Up Moves')
        plt.legend(loc=9)
        plt.show()
    
    def plot_clf_random_prediction(self):
        plot_roc_curve(self.clf, self.X_test, self.y_test)
        plt.title('Receiver Operating Characteristic for Up Moves')
        plt.legend(loc=9)
        plt.show()