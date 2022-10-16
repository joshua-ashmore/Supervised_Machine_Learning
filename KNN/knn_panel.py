"""File which displays the KNN panels."""
from KNN.knn_functions import KNNPricePredictor
import param
import datetime as dt

class KNNPanel(param.Parameterized):
    """KNN panel class."""
    stock = param.Selector(default="SPY", objects=["SPY","TSLA","MSFT"])
