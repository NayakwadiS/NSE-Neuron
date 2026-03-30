from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model and return a metric (e.g., RMSE)."""
        pass

