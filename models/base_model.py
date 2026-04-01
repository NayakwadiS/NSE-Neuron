from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on the given data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the model and return a metric (e.g., RMSE)."""
        raise NotImplementedError

