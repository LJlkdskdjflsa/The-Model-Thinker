"""
Linear Models Implementation
From Chapter 3 of The Model Thinker

This module implements various linear models including:
- Simple linear regression
- Multiple linear regression
- Linear growth models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class LinearModelResults:
    """Container for linear model results"""
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    predictions: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None


class LinearModel:
    """
    Implementation of linear models from The Model Thinker

    Linear models help us understand relationships between variables
    and make predictions based on those relationships.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearModelResults:
        """
        Fit a linear model using ordinary least squares

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            LinearModelResults object containing model parameters
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T

        # Add intercept column
        n_samples = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        # Calculate coefficients using normal equation
        # β = (X'X)^(-1)X'y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        beta = np.linalg.solve(XtX, Xty)

        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.is_fitted = True

        # Calculate predictions and metrics
        predictions = self.predict(X)
        residuals = y - predictions

        # Calculate R-squared
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        return LinearModelResults(
            coefficients=self.coefficients,
            intercept=self.intercept,
            r_squared=r_squared,
            predictions=predictions,
            residuals=residuals
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model

        Args:
            X: Feature matrix for prediction

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T

        return X @ self.coefficients + self.intercept

    def visualize(self, X: np.ndarray, y: np.ndarray,
                  title: str = "Linear Model Fit") -> None:
        """
        Visualize the linear model fit

        Args:
            X: Feature matrix
            y: Target values
            title: Plot title
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Actual vs Predicted
        predictions = self.predict(X)
        axes[0].scatter(y, predictions, alpha=0.6)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Residuals
        residuals = y - predictions
        axes[1].scatter(predictions, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class LinearGrowthModel:
    """
    Linear growth model: y(t) = y₀ + rt

    Models constant additive growth over time
    """

    def __init__(self, initial_value: float, growth_rate: float):
        """
        Initialize linear growth model

        Args:
            initial_value: Starting value (y₀)
            growth_rate: Rate of growth per time period (r)
        """
        self.initial_value = initial_value
        self.growth_rate = growth_rate

    def predict(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate value at time t

        Args:
            t: Time point(s)

        Returns:
            Value(s) at time t
        """
        return self.initial_value + self.growth_rate * t

    def time_to_value(self, target_value: float) -> float:
        """
        Calculate time needed to reach target value

        Args:
            target_value: Desired value

        Returns:
            Time needed to reach target
        """
        if self.growth_rate == 0:
            return np.inf if target_value != self.initial_value else 0
        return (target_value - self.initial_value) / self.growth_rate

    def visualize(self, t_max: float = 10, n_points: int = 100) -> None:
        """
        Visualize the linear growth model

        Args:
            t_max: Maximum time to plot
            n_points: Number of points to plot
        """
        t = np.linspace(0, t_max, n_points)
        y = self.predict(t)

        plt.figure(figsize=(10, 6))
        plt.plot(t, y, 'b-', linewidth=2, label='Linear Growth')
        plt.axhline(y=self.initial_value, color='r', linestyle='--',
                   alpha=0.5, label=f'Initial Value = {self.initial_value}')

        # Add growth rate annotation
        mid_t = t_max / 2
        mid_y = self.predict(mid_t)
        plt.annotate(f'Growth Rate = {self.growth_rate:.2f}/period',
                    xy=(mid_t, mid_y), xytext=(mid_t + t_max/10, mid_y),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=10)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Linear Growth Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def demonstrate_linear_models():
    """Demonstration of linear models"""

    print("=" * 50)
    print("LINEAR MODELS DEMONSTRATION")
    print("=" * 50)

    # 1. Simple Linear Regression
    print("\n1. SIMPLE LINEAR REGRESSION")
    print("-" * 30)

    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1) * 2
    true_slope = 3
    true_intercept = 5
    noise = np.random.randn(n_samples) * 0.5
    y = true_slope * X.squeeze() + true_intercept + noise

    model = LinearModel()
    results = model.fit(X, y)

    print(f"True parameters: slope={true_slope}, intercept={true_intercept}")
    print(f"Estimated parameters: slope={results.coefficients[0]:.3f}, "
          f"intercept={results.intercept:.3f}")
    print(f"R-squared: {results.r_squared:.3f}")

    # 2. Linear Growth Model
    print("\n2. LINEAR GROWTH MODEL")
    print("-" * 30)

    growth_model = LinearGrowthModel(initial_value=100, growth_rate=10)

    # Predict values
    times = [0, 5, 10, 20]
    for t in times:
        value = growth_model.predict(t)
        print(f"Value at t={t}: {value:.2f}")

    # Time to reach target
    target = 250
    time_needed = growth_model.time_to_value(target)
    print(f"\nTime to reach {target}: {time_needed:.2f} periods")

    return model, growth_model


if __name__ == "__main__":
    linear_model, growth_model = demonstrate_linear_models()

    # Visualizations (commented out for non-interactive environments)
    # linear_model.visualize(X, y)
    # growth_model.visualize(t_max=20)