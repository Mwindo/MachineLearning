# Regression is the initial example in both Dive into Deep Learning and Goodfellow.
# In this file:
# Start with the least-squares analytic model with a single independent variable,
# deriving the formula via simple calculus.
# Extend this to a model with an arbitrary number of independent variables.
# Then implement non-analytic versions with hyperparameters.

import torch
import pandas as pd

# Basic analytic least-squares linear regression model with 1 independent variable
class LinearRegressionSimple:

    def __init__(self, examples: list[tuple[float, float]]):
        self.examples = examples
        self.slope, self.intercept = LinearRegressionSimple.get_line(examples)

    @classmethod
    def from_csv(cls, file_path: str):
        data = pd.read_csv(file_path)
        examples = list(zip(data.iloc[:, 0], data.iloc[:, 1]))
        return cls(examples)

    @staticmethod
    def get_line(examples: list[tuple[float, float]]):
        # We want x, b for y* = mx + b.
        # Assume Gaussian noise/error, so minimize mean squared error.
        # This means minimizing sum of (y_i - (mx_i + b))^2 for i in [1, n].
        # Find critical values by taking deriviative of this with respect to m, b.
        mean_x = sum([example[0] for example in examples]) / len(examples)
        mean_y = sum([example[1] for example in examples]) / len(examples)
        m_numerator = sum([(example[0] - mean_x)*(example[1] - mean_y) for example in examples])
        m_denominator = sum([(example[0] - mean_x)**2 for example in examples])
        m = m_numerator / m_denominator
        b = mean_y - (m * mean_x)
        return m, b
    
    def r_squared(self):
        mean_y = sum([example[1] for example in self.examples]) / len(self.examples)
        numerator = sum([(example[1] - self.predict(example[0]))**2 for example in self.examples])
        denominator = sum([(example[1] - mean_y)**2 for example in self.examples])
        return 1 - (numerator / denominator)

    def predict(self, x: float):
        return self.slope * x + self.intercept

# Analytic linear regression with multiple independent variables
# Consider linear dependencies between variables, conditioning, etc.
class LinearRegressionAnalytic:

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        pass

class LinearRegressionNeuralNetwork:

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        pass

# Simple test
# Should get r-squared of 1 and accurate conversions
linear_regression_test = LinearRegressionSimple.from_csv('./datasets/FahrenheitToCelsius.csv')
print(linear_regression_test.predict(39))
print(linear_regression_test.r_squared())
