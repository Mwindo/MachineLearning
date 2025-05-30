# Regression is the initial example in both Dive into Deep Learning and Goodfellow.
# In this file:
# Start with the least-squares analytic model with a single independent variable,
# deriving the formula via simple calculus.
# Extend this to a model with an arbitrary number of independent variables.
# Then implement non-analytic versions with hyperparameters.

import torch
import pandas as pd
from enum import Enum

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
        # This is functionally equivalent to minimizing the negative log likelihood,
        # which is the maximum likelihood estimation for the model.
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

# Analytic linear regression with multiple independent variables and one dependent variable.
# Consider linear dependencies between variables, conditioning, etc.
class LinearRegressionMultiple:

    # TODO: I have the skeleton of ridge reduction, but it isn't working right yet
    # Also implement pcr
    # Also fix this whole collinearity strategy thing since, e.g., PCT and Ridge are not mutually exlusive
    # Also, drop duplicate or highly collinear columns?

    class CollinearityStrategy(Enum):
        RIDGE = "ridge" # L2 regularization
        PCR = "pcr" # Principle Components Reduction (reduce to orthogonal components via PCA)
        NONE = "none"

    def __init__(
            self, X: torch.Tensor,
            Y: torch.Tensor,
            collinearity_strategy: CollinearityStrategy = CollinearityStrategy.NONE
    ):
        # Y = XB + E, whee X is the design matrix, B is the coefficient matrix, and E is the error matrix
        self.X = X
        self.Y = Y
        self.collinearity_stategy = collinearity_strategy
        self.mu = X.mean(dim=0, keepdim=True)
        self.sigma = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        self.B = self.get_coefficient_matrix(X, Y)
    
    def prepare_design_matrix(
        self,
        X: torch.Tensor,
    ):
        def standardize(X: torch.Tensor):
            # Center X to 1) improve numerical stability and 2) make intercept more interpretable
            # (since each intercept will say how far from the mean a given variable deviates)
            # And scale X so that variance = 1 across variables so that, e.g., regularization
            # does not dispropotionately affect variables with different variances
            return (X - self.mu) / self.sigma

        def augment(X: torch.Tensor):
            # Add a column of 1s to handle the intercept
            if X.dim() == 1:
                X = X.unsqueeze(1)
            n, _ = X.shape
            ones   = torch.ones(n, 1, dtype=X.dtype, device=X.device)
            return torch.cat([ones, X], dim=1)

        return augment(standardize(X))

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        collinearity_strategy: CollinearityStrategy = CollinearityStrategy.NONE
    ):
        df = pd.read_csv(file_path)
        X_np = df.iloc[:, :-1].to_numpy()   # shape: (N, num_features)
        y_np = df.iloc[:, -1].to_numpy()    # shape: (N,)
        X = torch.from_numpy(X_np).float()  # (N, num_features)
        y = torch.from_numpy(y_np).float()  # (N,)
        return cls(X, y, collinearity_strategy)
    
    def get_coefficient_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        num_examples = X.shape[0]
        if len(Y) != num_examples:
            raise Exception("Number of inputs X does not match number of outputs Y")
        X_aug = self.prepare_design_matrix(X)  # shape: (n, p+1)
            
        if self.collinearity_stategy == LinearRegressionMultiple.CollinearityStrategy.RIDGE:
            # Do some stuff
            p1 = X_aug.shape[1]
            I = torch.eye(p1, dtype=X_aug.dtype, device=X_aug.device)
            I[0,0] = 0 # don’t penalize intercept
            ridge = 1 # use model selection to determine this
            B = (X_aug.transpose(0,1) @ X_aug + ridge*I).inverse() @ X_aug.transpose(0,1) @ Y
        else:
            B = (X_aug.transpose(0,1) @ X_aug).inverse() @ X_aug.transpose(0,1) @ Y
        return B

    def r_squared(self):
        y_pred = self.prepare_design_matrix(self.X) @ self.B
        SS_residual = torch.sum((self.Y - y_pred) ** 2)
        SS_total = torch.sum((self.Y - torch.mean(self.Y)) ** 2)
        return 1 - (SS_residual / SS_total)

    def predict(self, X: torch.Tensor):
        return self.prepare_design_matrix(X) @ self.B

class LinearRegressionNeuralNetwork:

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        pass

# Simple test
# Should get r-squared of 1 and accurate conversions
linear_regression_test = LinearRegressionSimple.from_csv('./datasets/FahrenheitToCelsius.csv')
print(linear_regression_test.predict(39))
print(linear_regression_test.r_squared())

# An absolutely stupid data set just to inspect that things are more or less working
linear_regression_multiple_test = LinearRegressionMultiple.from_csv('./datasets/StupidMultipleRegressionData.csv')
X_new = torch.tensor([
    [39.0,  5.5, 100],
])
print(linear_regression_multiple_test.predict(X_new))
print(linear_regression_multiple_test.r_squared())

linear_regression_multiple_test = LinearRegressionMultiple.from_csv('./datasets/StupidMultipleRegressionDataCollinear.csv', collinearity_strategy=LinearRegressionMultiple.CollinearityStrategy.RIDGE)
X_new = torch.tensor([
    [39.0,  5.5, 100, 78],
])
print(linear_regression_multiple_test.predict(X_new))
print(linear_regression_multiple_test.r_squared())