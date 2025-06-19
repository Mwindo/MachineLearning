# Regression is the initial example in both Dive into Deep Learning and Goodfellow.
# In this file:
# Start with the least-squares analytic model with a single independent variable,
# deriving the formula via simple calculus.
# Extend this to a model with an arbitrary number of independent variables.
# Then implement non-analytic versions with hyperparameters.

# The code here is pretty awful. It is just me mucking through things to make sure I understand concepts and basic implementation.
# Maybe I will come back and clean it up, but it's unlikely.

import torch
import pandas as pd
from enum import Enum
from dataclasses import dataclass


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

    # Also implement pcr

    @dataclass
    class MultipleRegressionOptions:
        drop_correlated_columns_min_correlation: float | None = None
        pcr_min_correlation: float | None = None # Principle Components Reduction (reduce to orthogonal components via PCA)
        ridge_reduction_parameter: float | None = None # L2 regularization
        # Good for smoothing out moderate collinearities, at the cost of bias.
        # You make X invertible no matter what.
        # Not necessarily good for duplicate columns or extremely high collinearity due to the bias it introduces.

    def __init__(
            self, X: torch.Tensor,
            Y: torch.Tensor,
            options: MultipleRegressionOptions | None = None
    ):
        # Y = XB + E, where X is the design matrix, B is the coefficient matrix, and E is the error matrix
        self.X = X
        self.Y = Y
        self.options = options
        self.mu = X.mean(dim=0, keepdim=True)
        self.sigma = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        self.column_mask = None
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
        
        def get_column_mask():
            # Create a column mask if we want to drop highly collinear data
            if self.column_mask is not None:
                return self.column_mask
            design_matrix = (self.X - self.mu) / self.sigma
            if self.options and self.options.drop_correlated_columns_min_correlation:
                covariance = design_matrix.T @ design_matrix / (design_matrix.shape[0] - 1)
                std = design_matrix.std(dim=0) + 1e-12
                correlation_matrix = covariance / std[:, None] / std[None, :]
                keep_mask = torch.ones(design_matrix.shape[1], dtype=torch.bool)
                # Work on the upper triangle only (excluding the diagonal)
                for i in range(correlation_matrix.shape[0]):
                    for j in range(i + 1, correlation_matrix.shape[1]):
                        if keep_mask[j] and torch.abs(correlation_matrix[i, j]) >= self.options.drop_correlated_columns_min_correlation:
                            # drop column j
                            keep_mask[j] = False
                self.column_mask = keep_mask
                return keep_mask
        
        design_matrix = standardize(X)
        if (column_mask := get_column_mask()) is not None:
           design_matrix = design_matrix[:, column_mask]
        return augment(design_matrix)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        options: MultipleRegressionOptions | None = None
    ):
        df = pd.read_csv(file_path)
        X_np = df.iloc[:, :-1].to_numpy()   # shape: (N, num_features)
        y_np = df.iloc[:, -1].to_numpy()    # shape: (N,)
        X = torch.from_numpy(X_np).float()  # (N, num_features)
        y = torch.from_numpy(y_np).float()  # (N,)
        return cls(X, y, options)
    
    def get_coefficient_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor
    ):
        num_examples = X.shape[0]
        if len(Y) != num_examples:
            raise Exception("Number of inputs X does not match number of outputs Y")
        X_aug = self.prepare_design_matrix(X)  # shape: (n, p+1)
            
        if self.options and self.options.ridge_reduction_parameter:
            # Do some stuff
            p1 = X_aug.shape[1]
            I = torch.eye(p1, dtype=X_aug.dtype, device=X_aug.device)
            I[0,0] = 0 # don’t penalize intercept
            ridge = self.options.ridge_reduction_parameter # use model selection to determine this
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
        print(self.prepare_design_matrix(X))
        return self.prepare_design_matrix(X) @ self.B

class LinearRegressionNeuralNetwork:

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        pass

# Simple test
# Should get r-squared of 1 and accurate conversions
linear_regression_test = LinearRegressionSimple.from_csv('./datasets/FahrenheitToCelsius.csv')
print('Simple regression tests')
print('  value:', linear_regression_test.predict(39))
print('  r^2:', linear_regression_test.r_squared())

# An absolutely stupid data set just to inspect that things are more or less working
linear_regression_multiple_test = LinearRegressionMultiple.from_csv('./datasets/StupidMultipleRegressionData.csv')
X_new = torch.tensor([
    [39.0,  5.5, 100],
])
print('Multiple regression tests no options')
print('  value:', linear_regression_multiple_test.predict(X_new))
print('  r^2:', linear_regression_multiple_test.r_squared())

# Another absolutely stupid data set, this time to show that ridge regression,
# while providing a terrible prediction, at least allows us to invert the design matrix
linear_regression_multiple_test = LinearRegressionMultiple.from_csv('./datasets/StupidMultipleRegressionDataCollinear.csv', LinearRegressionMultiple.MultipleRegressionOptions(ridge_reduction_parameter=1))
X_new = torch.tensor([
    [39.0,  5.5, 100, 78],
])
print('Multiple regression tests, ridge regression')
print('  value:', linear_regression_multiple_test.predict(X_new))
print('  r^2:', linear_regression_multiple_test.r_squared())

# The same absolutely stupid data set, this time to make sure that dropping
# the duplicate columns allows us to invert the design matrix and get a good prediction
linear_regression_multiple_test = LinearRegressionMultiple.from_csv('./datasets/StupidMultipleRegressionDataCollinear.csv', LinearRegressionMultiple.MultipleRegressionOptions(drop_correlated_columns_min_correlation=.9))
X_new = torch.tensor([
    [39.0,  5.5, 100, 78],
])
print('Multiple regression tests, dropping highly collinear columns')
print('  value:', linear_regression_multiple_test.predict(X_new))
print('  r^2:', linear_regression_multiple_test.r_squared())
print('  Column mask:', linear_regression_multiple_test.column_mask)