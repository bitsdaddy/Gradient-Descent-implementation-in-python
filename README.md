# Gradient-Descent-implementation-in-python

## Overview

Gradient descent is a fundamental optimization algorithm widely used in machine learning and numerical optimization. It is employed to minimize a cost function iteratively by adjusting the parameters of a model. The primary objective is to find the values of these parameters that result in the minimum cost.

## How Gradient Descent Works

1. **Objective Function:**
   - The optimization process begins with defining an objective function, often referred to as a cost function. In machine learning, this function quantifies the error or cost associated with the model's predictions.

2. **Gradient Calculation:**
   - The gradient of the objective function is computed with respect to the model parameters. The gradient points in the direction of the steepest ascent, and the negative gradient points in the direction of the steepest descent.

3. **Parameter Update:**
   - The parameters are updated iteratively by moving in the opposite direction of the gradient. The learning rate determines the step size for each update.

4. **Convergence:**
   - The process continues until a convergence criterion is met, such as a small change in parameters or reaching a predefined number of iterations.

## Implementation in Python

In this project, a basic implementation of gradient descent has been developed using Python. The implementation includes the following components:

- **Objective Function:**
  - The specific objective function being minimized is \(3w₀² + 4w₁² - 5w₀ + 7\).

- **Gradient Function:**
  - The gradient of the objective function is computed to guide the parameter updates.

- **Convergence Check:**
  - A convergence check ensures that the optimization stops when parameters are sufficiently close to the optimal values.

- **Gradient Descent Function:**
  - The main optimization routine iteratively updates parameters and checks for convergence or divergence.

## Usage

To use the gradient descent implementation, simply provide an initial set of parameters, a learning rate, and a convergence threshold. The algorithm will iteratively update the parameters until convergence is achieved.

```python
import numpy as np

# Example usage with adaptive learning rate
initial_parameters = [5, 10]
learning_rate = 0.1
convergence_threshold = 1e-6

descent(initial_parameters, initial_parameters, learning_rate, convergence_threshold)
