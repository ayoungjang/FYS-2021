import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm

data = pd.read_csv('./data_problem1.csv')

plt.hist(data)
# plt.show()


# Training data for class C0 (Gamma distribution with known α = 2)
x0 = np.array(data)  
n0 = len(x0)

# Training data for class C1 (Gaussian distribution)
x1 = np.array(data) 
n1 = len(x1)

# Maximum likelihood estimation for β for class C0
alpha = 2  # Known value
beta_hat = (1 / (n0 * alpha)) * np.sum(x0)

# Maximum likelihood estimation for µ for class C1
mu_hat = (1 / n1) * np.sum(x1)

# Maximum likelihood estimation for σ^2 for class C1
sigma_squared_hat = (1 / n1) * np.sum((x1 - mu_hat)**2)

print("==========1A===========")
print("Maximum Likelihood Estimations:")
print("Beta (for class C0):", beta_hat)
print("Mu (for class C1):", mu_hat)
print("Sigma^2 (for class C1):", sigma_squared_hat)
print("=====================")


alpha = 2  # Known value
beta_hat = (1 / (len(x_train[0]) * alpha)) * np.sum(x_train[0])

# Maximum likelihood estimation for µ for class C1
mu_hat = (1 / len(x_train[1])) * np.sum(x_train[1])

# Maximum likelihood estimation for σ^2 for class C1
sigma_squared_hat = (1 / len(x_train[1])) * np.sum((x_train[1] - mu_hat)**2)

# Implement Bayes' classifier
def bayes_classifier(x, beta, mu, sigma_squared):
    # Calculate the posterior probabilities for both classes C0 and C1
    p_C0_x = gamma.pdf(x, alpha, scale=1/beta)  # Gamma PDF for C0
    p_C1_x = norm.pdf(x, loc=mu, scale=np.sqrt(sigma_squared))  # Gaussian PDF for C1
    
    # Classify based on the class with higher posterior probability
    if p_C0_x > p_C1_x:
        return 0  # Class C0
    else:
        return 1  # Class C1

# Test the classifier on the test data and calculate accuracy
correct_predictions = 0
total_predictions = len(x_test[0]) + len(x_test[1])

for x in x_test[0]:
    if bayes_classifier(x, beta_hat, mu_hat, sigma_squared_hat) == 0:
        correct_predictions += 1

for x in x_test[1]:
    if bayes_classifier(x, beta_hat, mu_hat, sigma_squared_hat) == 1:
        correct_predictions += 1

test_accuracy = correct_predictions / total_predictions

print("Test Accuracy:", test_accuracy)