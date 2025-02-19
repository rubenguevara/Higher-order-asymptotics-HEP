import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

"""
This code was made before using approximations! 
it is just a sanity check to see that the test-statistics of a Poisson model approaches a chi^2_1 distribution with many toys
Here it is 10000 x 1000 !
"""
save_dir = './anal_testing/'
os.makedirs(save_dir, exist_ok=True)

# Parameters for the Poisson distribution
lambda_true = 5  # true rate parameter for the Poisson distribution
n = 10000  # sample size

# Function to compute log-likelihood for Poisson distribution
def log_likelihood_poisson(data, lam):
    return np.sum(stats.poisson.logpmf(data, mu=lam))

# Simulation to generate LRT statistics
log_likelihood_ratios = []

for _ in range(1000):  # simulate 1000 realizations
    # Generate samples
    data_sim = np.random.poisson(lam=lambda_true, size=n)
    
    # Estimate MLE for lambda (sample mean in Poisson)
    hat_lambda_sim = np.mean(data_sim)
    
    # Compute log-likelihoods under the null and alternative hypotheses
    log_likelihood_null = log_likelihood_poisson(data_sim, lambda_true)
    log_likelihood_alt = log_likelihood_poisson(data_sim, hat_lambda_sim)
    
    # Calculate likelihood ratio statistic
    lambda_statistic = -2 * (log_likelihood_null - log_likelihood_alt)
    log_likelihood_ratios.append(lambda_statistic)

# Plot the poisson distribution
count, bins, ignored = plt.hist(data_sim, 30, density=True, label='Simulated events (normalized)')
bins = np.linspace(min(bins),max(bins), int(max(bins)+1))
plt.plot(bins, stats.poisson.pmf(bins, lambda_true), linewidth=2, color='r', label='True Distribution')
plt.title('Poisson Distribution')
plt.legend()
plt.xlabel('k')
plt.ylabel(r'$\rho$')
plt.savefig(save_dir+'true_distribution.png')
plt.clf()

# Plotting the empirical distribution of LRT statistics
plt.hist(log_likelihood_ratios, bins=50, density=True, alpha=0.5, color='g', label='Empirical LRT Distribution')
plt.xlabel(r'$\lambda$ $(-2 LLR)$')
plt.ylabel('Density')
plt.title('Poisson Log-Likelihood Ratio Test Statistics')

# Overlay with theoretical chi-squared distribution
df = 1  # degrees of freedom for the null hypothesis comparison
x = np.linspace(0, max(log_likelihood_ratios), 1000)
chi2_pdf = stats.chi2.pdf(x, df=df)

# Plot the chi-squared density function
plt.plot(x, chi2_pdf, 'r-', lw=2, label=f'$\\chi^2_{df}$')
plt.legend()
plt.savefig(save_dir+'LLR_test_stat.png')
plt.clf()
