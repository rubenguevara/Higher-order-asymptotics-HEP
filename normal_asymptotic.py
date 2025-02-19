import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import os

def r(theta, theta_hat):
    return np.sign(theta_hat-theta)*np.sqrt( 2*( theta_hat*np.log(theta_hat/theta) -(theta_hat-theta)))

def t(theta, theta_hat):
    return np.sqrt(theta_hat)*np.log(theta_hat/theta)

def r_star(theta, theta_hat):
    return r(theta, theta_hat) +1/r(theta, theta_hat)*np.log(t(theta, theta_hat)/r(theta, theta_hat))

#make test statistics into a function because then scipy optimize can find minima
def t_mu(mu):  
    return r(b + mu, y_0) ** 2

def t_mu_star(mu):  
    return r_star(b + mu, y_0) ** 2

def t_mu_mp(mu):  
    return r(b + mu, y_0 + 0.5) ** 2

def t_mu_star_mp(mu):  
    return r_star(b + mu, y_0 + 0.5) ** 2

def calculate_ratio(numerator):
    ratio = np.divide(numerator, modified_chi2_pdf, where=modified_chi2_pdf!=0)
    ratio[numerator == 0] = np.nan 
    ratio[np.isnan(ratio)] = 1  
    return ratio


savedir='CLT_diff_MLE/'
os.makedirs(savedir, exist_ok=True)

num_toys=int(1e1)
y_0 = 27
b = 6.7

"""
This is for an old test where we calcualted the MLE for every model, when it really should be the same! 
This has updated to do both things
"""

res_t_mu = minimize_scalar(t_mu, bounds=(0, 100), method='bounded')
res_t_mu_star = minimize_scalar(t_mu_star, bounds=(0, 100), method='bounded')
res_t_mu_mp = minimize_scalar(t_mu_mp, bounds=(0, 100), method='bounded')
res_t_mu_star_mp = minimize_scalar(t_mu_star_mp, bounds=(0, 100), method='bounded')

"""
You can choose to have the same MLE for all, or update it here
"""

# Different MLE values
hat_mu_t_mu = res_t_mu.x
hat_mu_t_mu_star = res_t_mu_star.x
hat_mu_t_mu_mp = res_t_mu_mp.x
hat_mu_t_mu_star_mp = res_t_mu_star_mp.x

# # Same MLE values
# hat_mu_t_mu = res_t_mu.x
# hat_mu_t_mu_star = res_t_mu.x
# hat_mu_t_mu_mp = res_t_mu.x
# hat_mu_t_mu_star_mp = res_t_mu_star.x

t_mu_hat = t_mu(hat_mu_t_mu)
t_mu_star_hat = t_mu_star(hat_mu_t_mu_star)
t_mu_mp_hat = t_mu_mp(hat_mu_t_mu_mp)
t_mu_star_mp_hat = t_mu_star_mp(hat_mu_t_mu_star_mp)


fig, [ax2, ax3] = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[2, 1], sharex=True)

toys_t_mu = stats.norm.rvs(loc=t_mu_hat, scale=1, size=num_toys)**2
toys_t_mu_star = stats.norm.rvs(loc=t_mu_star_hat, scale=1, size=num_toys)**2
toys_t_mu_mp = stats.norm.rvs(loc=t_mu_mp_hat, scale=1, size=num_toys)**2
toys_t_mu_star_mp = stats.norm.rvs(loc=t_mu_star_mp_hat, scale=1, size=num_toys)**2

bins = 70

bin_edges = np.histogram_bin_edges(toys_t_mu, bins=bins)

first_bin_end = bin_edges[1]
x = np.linspace(0, first_bin_end, 3)
chi2_pdf = stats.chi2.pdf(x, df=1)

# to integrate chi^2_1 up to the first bin

x = np.linspace(0, bin_edges[-1], 1000)
modified_chi2_pdf = np.piecewise(x, 
                                [x <= first_bin_end, x > first_bin_end], 
                                [sum(chi2_pdf[1:]), lambda x: stats.chi2.pdf(x, df=1)])

val_t_mu,_,__ = ax2.hist(toys_t_mu, bins=bin_edges, density=True, alpha=0.5, label=rf'$\mathcal{{N}}(t_\mu|\hat\mu_1, 1)^2$ with {str(num_toys)} toys', histtype='step')
val_t_mu_s,_,__ = ax2.hist(toys_t_mu_star, bins=bin_edges, density=True, alpha=0.5, label=rf'$\mathcal{{N}}(t_\mu^*|\hat\mu_2, 1)^2$ with {str(num_toys)} toys', histtype='step', ls='--')
val_t_mu_mp,_,__ = ax2.hist(toys_t_mu_mp, bins=bin_edges, density=True, alpha=0.5, label=rf'Mid-P $\mathcal{{N}}(t_\mu|\hat\mu_3, 1)^2$ with {str(num_toys)} toys', histtype='step', ls=':')
val_t_mu_s_mp,_,__ = ax2.hist(toys_t_mu_star_mp, bins=bin_edges, density=True, alpha=0.5, label=rf'Mid-P $\mathcal{{N}}(t_\mu^*|\hat\mu_4, 1)^2$ with {str(num_toys)} toys', histtype='step', ls='-.')

ax2.plot(x, modified_chi2_pdf, label=r'$\chi^2_1$ (Integrated up to first bin)', color='black', linestyle='--')

ax2.set_ylabel(r'Density [$\rho$]')
ax3.set_xlabel('X')
ax2.set_yscale('log')
ax2.legend()

#change to calculate densities
x = bin_edges[:-1]

modified_chi2_pdf = np.piecewise(x, 
                                [x <= first_bin_end, x > first_bin_end], 
                                [sum(chi2_pdf[1:]), lambda x: stats.chi2.pdf(x, df=1)])

ratio = calculate_ratio(val_t_mu)
ratio_s = calculate_ratio(val_t_mu_s)
ratio_mp = calculate_ratio(val_t_mu_mp)
ratio_s_mp = calculate_ratio(val_t_mu_s_mp)
integral = sum(abs(ratio-1))
integral_s = sum(abs(ratio_s-1))
integral_mp = sum(abs(ratio_mp-1))
integral_s_mp = sum(abs(ratio_s_mp-1))

ax3.step(x, ratio, alpha=0.5, where='post', label=rf'SoD: {integral:.5}')
ax3.step(x, ratio_s, alpha=0.5, where='post', ls='--', label=rf'SoD: {integral_s:.5}')
ax3.step(x, ratio_mp, alpha=0.5, where='post', ls='-.', label=rf'SoD: {integral_mp:.5}')
ax3.step(x, ratio_s_mp, alpha=0.5, where='post', ls=':', label=rf'SoD: {integral_s_mp:.5}')
ax3.axhline(y=1, xmin=0, xmax=x[-1], color='k', ls='--')
ax3.legend(loc='lower right', ncols=4)
ax3.set_xlim([-0.05,x[-1]])
ax3.set_yscale('log')


ax3.set_ylabel(r'Ratio of densities')

plt.tight_layout()
plt.savefig(f'{savedir}chi2_book_with_{str(num_toys)}_toys.png')
plt.close()