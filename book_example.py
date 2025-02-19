import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os


# example from capter 3.3 in Applied Asymptotics: Case Studies in Small-Sampole Statistics

def r(theta, theta_hat):
    return np.sign(theta_hat-theta)*np.sqrt( 2*( theta_hat*np.log(theta_hat/theta) -(theta_hat-theta)))

def t(theta, theta_hat):
    return np.sqrt(theta_hat)*np.log(theta_hat/theta) # using the Wald statistic from the book
    # return (theta_hat-theta)/np.sqrt(theta_hat) # This is what I got by using the formula in 2.3 directly (incorrect)

def r_star(theta, theta_hat):
    return r(theta, theta_hat) +1/r(theta, theta_hat)*np.log(t(theta, theta_hat)/r(theta, theta_hat))


y_0 = 27
b = 6.7
y = np.linspace(-1, 30, 32)
savedir='Book_examples/'

os.makedirs(savedir, exist_ok=True)

poisson_B_cdf = stats.poisson.cdf(y, mu=b, loc=0)

plt.scatter(y,poisson_B_cdf, label='Poisson cdf. with POI b')
plt.grid(axis='x')
plt.ylabel('Cumulative distribution function')
plt.xlabel('y')

r_star_norm = stats.norm.cdf(r_star(b, y), loc=0)
r_star_norm_half = stats.norm.cdf(r_star(b, 0.5+y), loc=0)

plt.plot(y, r_star_norm, '--', label=r'$\Phi\{r*(y, b)\}$')
plt.plot(y, r_star_norm_half, ':', label=r'$\Phi\{r*(y+1/2, b)\}$')

plt.legend()
plt.savefig(savedir+'book_cdf.png')
plt.clf()


mus= np.linspace(-2, 40, 42*2+1)

poisson_upperLimit = stats.poisson.sf(y_0-1, mu=b+mus)
poisson_lowerLimit = stats.poisson.sf(y_0, mu=b+mus)

plt.ylabel('Survival function')
plt.xlabel(r'$\mu$')

r_star_sf = 1- stats.norm.cdf(r_star(b+mus, y_0), loc=0)

plt.plot(mus, poisson_upperLimit, '--', label=r'$\Pr(Y\geq y^0\vert\theta)$')
plt.plot(mus, r_star_sf, 'r-', label=r'1-$\Phi\{r*(y^0, \theta)\}$')
plt.plot(mus, poisson_lowerLimit, '--', label=r'$\Pr(Y> y^0\vert\theta)$')

plt.legend()
plt.savefig(savedir+'book_sig.png')
plt.clf()

