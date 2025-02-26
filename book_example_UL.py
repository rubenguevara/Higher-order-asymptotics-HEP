import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Further comparisons of example from capter 3.3 in Applied Asymptotics: Case Studies in Small-Sampole Statistics

def r(theta, theta_hat):
    return np.sign(theta_hat-theta)*np.sqrt( 2*( theta_hat*np.log(theta_hat/theta) -(theta_hat-theta)))

def t(theta, theta_hat):
    return np.sqrt(theta_hat)*np.log(theta_hat/theta) # using the Wald statistic from the book
    # return (theta_hat-theta)/np.sqrt(theta_hat) # This is what I got by using the formula in 2.3 directly (incorrect)

def r_star(theta, theta_hat):
    return r(theta, theta_hat) +1/r(theta, theta_hat)*np.log(t(theta, theta_hat)/r(theta, theta_hat))

y_0 = 1
b = 1
savedir='Book_examples_UL_b1_y1/'
os.makedirs(savedir, exist_ok=True)


mus= np.linspace(-2, 40, 42*2+1)

poisson_upperLimit = stats.poisson.cdf(y_0, mu=b+mus)
plt.ylabel('Upper limit / cdf')
plt.xlabel(r'$\mu$')

r_star_sf = stats.norm.cdf(r_star(b+mus, y_0), loc=0)
r_sf = stats.norm.cdf(r(b+mus, y_0), loc=0)
r_star_sf_mp = stats.norm.cdf(r_star(b+mus, y_0+0.5), loc=0)
r_sf_mp = stats.norm.cdf(r(b+mus, y_0+0.5), loc=0)

plt.plot(mus, r_sf, 'g-', label=r'$\Phi\{r(y^0, \theta)\}$')

plt.plot(mus, r_star_sf, 'r-', label=r'$\Phi\{r*(y^0, \theta)\}$')
plt.plot(mus, r_sf_mp, 'y-', label=r'$\Phi\{r(y^0+1/2, \theta)\}$')
plt.plot(mus, r_star_sf_mp, 'k-', label=r'$\Phi\{r*(y^0+1/2, \theta)\}$')
plt.plot(mus, poisson_upperLimit, '--', label=r'$\Pr(Y\leq y^0\vert\theta)$')


# Find the mu value corresponding to 95% CL
cl_95 = 0.05
# Plot the 95% confidence level line
plt.axhline(y=cl_95, color='b', linestyle='--', label='95% CL')

# plt.axvline(x=mu_95, color='b', linestyle='--', label='95% CL $\\mu$')
# Annotate the plot with the found mu_95
# plt.annotate(f'95% CL: $\\mu$ = {mu_95:.2f}', xy=(mu_95, cl_95), xytext=(mu_95 + 1, 0.7),
                # arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)


plt.legend()
plt.savefig(savedir+'book_sig.png')
plt.clf()


#Zoom on the upper tail, as this is of interest in HEP due to signal excess!

mus= np.linspace(2, 5, 300)

poisson_upperLimit = stats.poisson.cdf(y_0, mu=b+mus)


mu_95 = mus[np.argmax(poisson_upperLimit <= cl_95)]


plt.ylabel('cdf')
plt.xlabel(r'$\mu$')

r_star_sf = stats.norm.cdf(r_star(b+mus, y_0), loc=0)
r_sf = stats.norm.cdf(r(b+mus, y_0), loc=0)
r_star_sf_mp = stats.norm.cdf(r_star(b+mus, y_0+0.5), loc=0)
r_sf_mp = stats.norm.cdf(r(b+mus, y_0+0.5), loc=0)



mu_95_r = mus[np.argmax(r_sf <= cl_95)]
mu_95_rs = mus[np.argmax(r_star_sf <= cl_95)]
mu_95_r_mp = mus[np.argmax(r_sf_mp <= cl_95)]
mu_95_rs_mp = mus[np.argmax(r_star_sf_mp <= cl_95)]

plt.plot(mus, r_sf, 'g-')
plt.plot(mus, r_star_sf, 'r-')
plt.plot(mus, r_sf_mp, 'y-')
plt.plot(mus, r_star_sf_mp, 'k-')
plt.plot(mus, poisson_upperLimit, '--')


plt.axhline(y=cl_95, color='b', linestyle='--', label='95% CL')

plt.axvline(x=mu_95_r, color='g', label=f'95% CL r: $\\mu$ = {mu_95_r:.2f}')
plt.axvline(x=mu_95_rs, color='r', label=f'95% CL $r^*$: $\\mu$ = {mu_95_rs:.2f}')
plt.axvline(x=mu_95_r_mp, color='y', label=f'95% CL Mid-P r: $\\mu$ = {mu_95_r_mp:.2f}')
plt.axvline(x=mu_95_rs_mp, color='k', label=f'95% CL Mid-P $r^*$: $\\mu$ = {mu_95_rs_mp:.2f}')
plt.axvline(x=mu_95, linestyle='--', label=f'95% CL: $\\mu$ = {mu_95:.2f}')

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig(savedir+'book_sig_zoom.png')
plt.clf()