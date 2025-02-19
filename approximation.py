import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def r(theta, theta_hat):
    return np.sign(theta_hat-theta)*np.sqrt( 2*( theta_hat*np.log(theta_hat/theta) -(theta_hat-theta)))

def t(theta, theta_hat):
    return np.sqrt(theta_hat)*np.log(theta_hat/theta)

def r_star(theta, theta_hat):
    return r(theta, theta_hat) +1/r(theta, theta_hat)*np.log(t(theta, theta_hat)/r(theta, theta_hat))


def calculate_ratio(numerator):
    ratio = np.divide(numerator, modified_chi2_pdf, where=modified_chi2_pdf!=0)
    ratio[numerator == 0] = np.nan 
    ratio[np.isnan(ratio)] = 1  
    return ratio


b = 0.78
formatted_b = f"{b}".replace('.', 'p')
save_dir = f'./Poisson_approximation_b{formatted_b}/'
n = 10000
os.makedirs(save_dir, exist_ok=True)

# plot the events generated from a poisson with poi=b and n samples
dataset = stats.poisson.rvs(b, size=n)
_, bin_og,__ = plt.hist(dataset, bins=10, label='Dataset')
plt.title(f'Poisson Distribution background hypothesis b={str(b)} with {"{:,}".format(n)} events')
plt.legend()
plt.xlabel('y')
plt.ylabel(r'Events')
plt.savefig(save_dir+'distribution.png')
plt.close()

# Only showing unique values, to then plot these only (and not repeat) 
y, weights = np.unique(dataset, return_counts=True)


fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, figsize=(8,12), sharex=True, dpi=200)
mus= np.linspace(0, 30, 100)
for y0 in y:
    ax1.plot(mus-b, r(mus, y0)**2, label=rf'y = {y0}')
    ax2.plot(mus-b, r_star(mus, y0)**2, label=rf'y = {y0}')
    ax3.plot(mus-b, r(mus, y0+0.5)**2, label=rf'y = {y0}')
    ax4.plot(mus-b, r_star(mus, y0+0.5)**2, label=rf'y = {y0}')

ax2.axvline(0, linestyle='--', alpha=0.7)
ax3.axvline(0, linestyle='--', alpha=0.7)
ax4.axvline(0, linestyle='--', alpha=0.7)

ax1.axvline(0, linestyle='--', alpha=0.7, label=rf'H0 with $\theta={{b}}$')

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.65),
            ncol=4, fancybox=True, shadow=True)

ax4.set_xlabel(r'$\mu$')
ax1.set_ylabel(rf'$r(y, \mu + b)^2$')
ax2.set_ylabel(rf'$r^*(y, \mu + b)^2$')
ax3.set_ylabel(rf'$r(y+1/2, \mu + b)^2$')
ax4.set_ylabel(rf'$r^*(y+1/2, \mu + b)^2$')

plt.savefig(save_dir+'nlls_event.png')
plt.close()

# changing the test statistics to the hep convention that only cares about excesses
t_0 = np.where(y<b, 0, 1)*r(b, y)**2
t_0_s = np.where(y<b, 0, 1)*r_star(b, y)**2
t_0_mp = np.where(y<b, 0, 1)*r(b, y+0.5)**2
t_0_s_mp = np.where(y<b, 0, 1)*r_star(b, y+0.5)**2

fig, ax1 = plt.subplots(1,1)#, figsize=(8,10), dpi=150)
ax1.scatter(y, t_0, label=rf'$t_0 | y = r(y, b)^2$')
ax1.scatter(y, t_0_s, label=rf'$t_0^* | y = r^*(y, b)^2$')
ax1.scatter(y, t_0_mp, label=rf'$t_0^{{mp}}|y=r(y+1/2, b)^2$')
ax1.scatter(y, t_0_s_mp, label=rf'$t_0^{{*mp}}|y=r^*(y+1/2, b)^2$')
ax1.plot(y, t_0)
ax1.plot(y, t_0_s, ls='--')
ax1.plot(y, t_0_mp, ls='-.')
ax1.plot(y, t_0_s_mp, ls=':')
ax1.legend()
ax1.set_xlabel(r'Observed events [y]')
ax1.set_ylabel(r'NLL at null-hypothesis')
plt.savefig(save_dir+'NLL_null.png')
plt.close()


fig, [ax2, ax3] = plt.subplots(2,1, figsize=(8,8), dpi=150, height_ratios=[2, 1], sharex=True)

if n <6:
    bins = 10
else:
    bins = 50
val_t_mu,binner,__= ax2.hist(r(b, dataset)**2, bins=bins, label=rf'$f(t_0|y)$', density=True, histtype='step', alpha=0.5)
val_t_mu_s,_,__= ax2.hist(r_star(b, dataset)**2, bins=binner, label=rf'$f(t_0^*|y)$', density=True, histtype='step', ls='--', alpha=0.5)
val_t_mu_mp,_,__= ax2.hist(r(b, dataset+0.5)**2, bins=binner, label=rf'$f(t_0^{{mp}}|y)$', density=True, histtype='step', ls='-.', alpha=0.5)
val_t_mu_s_mp,_,__= ax2.hist(r_star(b, dataset+0.5)**2, bins=binner, label=rf'$f(t_0^{{*mp}}|y)$', density=True, histtype='step', ls=':', alpha=0.5)

first_bin_end = binner[1]
x = np.linspace(0, first_bin_end, 3)
chi2_pdf = stats.chi2.pdf(x, df=1)


h_mu= np.linspace(0, binner[-1],1000)
modified_chi2_pdf = np.piecewise(h_mu, 
                                [h_mu < first_bin_end, h_mu >= first_bin_end], 
                                [sum(chi2_pdf[1:]), lambda h_mu: stats.chi2.pdf(h_mu, df=1)])
ax2.plot(h_mu, modified_chi2_pdf, 'k--', label=r'pmf. $\chi^2_1$')
ax2.legend()
ax2.set_yscale('log')
ax2.set_ylabel(r'Density [$\rho$]')
ax3.set_xlabel(r'NLL at null-hypothesis')

x = binner[:-1]

modified_chi2_pdf = np.piecewise(x, 
                                [x < first_bin_end, x >= first_bin_end], 
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
ax3.set_xlim([-0.05,x[-1]])
ax3.legend(loc='lower right', ncols=4)
ax3.set_yscale('log')

ax3.set_ylabel(r'Ratio of densities')

fig.suptitle(rf'$\chi^2_1$ comparison with test-statitics approximations with {"{:,}".format(n)} events')


plt.tight_layout()
plt.savefig(save_dir+'NLL_densities.png')
plt.close()

#ratios of p-values
fig, ax1 = plt.subplots(1,1)#, figsize=(6,6), sharex=True)

poisson_upperLimit = stats.poisson.sf(y-1, mu=b)
chi2_pval = stats.chi2.sf(y, df=1)
poisson_lowerLimit = stats.poisson.sf(y, mu=b)

ax1.set_ylabel('p-value')
ax1.set_xlabel(r'y')

r_star_sf = stats.norm.sf(r_star(b, y-1), loc=0)
r_sf = stats.norm.sf(r(b, y-1), loc=0)
r_star_sf_mp = stats.norm.sf(r_star(b, y-0.5), loc=0)
r_sf_mp = stats.norm.sf(r(b, y-0.5), loc=0)

ax1.plot(y, r_sf, 'g-', label=r'1-$\Phi\{r(y, b)\}$')
ax1.plot(y, r_star_sf, 'r-', label=r'1-$\Phi\{r*(y, b)\}$')
ax1.plot(y, r_sf_mp, 'y-', label=r'1-$\Phi\{r(y+1/2, b)\}$')
ax1.plot(y, r_star_sf_mp, 'k-', label=r'1-$\Phi\{r*(y+1/2, b)\}$')
ax1.plot(y, poisson_upperLimit, '--', label=r'$\Pr(Y\geq y\vert b)$')
ax1.legend()

# Old relic where we compared the p-value of poisson vs chi^2_1, not in use
# ax2.plot(y, poisson_upperLimit/chi2_pval, '--', label=r'$\Pr(Y>y\vert b)$')
# ax2.plot(y, r_sf/chi2_pval, 'g-', label=r'1-$\Phi\{r(y, b)\}$')
# ax2.plot(y, r_star_sf/chi2_pval, 'r-', label=r'1-$\Phi\{r*(y, b)\}$')
# ax2.plot(y, r_sf_mp/chi2_pval, 'y-', label=r'1-$\Phi\{r(y+1/2, b)\}$')
# ax2.plot(y, r_star_sf_mp/chi2_pval, 'k-', label=r'1-$\Phi\{r*(y+1/2, b)\}$')
# ax2.plot(y, poisson_lowerLimit/chi2_pval, '-.', label=r'$\Pr(Y\geq y\vert b)$')
# ax2.set_ylabel(r'ratop of p-values f/$\chi^2_1$')
# # ax2.legend()

plt.savefig(save_dir+'p_values.png')
plt.clf()


fig, ax1 = plt.subplots(1,1, figsize=(6,6))
ax1.set_ylabel('p-value')
ax1.set_xlabel(r'$t_0|y$')

ax1.scatter(t_0, r_sf, label=r'1-$\Phi\{r(y, b)\}$')
ax1.scatter(t_0, r_star_sf, label=r'1-$\Phi\{r*(y, b)\}$')
ax1.scatter(t_0, r_sf_mp, label=r'1-$\Phi\{r(y+1/2, b)\}$')
ax1.scatter(t_0, r_star_sf_mp, label=r'1-$\Phi\{r*(y+1/2, b)\}$')
ax1.scatter(t_0, poisson_upperLimit, marker='+', color='k', label=r'$\Pr(Y\geq y\vert b)$')
ax1.plot(t_0, r_sf)
ax1.plot(t_0, r_star_sf, '--')
ax1.plot(t_0, r_sf_mp, '-.')
ax1.plot(t_0, r_star_sf_mp, ':')
ax1.plot(t_0, poisson_upperLimit, 'k--')
ax1.legend()
ax1.set_yscale('log')
plt.savefig(save_dir+'p_values_vs_nom_2NLL.png')
plt.close()


sig_r = stats.norm.isf(r_sf)
sig_r_mp = stats.norm.isf(r_sf_mp)
sig_r_star = stats.norm.isf(r_star_sf)
sig_r_star_mp = stats.norm.isf(r_star_sf_mp)
sig_UL = stats.norm.isf(poisson_upperLimit)


plt.scatter(y, sig_r, label=r'p = 1-$\Phi\{r(y, b)\}$')
plt.scatter(y, sig_r_star, label=r'p = 1-$\Phi\{r*(y, b)\}$')
plt.scatter(y, sig_r_mp, label=r'p = 1-$\Phi\{r(y+1/2, b)\}$')
plt.scatter(y, sig_r_star_mp, label=r'p = 1-$\Phi\{r*(y+1/2, b)\}$')
plt.scatter(y, sig_UL, marker='+', color='k', label=r'p = $\Pr(Y\geq y\vert b)$')
plt.plot(y, sig_r)
plt.plot(y, sig_r_star, '--')
plt.plot(y, sig_r_mp, '-.')
plt.plot(y, sig_r_star_mp, ':')
plt.plot(y, sig_UL, 'k--')
plt.legend()
plt.xlabel('Events in sample [y]')
plt.ylabel(r'Significance [$\Phi^{-1}(1-p)$]')
plt.savefig(save_dir+'significance.png')
plt.clf()



plt.scatter(y[np.where(sig_r>0.8)], sig_r[np.where(sig_r>0.8)], label=r'p = 1-$\Phi\{r(y, b)\}$')
plt.scatter(y[np.where(sig_r>0.8)], sig_r_star[np.where(sig_r>0.8)], label=r'p = 1-$\Phi\{r*(y, b)\}$')
plt.scatter(y[np.where(sig_r>0.8)], sig_r_mp[np.where(sig_r>0.8)], label=r'p = 1-$\Phi\{r(y+1/2, b)\}$')
plt.scatter(y[np.where(sig_r>0.8)], sig_r_star_mp[np.where(sig_r>0.8)], label=r'p = 1-$\Phi\{r*(y+1/2, b)\}$')
plt.scatter(y[np.where(sig_r>0.8)], sig_UL[np.where(sig_r>0.8)], marker='+', color='k', label=r'p = $\Pr(Y\geq y\vert b)$')
plt.plot(y[np.where(sig_r>0.8)], sig_r[np.where(sig_r>0.8)])
plt.plot(y[np.where(sig_r>0.8)], sig_r_star[np.where(sig_r>0.8)], '--')
plt.plot(y[np.where(sig_r>0.8)], sig_r_mp[np.where(sig_r>0.8)], '-.')
plt.plot(y[np.where(sig_r>0.8)], sig_r_star_mp[np.where(sig_r>0.8)], ':')
plt.plot(y[np.where(sig_r>0.8)], sig_UL[np.where(sig_r>0.8)], 'k--')
plt.legend()
plt.xlabel('Events in sample [y]')
plt.ylabel(r'Significance [$\Phi^{-1}(1-p)$]')
plt.ylim(bottom=1)
plt.savefig(save_dir+'significance_zoom.png')
plt.clf()


# Making markdown tables to be able to see the numbers
data_sf = {'r': r_sf, 'r*':r_star_sf, 'Mid-P r': r_sf_mp, 'Mid-P r*': r_star_sf_mp, r'$\sqrt{t_0\vert y}$':np.sqrt(t_0), r'Effective $\sqrt{t_0\vert y}$': sig_UL}
df= pd.DataFrame(data_sf)

latex_output = df.to_latex()
markdown_output = df.to_markdown()

with open(save_dir+'pvalues.tex', 'w') as file:
    file.write(latex_output)
with open(save_dir+'pvalues.txt', 'w') as file:
    file.write(markdown_output)

data_sf = {'$t_0': t_0, 'r*':t_0_s, 'Mid-P r': t_0_mp, 'Mid-P r*': t_0_s_mp, r'$\sqrt{t_0\vert y}$':np.sqrt(t_0), r'Effective $\sqrt{t_0\vert y}$': sig_UL}
df= pd.DataFrame(data_sf)
latex_output = df.to_latex()
markdown_output = df.to_markdown()

with open(save_dir+'test_stat.tex', 'w') as file:
    file.write(latex_output)
with open(save_dir+'test_stat.txt', 'w') as file:
    file.write(markdown_output)

data_sf = {'Mid-P r* (p-value)': r_star_sf_mp, 'Poisson p-value': poisson_upperLimit,r'$\sqrt{t_0\vert y}$':np.sqrt(t_0), r'Effective $\sqrt{t_0\vert y}$': sig_UL, r'$\sqrt{t_0^{*mp}\vert y}$':np.sqrt(t_0_s_mp), r'Effective $\sqrt{t_0^{*mp}\vert y}$': sig_r_star_mp, }
df= pd.DataFrame(data_sf)
latex_output = df.to_latex()
markdown_output = df.to_markdown()

with open(save_dir+'intersting_Vals.tex', 'w') as file:
    file.write(latex_output)
with open(save_dir+'interesting_Vals.txt', 'w') as file:
    file.write(markdown_output)

# Old relic where we looked at the MLE distribution for each model, this is not studied further

# from scipy.optimize import minimize_scalar

# def t_mu(mu):  
#     return r(b + mu, y_0) ** 2

# def t_mu_star(mu):  
#     return r_star(b + mu, y_0) ** 2

# def t_mu_mp(mu):  
#     return r(b + mu, y_0 + 0.5) ** 2

# def t_mu_star_mp(mu):  
#     return r_star(b + mu, y_0 + 0.5) ** 2
    
# hat_mu_t_mu = []
# hat_mu_t_mu_star = []
# hat_mu_t_mu_mp = []
# hat_mu_t_mu_star_mp = []

# for y_0 in y:
#     res_t_mu = minimize_scalar(t_mu, bounds=(0, 100), method='bounded')
#     res_t_mu_star = minimize_scalar(t_mu_star, bounds=(0, 100), method='bounded')
#     res_t_mu_mp = minimize_scalar(t_mu_mp, bounds=(0, 100), method='bounded')
#     res_t_mu_star_mp = minimize_scalar(t_mu_star_mp, bounds=(0, 100), method='bounded')

#     hat_mu_t_mu.append(res_t_mu.x)
#     hat_mu_t_mu_star.append(res_t_mu_star.x)
#     hat_mu_t_mu_mp.append(res_t_mu_mp.x)
#     hat_mu_t_mu_star_mp.append(res_t_mu_star_mp.x)
    
# hat_mu_t_mu = np.asarray(hat_mu_t_mu)
# hat_mu_t_mu_star = np.asarray(hat_mu_t_mu_star)
# hat_mu_t_mu_mp = np.asarray(hat_mu_t_mu_mp)
# hat_mu_t_mu_star_mp = np.asarray(hat_mu_t_mu_star_mp)

# plt.scatter(y, hat_mu_t_mu-b, label=r'r')
# plt.scatter(y, hat_mu_t_mu_star-b, label=r'$r^*$')
# plt.scatter(y, hat_mu_t_mu_mp-b, label=r'Mid-P r')
# plt.scatter(y, hat_mu_t_mu_star_mp-b, label=r'Mid-P $r^*$')
# plt.legend()
# # plt.yscale('log')
# plt.ylabel(r'$\hat\mu$')
# plt.xlabel('Observed events [y]')
# plt.savefig(save_dir+'plot_2.png')
# plt.close()

# _,bins_og,__ = plt.hist(hat_mu_t_mu-b, bins=10, weights=weights,  histtype='step', label=r'r', alpha=0.5)

# h_mu = np.linspace(0, bins_og[-1], 1000)

# plt.hist(hat_mu_t_mu_star-b, bins=bins_og, weights=weights,  label=r'$r^*$', histtype='step', ls='--', alpha=0.5)
# plt.hist(hat_mu_t_mu_mp-b, bins=bins_og, weights=weights,  label=r'Mid-P r', histtype='step', ls='-.', alpha=0.5)
# plt.hist(hat_mu_t_mu_star_mp-b, bins=bins_og, weights=weights,  label=r'Mid-P $r^*$', histtype='step', ls=':', alpha=0.5)

# # plt.plot(h_mu, stats.chi2.pdf(h_mu, df=1), 'k--', label=r'$\chi^2_1$')
# plt.ylabel(r'Events')
# plt.xlabel(r'$\hat\mu$')
# plt.legend()
# # plt.yscale('log')
# plt.savefig(save_dir+'plot_3.png', dpi=400)
# plt.close()


# plt.plot(y, r(hat_mu_t_mu, y)**2, label=rf'$r(y, b)$')
# plt.plot(y, r_star(hat_mu_t_mu_star, y)**2, label=rf'$r^*(y, b)$', ls='--')
# plt.plot(y, r(hat_mu_t_mu_mp, y+0.5)**2, label=rf'$r(y+1/2, b)$', ls='-.')
# plt.plot(y, r_star(hat_mu_t_mu_star_mp, y+0.5)**2, label=rf'$r^*(y+1/2, b)$', ls=':')
# plt.legend()
# plt.xlabel(r'Observed events [y]')
# plt.ylabel(r'NLL at null-hypothesis')
# plt.savefig(save_dir+'plot_4.png')
# plt.close()