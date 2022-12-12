from main import compute_Rhats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

def plot_trajectory(Omegas, hs, title, burn_length=0, lower_bound_h=.6, upper_bound_h=.8, lower_bound_Omegam=.2, upper_bound_Omegam=.5, lab=None, label1=None, label2=None, logL=None):
    if label1 is None:
        label1 = r"$\Omega_m$"
    if label2 is None:
        label2 = r"$h$"
    if lab is not None:
        label1 = label1 + " (" + lab + ")"
        label2 = label2 + " (" + lab + ")"
    plt.plot(Omegas[burn_length:], label=label1)
    plt.plot(hs[burn_length:], label=label2)
    if lab is None:
        plt.legend()
    else:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3, fancybox=True, shadow=True)
    plt.xlabel("iteration")
    plt.title(title)

def plot_contour(Omegas, hs, title, logL, burn_length=0, lower_bound_h=.6, upper_bound_h=.8, lower_bound_Omegam=.2, upper_bound_Omegam=.5, overwrite=False, lab=None):
    hmin = lower_bound_h
    hmax = upper_bound_h
    Omin = lower_bound_Omegam
    Omax = upper_bound_Omegam
    hh = np.linspace(hmin, hmax, 30)
    OO = np.linspace(Omin, Omax, 30)
    OO_hh = np.meshgrid(OO, hh)
    OO_hh = np.array(OO_hh).reshape(2, -1).T
    likelihoods = [logL(O, h, fb=False) for O, h in OO_hh]
    plt.contourf(OO, hh, np.array(likelihoods).reshape(len(OO), len(hh)), 15)
    if not overwrite:
        plt.colorbar()
        plt.plot(Omegas[burn_length:], hs[burn_length:], '-d', alpha=.5, lw=1)
    else:
        plt.plot(Omegas[burn_length:], hs[burn_length:], '-d', alpha=.5, lw=1, label=lab)
        plt.legend()
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$h$")
    plt.title(title)

def plot_contour_and_trajectory(logL, Omegas, hs, burn_length=0, lower_bound_h=.6, upper_bound_h=.8, lower_bound_Omegam=.2, upper_bound_Omegam=.5, overwrite=False, label=None, title="Trajectory of parameters with the Metropolis-Hastings sampler"):
    if not overwrite:
        plt.figure(figsize=(14, 7))
    title = title + "\nafter burn-in phase" if burn_length > 0 else title
    plt.suptitle(title)
    plt.subplot(121)
    plot_contour(Omegas, hs, "2D trajectory in parameter space and values of\nthe log-likelihood in the 'fine parameter area'", logL, burn_length, lower_bound_h, upper_bound_h, lower_bound_Omegam, upper_bound_Omegam, overwrite, label)
    plt.subplot(122)
    plot_trajectory(Omegas, hs, "Trajectories of each parameter", logL, burn_length, lower_bound_h, upper_bound_h, lower_bound_Omegam, upper_bound_Omegam, label)
    if not overwrite:
        plt.show()

def plot_evolution_acceptance_rate(Omegas_list, hs_list, labels, start=0, end=None, title=None):
    if(end is None): end = len(Omegas_list[0])
    if(title is None):
        title = "Evolution of the acceptance rate"
    else:
        title = "Evolution of acceptance rate\n" + title
    Nsamples = end-start
    Naccepted = 0
    for Omegas, hs, label in zip(Omegas_list, hs_list, labels):
        Naccepted = 0
        acceptance_rates = []
        for i in range(start, end-1):
            if( (Omegas[i+1]!=Omegas[i]) and (hs[i+1]!=hs[i]) ): Naccepted += 1
            acceptance_rates.append(Naccepted/(i-start+1))
        plt.plot(acceptance_rates, label=label)
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("acceptance rate")
        plt.title(title)
    plt.show()


def evolution_of_inter_chains_variances(chains, start=0, end=None): # not optimal at all because we recompute the variances at each step
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nchains, Nsamples, Nparams = np.shape(chains)
    variances = np.zeros((Nsamples-2, Nparams))
    for i in range(2, Nsamples):
        variances[i-2] = inter_chains_variances(chains, start, i)
    return variances

def evolution_of_intra_chains_variances(chains, start=0, end=None): # not optimal
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nchains, Nsamples, Nparams = np.shape(chains)
    variances = np.zeros((Nsamples-2, Nparams))
    for i in range(2, Nsamples):
        variances[i-2] = intra_chains_variances(chains, start, i)
    return variances

def evolution_of_Rhats(chains, start=0, end=None):
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nchains, Nsamples, Nparams = np.shape(chains)
    Rhats = np.zeros((Nsamples-2, Nparams))
    for i in range(2, Nsamples):
        Rhats[i-2] = compute_Rhats(chains, start, i)
    return Rhats

def time_until_convergence_Rhat(RR, Rhat_threshold=1.1):
    """
    Computes the time until convergence of the Markov chain, by checking the Rhat values.
    """
    for i in range(len(RR)):
        if (RR[i][0] < Rhat_threshold) and (RR[i][1] < Rhat_threshold):
            return i
    return -1


def plot_evolution_Rhats(Rhats, labels, title=None):
    if title is not None:
        title = "Gelman-Rubin test: evolution of the PSRF\n" + title
    else:
        title = "Gelman-Rubin test: evolution of the PSRF"
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    plt.subplot(121)
    plt.title("All samples")
    for i in range(len(labels)):
        plt.plot(Rhats[:, i], label=labels[i])
    plt.xlabel("Samples")
    plt.ylabel(r"$\hat{R}$")
    plt.legend()
    time_until_11 = time_until_convergence_Rhat(Rhats, 1.1)
    print("Convergence reached after %d samples (threshold: Rhat < 1.1)" % (time_until_11))
    print("Convergence reached after %d samples (threshold: Rhat < 1.05)" % (time_until_convergence_Rhat(Rhats, Rhat_threshold=1.05)))
    print("Convergence reached after %d samples (threshold: Rhat < 1.01)" % (time_until_convergence_Rhat(Rhats, Rhat_threshold=1.01)))
    plt.subplot(122)
    plt.title("After rough convergence")
    for i in range(len(labels)):
        plt.plot(range(time_until_11, len(Rhats)), Rhats[time_until_11:, i], label=labels[i])
    plt.axhline(1.01, color="black", linestyle="--", label=r"$\hat R = 1.01$")
    plt.xlabel("Samples")
    plt.ylabel(r"$\hat{R}$")
    plt.legend()
    plt.show()

def plot_CI_ellipse(mean, covariance, n_sigma=None, ax=None, xrange=None, yrange=None, thickness=1, alpha=.5, fill_ellipses=False):
    if ax is None:
        ax = plt.gca()
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    sigmas = np.sqrt(vals)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    thick = thickness
    if n_sigma is not None:
        width, height = 2 * sigmas
        for nsig in range(1, n_sigma+1):
            ax.add_patch(Ellipse(mean, nsig * width, nsig * height, theta, color=colors[nsig-1], fill=fill_ellipses, alpha=alpha, label="{}$\sigma$".format(nsig), linewidth=thick))
    else:
        percentages = [68.3, 95.4, 99.7]
        ss = [2.279, 5.991, 9.210] # 68%, 95%, 99.7% (chi2)
        for n in range(3):
            width, height = 2 * sigmas * np.sqrt(ss[n])
            ax.add_patch(Ellipse(mean, width, height, theta, color=colors[n], fill=fill_ellipses, alpha=alpha, label="{}%".format(percentages[n]), linewidth=thick))

    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$h$")
    ax.set_title("Posterior likelihood contours (assumed Gaussian)")
    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)
    plt.legend()
    
def plot_contours(chain, BURN_IN, bw="auto", xmin=.15, xmax=.45, ymin=.65, ymax=.75, weights=None, plot_weights=True, n=1000, levels=[0.99, 0.95, 0.68], colors_list=['r', 'g', 'b', 'c', 'm', 'y', 'k'], title="Approx posterior 'likelihood' contours (based on\n ranks of gaussian kernel density estimates)", scatter=True):
    Omegas = chain[:, 0]
    hs = chain[:, 1]
    if weights is not None:
        weights = weights / np.max(weights)
    if bw == "auto":
        d = 2
        bw = len(Omegas)**(-1./(d+4))
    kde = gaussian_kde(np.vstack([Omegas[BURN_IN:], hs[BURN_IN:]]), weights=weights, bw_method=bw)
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)/np.sum(kde(positions).T)
    t = np.linspace(0, Z.max(), n)
    integral = ((Z >= t[:, None, None]) * Z).sum(axis=(1, 2))
    f = interp1d(integral, t)
    t_contours = f(np.array(levels))
    # plt.figure(figsize=(9, 5))
    # plt.contourf(X,Y,Z, levels=t_contours, colors=colors_list, alpha=.5)
    if weights is not None:
        s = 10
    else:
        s = 1
    if scatter:
        if plot_weights & (weights is not None):
            plt.scatter(Omegas[BURN_IN:], hs[BURN_IN:], c=weights, s=s, marker='o', alpha=0.2, cmap='viridis')
            cb = plt.colorbar(label='weights (normalized by max)')
            cb.set_alpha(1)
            cb.draw_all()
        else:
            plt.scatter(Omegas[BURN_IN:], hs[BURN_IN:], s=s, marker='o', alpha=0.2)
    ct = plt.contour(X,Y,Z, levels=t_contours, linewidths=1.5, colors=colors_list)
    plt.clabel(ct, fmt={t_contours[0]: '99%', t_contours[1]: '95%', t_contours[2]: '68%'}, colors=colors_list, fontsize=14, inline=True)
    plt.title(title)

def estimate_mean_and_covariance_of_parameters(chain, weights=None, burn_in=0):
    Omegas = chain[burn_in:, 0]
    hs = chain[burn_in:, 1]
    if weights is None:
        mean = np.array([np.mean(Omegas), np.mean(hs)])
        covariance = np.cov(Omegas, hs)
    else:
        mean = np.array([np.sum(weights*Omegas), np.sum(weights*hs)])
        covariance = np.cov(Omegas, hs, aweights=weights)
    return mean, covariance

def superimpose_posteriors_contours_on_true_likelihood(logL, chain, burn_in=0, assumed_Gaussian=True, bw="auto", weights=None, title="", xrange=None, yrange=None, lower_bound_h=.6, upper_bound_h=0.8, lower_bound_Omegam=0.2, upper_bound_Omegam=0.5):
    plt.figure(figsize=(13, 6))

    Omegas = chain[:,0]
    hs = chain[:,1]
    
    xrange = [min(Omegas[burn_in:]), max(Omegas[burn_in:])]
    yrange = [min(hs[burn_in:]), max(hs[burn_in:])]
    hmin = max(lower_bound_h, yrange[0])
    hmax = min(upper_bound_h, yrange[1])
    Omin = max(lower_bound_Omegam, xrange[0])
    Omax = min(upper_bound_Omegam, xrange[1])
    hh = np.linspace(hmin, hmax, 50)
    OO = np.linspace(Omin, Omax, 50)
    OO_hh = np.meshgrid(OO, hh)
    OO_hh = np.array(OO_hh).reshape(2, -1).T
    likelihoods = [logL(O, h, fb=False) for O, h in OO_hh]
    plt.contourf(OO, hh, np.array(likelihoods).reshape(len(OO), len(hh)), 10, alpha=.5, cmap='viridis', antialiased=True)
    plt.colorbar(label="'True' likelihood (fine sampling, arbitrary color scale)")
    
    plt.plot(Omegas[burn_in:], hs[burn_in:], '.', color='k', alpha=.2, markersize=1)
    if assumed_Gaussian:
        mean, covariance = estimate_mean_and_covariance_of_parameters(chain, weights, burn_in=burn_in)
        legend_line = plt.scatter( np.NaN, np.NaN, marker = '.', color='k', s=5, label='MCMC samples') # fake line for non-transparent legend
        plt.plot(mean[0], mean[1], 'x', color='r', markersize=10, label="Mean")
        plot_CI_ellipse(mean, covariance, xrange=xrange, yrange=yrange, alpha=1, thickness=1.2)
    else:
        plot_contours(chain, burn_in, bw, weights=weights, plot_weights=False, xmin=xrange[0], xmax=xrange[1], ymin=yrange[0], ymax=yrange[1], title="Approx posterior likelihood contours (based on gaussian kernel density estimates)", scatter=False)
    
    plt.show()

def gen_Gaussian_prior(mean=0.738, s=0.024):
    return lambda x: np.exp(-0.5 * (x - mean)**2 / s**2) #/ np.sqrt(2 * np.pi * s**2)

def get_weights(chain, prior_fct, burn_in=0):
    Omegas = chain[burn_in:, 0]
    hs = chain[burn_in:, 1]
    priors = prior_fct(hs)
    return priors/np.sum(priors)

def plot_densities(chain, burn_in, weights, mean, mean_sec5, bw="auto", title="Visualisation of estimated densities (smoothed with gaussian kernel) with and without prior on h"):
    Omegas = chain[:, 0]
    hs = chain[:, 1]
    cs = {0: 'g', 1: 'm'}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title)
    if bw == "auto":
        d = 2
        bw = len(Omegas)**(-1./(d+4))
    sns.kdeplot(Omegas[burn_in:], ax=axs[0], label="w/o prior", bw_method=bw, color=cs[0])
    sns.kdeplot(Omegas[burn_in:], ax=axs[0], label="w/ gaussian prior on h", weights=weights, bw_method=bw, color=cs[1])
    sns.kdeplot(hs[burn_in:], ax=axs[1], label="w/o pior", bw_method=bw, color=cs[0])
    sns.kdeplot(hs[burn_in:], ax=axs[1], label="w/ gaussian prior on h", weights=weights, bw_method=bw, color=cs[1])
    axs[0].axvline(mean[0], color=cs[1], linestyle='--', label=r"$\hat{\Omega}_m = %.3f$ (w/ prior)" % mean[0])
    axs[0].axvline(mean_sec5[0], color=cs[0], linestyle='--', label=r"$\hat{\Omega}_m = %.3f$ (w/o prior)" % mean_sec5[0])
    axs[1].axvline(mean[1], color=cs[1], linestyle='--', label=r"$\hat{h}$ = %.3f (w/ prior)" % mean[1])
    axs[1].axvline(mean_sec5[1], color=cs[0], linestyle='--', label=r"$\hat{h}$ = %.3f (w/o prior)" % mean_sec5[1])
    axs[0].set_xlabel(r"$\Omega_m$")
    axs[1].set_xlabel(r"$h$")
    axs[0].set_ylabel("arbitrary unit")
    axs[1].set_ylabel("arbitrary unit")
    axs[0].set_title("Estimated density of $\Omega_m$")
    axs[1].set_title("Estimated density of $h$")
    axs[0].legend()
    axs[1].legend()
    plt.show()

def plot_marginals(chain, burn_in=0, bins = 50, title=None, autolims=True, hmin=.65, hmax=.75, Omegamin=.15, Omegamax=.45):
    plt.figure(figsize=(16, 5))
    OO = chain[burn_in:, 0]
    hh = chain[burn_in:, 1]
    plt.subplot(1, 2, 1)
    if not autolims:
        n, x, _ = plt.hist(OO, bins=bins, range=(Omegamin, Omegamax), density=True)
    else:
        n, x = np.histogram(OO, bins=bins)
        cffOO = np.cumsum(n)/np.sum(n)
        Omegamin = x[np.where(cffOO > .001)[0][0]]
        Omegamax = x[np.where(cffOO > .999)[0][0]]
        n, x, _ = plt.hist(OO, bins=bins, range=(Omegamin, Omegamax), density=True)
    density = gaussian_kde(OO)
    plt.plot(x, density(x), color='r', lw=2)
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel("Density")
    plt.subplot(1, 2, 2)
    if not autolims:
        n, x, _ = plt.hist(hh, bins=bins, range=(hmin, hmax), density=True)
    else:
        n, x = np.histogram(hh, bins=bins)
        cffhh = np.cumsum(n)/np.sum(n)
        hmin = x[np.where(cffhh > .001)[0][0]]
        hmax = x[np.where(cffhh > .999)[0][0]]
        n, x, _ = plt.hist(hh, bins=bins, range=(hmin, hmax), density=True)
    density = gaussian_kde(hh)
    plt.plot(x, density(x), color='r', lw=2)
    plt.xlabel(r"$h$")
    plt.ylabel("Density")
    if title is not None:
        plt.suptitle(title)
    plt.show()


def plot_contour_and_trajectory_3dL(samples, burn_length=0, overwrite=False, label=None, title="Trajectory of parameters with the Metropolis-Hastings sampler"):
    Omegas = samples[burn_length:, 0]
    Omegal = samples[burn_length:, 1]
    hs = samples[burn_length:, 2]
    Omegak = Omegas + Omegal -1
    if not overwrite:
        plt.figure(figsize=(14, 14))
    title = title + "\nafter burn-in phase" if burn_length > 0 else title
    plt.suptitle(title)
    plt.subplot(221)
    if not overwrite:
        plt.plot(Omegas, hs, '-d', alpha=.5, lw=1)
    else:
        plt.plot(Omegas, hs, '-d', alpha=.5, lw=1, label=label)
        plt.legend()
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$h$")
    plt.title(title)
    plt.subplot(222)
    plot_trajectory(Omegas, hs, "Trajectories of each parameter", label)
    plt.subplot(223)
    if not overwrite:
        plt.plot(Omegas, Omegak, '-d', alpha=.5, lw=1)
    else:
        plt.plot(Omegas, Omegak, '-d', alpha=.5, lw=1, label=label)
        plt.legend()
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel(r"$\Omega_k$")
    plt.title(title)
    plt.subplot(224)
    plot_trajectory(Omegas, Omegak, "Trajectories - hi - of each parameter", label, label1=r"$\Omega_m$", label2=r"$\Omega_k$")
    plt.tight_layout()

    if not overwrite:
        plt.show()
