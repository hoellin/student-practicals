from IPython. display import clear_output
import numpy as np
import jax.numpy as jnp
from numpy.fft import fft, ifft

CONST_c = 299792.458 # km/s (exact)
CONST_H0 = 70 # km/s/Mpc (typical)
CONST_h = 0.7 # (typical)
CONS_test = 2

def eta(a,Omegam,force_bounds=True):
    if force_bounds and ((Omegam<.2) or (Omegam>1)):
        print("Omegam out of range (error < 0.4% not guaranteed for this value)")
        raise ValueError
    s3 = (1-Omegam)/Omegam
    s = np.sign(s3)*np.power(np.abs(s3),1/3)
    return 2*np.sqrt(s3+1) * (1/a**4 - 0.1540*s/a**3 + 0.4304*s**2/a**2 + 0.19097*s3/a + 0.066941*s**4)**(-1/8)

def DL_star(z, Omegam, c=CONST_c, fb=True):
    return c*(1+z)/100 * (eta(1,Omegam,force_bounds=fb) - eta(1/(1+z),Omegam,force_bounds=fb))

def mu(z, Omegam, h=CONST_h, c=CONST_c, fb=True):
    return 25 - 5*np.log10(h) + 5*np.log10(DL_star(z, Omegam, c, fb))

lower_bound_Omegam = .2
upper_bound_Omegam = .5
lower_bound_h = .6
upper_bound_h = .8

def MH_sampler_gaussian(Nsamples, Omegam0, h0, sigma, logL, force_bounds=False):
    Omegas = [Omegam0]
    hs = [h0]
    for i in range(Nsamples):
        Omegam_new = Omegam0 + np.random.normal(0, sigma)
        h_new = h0 + np.random.normal(0, sigma)
        if force_bounds:
            if( (Omegam_new<lower_bound_Omegam) or (Omegam_new>upper_bound_Omegam) or (h_new<lower_bound_h) or (h_new>upper_bound_h) ):
                continue # warning: doing this means that the proposal distribution is truncated
                         #          AND that its truncation depends on the current value of the parameters
                         #          so this makes the sampler not truly a MH sampler
        if( np.log(np.random.uniform(0,1)) < logL(Omegam_new, h_new, fb=force_bounds) - logL(Omegam0, h0, fb=force_bounds) ): # ratio of likelihoods = diff of log-likelihoods
            Omegam0 = Omegam_new
            h0 = h_new
        Omegas.append(Omegam0)
        hs.append(h0)
    return Omegas, hs

def MH_sampler_top_hat(Nsamples, Omegam0, h0, sigma, logL, force_bounds=False):
    Omegas = [Omegam0]
    hs = [h0]
    for i in range(Nsamples):
        Omegam_new = Omegam0 + np.random.uniform(-sigma, sigma) # top-hat distribution with half-width sigma
        h_new = h0 + np.random.uniform(-sigma, sigma)
        if force_bounds:
            if( (Omegam_new<lower_bound_Omegam) or (Omegam_new>upper_bound_Omegam) or (h_new<lower_bound_h) or (h_new>upper_bound_h) ):
                continue # warning: doing this means that the proposal distribution is truncated
                         #          AND that its truncation depends on the current value of the parameters
                         #          so this makes the sampler not truly a MH sampler
        if( np.log(np.random.uniform(0,1)) < logL(Omegam_new, h_new, fb=force_bounds) - logL(Omegam0, h0, fb=force_bounds) ):
            Omegam0 = Omegam_new
            h0 = h_new
        Omegas.append(Omegam0)
        hs.append(h0)
    return Omegas, hs

def MH_sampler(Nsamples, Omegam0, h0, sigma, logL, proposal='top-hat', force_bounds=False):
    if(proposal=='top-hat'):
        return MH_sampler_top_hat(Nsamples, Omegam0, h0, sigma/0.68, logL, force_bounds) # /0.68 to make the behaviour of the top-hat and gaussian samplers similar
    elif(proposal=='gaussian'):
        return MH_sampler_gaussian(Nsamples, Omegam0, h0, sigma, logL, force_bounds)
    else:
        raise ValueError("proposal must be 'top-hat' or 'gaussian'")


def compute_acceptance_rate(Omegas, hs, start=0, end=None):
    if(end is None): end = len(Omegas)
    Nsamples = end-start
    Naccepted = 0
    for i in range(start, end-1):
        if( (Omegas[i+1]!=Omegas[i]) and (hs[i+1]!=hs[i]) ): Naccepted += 1
    return Naccepted/Nsamples

def merge_lists(Omegas_list, hs_list):
    chain = np.zeros((len(Omegas_list), len(Omegas_list[0]), 2))
    for i in range(len(Omegas_list)):
        chain[i, :, 0] = Omegas_list[i]
        chain[i, :, 1] = hs_list[i]
    return chain

def inter_chains_variances(chains, start=0, end=None): # between chains variance
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nchains, Nsamples, Nparams = np.shape(chains)
    variances = np.zeros(Nparams)
    for i in range(Nparams):
        means = np.mean(chains[:, :, i], axis=1)
        mean = np.mean(means)
        variances[i] = Nsamples/(Nchains-1) * np.sum((means[i] - mean)**2)
    return variances

def intra_chains_variances(chains, start=0, end=None): # within-chain variance
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nparams = np.shape(chains)[2]
    variances = np.zeros(Nparams)
    for i in range(Nparams):
        variances[i] = np.mean(np.var(chains[:, :, i], axis=1, ddof=1))
    return variances

def compute_Rhats(chains, start=0, end=None):
    if(end is None):
        end = len(chains[0])
    chains = chains[:, start:end, :]
    Nchains, Nsamples, Nparams = np.shape(chains)
    Rhat = np.zeros(Nparams)
    Ws = intra_chains_variances(chains, start, end)
    Bs = inter_chains_variances(chains, start, end)
    for i in range(Nparams):
        lb_var_hat = Ws[i]
        B = Bs[i]
        ub_var_hat = Nsamples/(Nsamples-1) * lb_var_hat + B/Nsamples
        Rhat[i] = np.sqrt(ub_var_hat/lb_var_hat)
    return Rhat

def HMC_sampler(dUdh, dUdOm, Om0, h0, H, M=None, prior=None, nsteps=5, timestep=.01, niter=5, randomized_N=False, method="randint"):
    if M is None:
        M = jnp.eye(2)
    x1 = [Om0]
    x2 = [h0]
    N = nsteps
    stepsize = timestep / nsteps
    for i in range(niter):
        clear_output(wait=True)
        # print("####################") # for debugging
        print("## Iteration: ", i, " ##")
        # print("####################")
        p = np.random.multivariate_normal([0,0], M, 1).T
        p1 = p[0]
        p2 = p[1]
        x1_star = x1[-1]
        x1_star0 = x1_star
        x2_star = x2[-1]
        x2_star0 = x2_star
        p1_star = p1[0]
        p1_star0 = p1[0]
        p2_star = p2[0]
        p2_star0 = p2[0]
        if randomized_N:
            if method=="randint":
                N = np.random.randint(1, nsteps)
        for j in range(N):
            # print("## Step ", j) # for debugging
            p1_star = p1_star - stepsize/2 * dUdOm(x1_star, x2_star)
            p2_star = p2_star - stepsize/2 * dUdh(x1_star, x2_star)
            x1_star = x1_star + stepsize * p1_star
            x2_star = x2_star + stepsize * p2_star
            p1_star = p1_star - stepsize/2 * dUdOm(x1_star, x2_star)
            p2_star = p2_star - stepsize/2 * dUdh(x1_star, x2_star)
        H_prev = H(x1_star0, x2_star0, jnp.array([p1_star0, p2_star0]), M, prior)
        H_new = H(x1_star, x2_star, jnp.array([p1_star, p2_star]), M, prior)
        ub = min(1, jnp.exp(-H_new + H_prev))
        # verify x values are not out of bounds, and compare to uniform
        if (x1_star > 0 and x1_star < 1 and x2_star > 0 and x2_star < 1) and (np.random.uniform() < ub):
            x1.append(x1_star)
            x2.append(x2_star)
        else:
            x1.append(x1[-1])
            x2.append(x2[-1])

    return np.array(x1, dtype="object").astype(float), np.array(x2, dtype="object").astype(float)

def autocorr(chain, start=0, end=-1):
    chain = chain[start:end]
    mu = np.mean(chain, axis=0)
    min_length = 2 * len(chain) - 1
    n = 2 ** int(np.ceil(np.log2(min_length))) # zero pad towards power of 2 for efficiency
    ft = fft(chain - mu, n)
    corr = ifft(ft*np.conjugate(ft), axis=0)[:len(chain)].real
    return corr/corr[0]

def compute_autocorr_time(chain, start=0, end=-1, M=None):
    chain = chain[start:end]
    if M is None:
        M = np.sqrt(len(chain))
    corr = autocorr(chain)
    return 0.5 + 2 * np.sum(corr[1:int(M)])

def compute_autocor_time_multiple_chains(chains, start=0, end=-1, M=None):
    return np.mean([compute_autocorr_time(chain, start=start, end=end, M=M) for chain in chains])

def MH_3d_sampler_gaussian(logL_3d, Om0, OL0, h0, niter=1000, sigma=0.01, M=None):
    Om = Om0
    OL = OL0
    h = h0
    samples = np.zeros((niter, 3))
    for i in range(niter):
        if M is None:
            M = np.eye(3)
        proposals = np.random.multivariate_normal([Om, OL, h], sigma**2*M)
        Om_new = proposals[0]
        OL_new = proposals[1]
        h_new = proposals[2]
        Om_new = np.clip(Om_new, 0, 1)
        OL_new = np.clip(OL_new, 0, 1)
        h_new = np.clip(h_new, 0, 1)
        logL_old = logL_3d(Om, OL, h)
        logL_new = logL_3d(Om_new, OL_new, h_new)
        if( np.log(np.random.uniform(0,1)) < logL_new - logL_old):
            Om = Om_new
            OL = OL_new
            h = h_new
        samples[i] = [Om, OL, h]
    return samples

