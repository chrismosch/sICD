import time
import numpy as np
import scipy as sp
from scipy import stats
from scipy.special import gammaln, xlogy, logsumexp
from genICD import *

def f_case1(U, nm_excl_i):
    return ((2.**U)*(nm_excl_i+U))**-1


def d2f_case1(U, nm_excl_i):
    return 2.**(1-U)/(nm_excl_i+U)**3 + (2**-U)*np.log(2)**2/(nm_excl_i+U) + 2**(1-U)*np.log(2)/(nm_excl_i+U)**2


def f_case3(X, Ystar, Ydagger, nm_excl_i):
    return Ydagger / ( 2.**(X+Ystar+Ydagger) * (nm_excl_i+X+Ystar+Ydagger) )


def d2f_dYstar2_case3(X, Ystar, Ydagger, nm_excl_i):
    U = X+Ystar+Ydagger
    return Ydagger*d2f_case1(U, nm_excl_i)


def d2f_dYdagger2_case3(X, Ystar, Ydagger, nm_excl_i):
    U = X+Ystar+Ydagger
    return ( Ydagger*2.**(1-U) / (nm_excl_i+U)**3
            + (Ydagger*(2**-U)*np.log(2)**2 - np.log(2)*2**(1-U)) / (nm_excl_i+U) 
            - 2*(2**-U - Ydagger*np.log(2)*2**-U) / (nm_excl_i+U)**2 )


# vectorized over k, returns vector with p_zmi for k=0,...,K-1
def logp_zmi_us(eta_wmi, nwmi_kexcli, n_per_mk_excli, m, i, z, phi, pi, alpha, gamma):
    return np.log(nwmi_kexcli+eta_wmi) + np.log(E_g_X_Y(n_per_mk_excli, m, i, z, phi, pi, alpha, gamma))


def logp_zmi_s(eta_wmi, nwmi_kexcli, n_per_mk_excli, m, i, z, phi, pi, alpha, gamma, y_m, a, sigma_y):
    K = len(phi)
    X_m = n_per_mk_excli[m,:] + np.eye(K)  # vectorizes over k (1st row ^= zij=0 -> add 1 to count for 1st topic, 2nd row: add 1 to count for 2nd topic)
    return np.log(nwmi_kexcli+eta_wmi) + np.log(E_g_X_Y(n_per_mk_excli, m, i, z, phi, pi, alpha, gamma)) + logp_y_m(y_m, X_m, a, sigma_y)


def logp_y_m(y_m, X_m, a, sigma_y):
    return -1./(2*sigma_y**2)*(y_m-X_m.dot(a))**2


# vectorized over k, returns vector with p_zmi for k=0,...,K-1 (only case 1,2 depend on k)
def E_g_X_Y(n_per_mk_excl_i, m, i, z, phi, pi, alpha, gamma):
          
    X = phi[n_per_mk_excl_i[m,:]>0].sum()
    Jstar_m = (n_per_mk_excl_i[m,:]==0) & (n_per_mk_excl_i.sum(axis=0)>0)
    EYstar = np.sum(pi[Jstar_m]*phi[Jstar_m])
    VYstar = np.sum(pi[Jstar_m]*(1-pi[Jstar_m])*phi[Jstar_m]**2)
    EYdagger = alpha * gamma
    VYdagger = alpha*gamma*(gamma+1) - (alpha*gamma)**2/(2*alpha+1.)
    EY = EYstar + EYdagger
    VY = VYstar + VYdagger
    
    mask1 = (n_per_mk_excl_i[m,:] > 0)
    #case1 = (n_per_mk_excl_i[m,:]+phi) * (f_case1(X+EY,n_per_mk_excl_i[m,:].sum()) + 0.5*d2f_case1(X+EY,n_per_mk_excl_i[m,:].sum())*VY)  # vector
    case1 = (f_case1(X+EY,n_per_mk_excl_i[m,:].sum()) + 0.5*d2f_case1(X+EY,n_per_mk_excl_i[m,:].sum())*VY)  # vector
    mask2 = ~mask1 & (n_per_mk_excl_i.sum(axis=0) > 0)
    EY_excl_k = EY - pi*phi  # vector
    VY_excl_k = VY - pi*(1-pi)*phi**2  # vector
    #case2 = phi*pi * (f_case1(X+EY_excl_k+phi,n_per_mk_excl_i[m,:]) + 0.5*d2f_case1(X+EY_excl_k+phi,n_per_mk_excl_i[m,:])*VY_excl_k)  # vector
    case2 = (f_case1(X+EY_excl_k+phi,n_per_mk_excl_i[m,:]) + 0.5*d2f_case1(X+EY_excl_k+phi,n_per_mk_excl_i[m,:])*VY_excl_k)  # vector
    mask3 = ~mask1 & ~mask2
    case3 = (f_case3(X,EYstar,EYdagger,n_per_mk_excl_i[m,:].sum()) 
                + 0.5*d2f_dYstar2_case3(X,EYstar,EYdagger,n_per_mk_excl_i[m,:].sum())*VYstar 
                + 0.5*d2f_dYdagger2_case3(X,EYstar,EYdagger,n_per_mk_excl_i[m,:].sum())*VYdagger )
    return mask1*case1 + mask2*case2 + mask3*case3


def draw_via_PIT(logpdf,args,lb,ub,Npoints):
    # calc pdf for region of interest & normalize -> distribution for region of interest
    points = np.linspace(lb,ub,Npoints)
    logpdf_approx = logpdf(points,*args)
    logpdf_approx = logpdf_approx - logpdf_approx.max()
    logc = logsumexp(logpdf_approx)  # normalizing constant, sp.integrate.quad(p_new_pi,0,1,args=(alpha_sample,N))
    logpdf_approx_norm = logpdf_approx - logc
    cdf_approx = np.cumsum(np.exp(logpdf_approx_norm))
    cdf_approx_unique, idx = np.unique(cdf_approx, return_index=True)
    points_unique = points[idx]
    
    unif = np.random.uniform(0,1)
    sample = np.interp(unif, cdf_approx_unique, points_unique) # PIT via inverse interpolation
    
    # uniform random jitter centered at 0 & width equal to stepsize (cf BDA p76)
    stepsize = (ub-ub)/float(Npoints)
    sample = sample + np.random.uniform(-stepsize/2.,stepsize/2.)
    
    return sample #,cdf_approx_unique, points_unique


def logp_new_pi(pi, alpha, N):
    Ns = np.array(range(1,N+1))
    if isinstance(pi, np.ndarray):    
        Ns = Ns[:,np.newaxis]
    logp_pi = xlogy(alpha-1,pi) + xlogy(N,1-pi) + alpha*np.sum(1./Ns*(1-pi)**Ns,axis=0)
    return logp_pi


def logp_phi_n(n_per_mk, b, gamma, phi):
    logp = ( xlogy(gamma-1,phi) - phi - gammaln(gamma) 
            + np.sum(b * (gammaln(phi+n_per_mk) - gammaln(phi) - gammaln(n_per_mk+1) - (phi+n_per_mk)*np.log(2)), axis=0) )
    return np.log(phi>0) + logp


def p_b(b, pi, phi, nm_k):
    mask1 = nm_k>0
    mask2 = (b==0) & (nm_k==0)
    mask3 = (b==1) & (nm_k==0)
    p0 = pi
    p1 = 2**phi*(1-pi)
    return mask1*b + mask2*p1/(p0+p1) + mask3*p0/(p0+p1)
    
    
def sICD_collapsed_gibbs(iters, w, Nvocab, alpha, gamma, eta, z0, phi0, pi0, b0, supervised=False, y=None, a0=None, 
                         mu_a=0, sigma_a=1, sigma_y=1, iters_MH=100, sigma_proposal=1, Npoints=1000):
    
    z, phi, pi, b, a = z0, phi0, pi0, b0, a0
    
    M,K = b.shape
    n_per_mk = get_n_per_mk(z, K)
    n_per_ik  = get_n_per_ik(z, w, K, Nvocab)
    n = np.array([len(w_m) for w_m in w])
    
    # only keep active features
    activefeatures = (n_per_mk.sum(axis=0) != 0)
    n_per_mk = n_per_mk[:,activefeatures]
    n_per_ik = n_per_ik[:,activefeatures]
    phi = phi[activefeatures]
    pi = pi[activefeatures]
    b = b[:,activefeatures]
    if supervised:
        a = a[activefeatures]
    K_plus = len(pi)
    # adjust z since dropping inactive features leads to rearrangement (e.g. droping feature 2 -> feature 3 becomes feature 2)
    old2new_index = {old_idx:new_idx for old_idx,(status,new_idx) in enumerate(zip(activefeatures,np.cumsum(activefeatures)-1)) if status}
    z = [np.array([old2new_index[k] for k in z_m]) for z_m in z]

    pis, phis, bs, zs, Ks, a_samples, y_hats = [],[],[],[],[],[],[]
    loglikelihoods, acceptance_ratios, runtime = [],[],[]

    start_time = time.time()

    for t in range(iters):
        # update pi (semi-ordered slice sampler)
        # draw s conditonal on pi,B
        # stick_length of last active feature (pi_sorted_acrtive sorted by decreasing pi -> last element is min)
        pi_star = min(1,sorted(pi,reverse=True)[-1])
        s = np.random.uniform(0,pi_star)
        # until pi_(K^o+1)<s: generate inactive features & draw pi, let K^o be # of features generated
        pi_new = []
        pi_ball = s+1 # just sth >s
        pi_kmin1 = pi_star  # ?what is correct first value
        while pi_ball >= s: 
            # draw new pi
            pi_ball = draw_via_PIT(logpdf=logp_new_pi, args=[alpha,M], lb=0,ub=pi_kmin1, Npoints=Npoints)
            pi_new.append(pi_ball)
            pi_kmin1 = pi_ball
        pi_new = pi_new[:-1]  # exclude pi_(K^o+1) 
        pi = np.append(pi, pi_new)
        K_ball = len(pi_new)
        K = K_plus+K_ball

        # update phi
        accepted = 0
        phi_samples = []
        # for active features
        for i in range(iters_MH):
            lposterior_old = logp_phi_n(n_per_mk, b, gamma, phi)
            phi_proposal = np.random.normal(loc=phi, scale=sigma_proposal, size=K_plus)
            lposterior_proposal = logp_phi_n(n_per_mk, b, gamma, phi_proposal)
            logr = lposterior_proposal - lposterior_old
            mask = np.log(np.random.uniform(size=K_plus)) < logr
            phi[mask] = phi_proposal[mask]
            accepted += mask.sum()
            phi_samples.append(phi.copy())
        phi_posterior_mean = np.mean(phi_samples[iters_MH//2:],axis=0)
        # for inactive features (logp_phi_n simplifies to Gamma(gamma,1) prior)
        phi_new = np.random.gamma(gamma,1, size=K_ball)

        phi = np.append(phi_posterior_mean, phi_new)
        acceptance_ratios.append(accepted/float(iters_MH*K_plus))

        n_per_mk = np.concatenate((n_per_mk, np.zeros((M,K_ball))),axis=1)
        n_per_ik = np.concatenate((n_per_ik, np.zeros((Nvocab,K_ball))),axis=1)
        if supervised:
            a = np.append(a, np.random.normal(mu_a, sigma_a, size=K_ball))

        # update b
        p_bmk = p_b(np.ones((M,K_plus+K_ball)), pi, phi, n_per_mk)
        b = np.random.binomial(1,p_bmk)

        # update z_mi
        for m in range(M):
            for i in range(n[m]):
                wmi = w[m][i]
                # exclude current assignment of word from counts
                k_old = z[m][i]
                n_per_mk[m, k_old] -= 1
                n_per_ik[wmi, k_old] -= 1
                # get probability for all possible new assignments
                if supervised:
                    logp_zmi = logp_zmi_s(eta[wmi], n_per_ik[wmi,:], n_per_mk, m, i, z, phi, pi, alpha, gamma, y[m], a, sigma_y)
                else:
                    logp_zmi = logp_zmi_us(eta[wmi], n_per_ik[wmi,:], n_per_mk, m, i, z, phi, pi, alpha, gamma)
                logp_zmi = logp_zmi - logp_zmi.max()
                logp_zmi_norm = logp_zmi - logsumexp(logp_zmi)
                k_new = np.random.choice(range(K),p=np.exp(logp_zmi_norm))
                z[m][i] = k_new
                n_per_mk[m, k_new] += 1
                n_per_ik[wmi, k_new] += 1

        # only keep active features
        activefeatures = (n_per_mk.sum(axis=0) != 0)
        n_per_mk = n_per_mk[:,activefeatures]
        n_per_ik = n_per_ik[:,activefeatures]
        phi = phi[activefeatures]
        pi = pi[activefeatures]
        b = b[:,activefeatures]
        if supervised:
            a = a[activefeatures]
        K_plus = len(pi)
        K = K_plus
        # adjust z since dropping inactive features leads to rearrangement (e.g. droping feature 2 -> feature 3 becomes feature 2)
        old2new_index = {old_idx:new_idx for old_idx,(status,new_idx) in enumerate(zip(activefeatures,np.cumsum(activefeatures)-1)) if status}
        z = [np.array([old2new_index[k] for k in z_m]) for z_m in z]

        # update a
        if supervised:
            X = n_per_mk / n_per_mk.sum(axis=1)[:,np.newaxis]
            a_postvar = np.linalg.pinv(X.T.dot(X)+np.eye(K)*sigma_a**2)
            a_postmean = a_postvar.dot(X.T).dot(y)
            a = np.random.multivariate_normal(a_postmean, a_postvar)
            y_hat = X.dot(a)

        runtime.append(time.time()-start_time)
        beta_postmean_unnorm = n_per_ik+eta[:,np.newaxis]
        beta_postmean = beta_postmean_unnorm / beta_postmean_unnorm.sum(axis=0)
        loglikelihood = np.sum(stats.multinomial.logpmf(n_per_ik.T, n=n_per_ik.sum(axis=0), p=beta_postmean.T))
        loglikelihoods.append(loglikelihood)
        bs.append(b.copy())
        # ?easier way to copy the whole list including sublists (I tried: list(z), import copy, copy.copy(z), z as list of list instead of list of arrays)
        zs.append([z_m.copy() for z_m in z])
        phis.append(phi.copy())
        pis.append(pi.copy())
        Ks.append(K)
        if supervised:
            a_samples.append(a.copy())
            y_hats.append(y_hat.copy())
            
    logs = {'iteration':range(1,iters+1), 'runtime':runtime, 'loglikelihood':loglikelihoods, 'acceptance_ratio':acceptance_ratios}
    samples = {'K':Ks, 'z':zs, 'b':bs, 'phi':phis, 'pi':pis}
    if supervised:
        samples['a'] = a_samples
        samples['y_hat'] = y_hats
    
    return samples, logs