import numpy as np
from scipy.special import gamma
from scipy.integrate import quad,simps
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

# Equation 18 in Ciotti 1999
def b_n(n):
    return 2*n-1/3.+4/405./n+46./25515./n**2+131./1148175./n**3-2194697./30690717750./n**4

# Calculation of Sigma_!
def sigma_1(re,n,m):
    # m in the unit of Msun
    # re in the unit of kpc
    bn = b_n(n)
    x = bn*(1/re)**(1/n)
    def int_fun(t):
        return np.exp(-t)*t**(2*n-1)
    small_gamma = quad(int_fun,a=0,b=x)
    return m/np.pi*small_gamma[0]/gamma(2*n)

# load catalog that Nikko sent
data = np.loadtxt('sigma1.csv',delimiter=',',usecols=(5,6,7,8,9,10,))
redshift = data[:,0]
logm = data[:,1]
n = data[:,2]
n_err = data[:,3]
re = data[:,4]
re_err = data[:,5]

# convert re from arcsec to kpc
tmp = cosmo.arcsec_per_kpc_proper(redshift).value
re_kpc = re/tmp
re_kpc_err = re_err/tmp

# calculate and record Logsigma_1
sigma1 = np.zeros(len(n))
fo = open('best-fit_sigma1.dat','w')
fo.write('#col0: ID in this catalog, the same order as the file you sent to me, 39 galaxies in total\n')
fo.write('#col1: LogSigma_1 [Msun/kpc^2]\n')
for i in range(len(sigma1)):
    sigma1[i] = sigma_1(re_kpc[i],n[i],10**logm[i])
    fo.write(str(i)+' '+str(np.log10(sigma1[i]))+'\n')
fo.close()

# sanity check, logsigma1 should also smaller than logm-log(pi)
plt.plot(logm,np.log10(sigma1),'o')
plt.plot(logm,logm-np.log10(np.pi),'k-')
plt.show()
