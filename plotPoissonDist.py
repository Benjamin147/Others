import matplotlib.pyplot as plt
import numpy as np
import h5py as h

from scipy.stats import poisson
from scipy.stats import norm
P = poisson.pmf # exp(-mu)*mu**k/k!, P(k,mu)
G = norm.pdf # exp(-(x-loc)**2/2/std**2)/sqrt(2*pi), G(x, loc, std) 


################################################# define some functions
def inter(x):
    '''
    Interpolates between two values
    '''
    return (x[1:]+x[:-1])/2

def peak(x, k, mu, std, std_ele=0.18/2.355):
    '''
    Returns one peak of the distribution model
    x: np array of charge values
    k: returns the k-th photoelectron peak
    mu: poisson mean of the distribution
    std: standard deviation of the 1pe peak
    std_ele: standard deviation of the pedestrial, default is imperical
    '''
    if k == 0:
        return P(k,mu)*G(x, 0, std_ele)  # pedestrial
    else:
        return P(k,mu)*G(x,k*spe,np.sqrt(k)*std)

def poissonModel(x, n=20, mu, std, std_ele=0.18/2.355):
    '''
    Calculates the sum of all peaks 
    x: np array of charge values
    n: number of considered pe peaks
    '''
    peaks = []
    dx=np.mean(x[1:]-x[:-1])
    y=peak(x,0, mu, std, std_ele)
    peaks.append(y)
    for i in range(1,n):
        yi = peak(x,i, mu, std, std_ele)
        peaks.append(yi)
        y += yi 
    return x, dx, np.array(peaks), y, 1-np.trapz(y,x)

def comparison(x, charge, mu, std, std_ele=0.18/2.355):
    '''
    Calculates the chi2 red between data and model
    '''
    # calculate the model
    x, dx, peaks, y, err = poissonModel(num_bins, n=10)
    # histogram the data
    y_data, bin_edges = np.histogram(charge, bins=x, normed=True)
    # calculate the error and the chi square
    y_err = np.sqrt(y_data/1e5)
    y_inter = (y[1:]+y[:-1])/2
    mask = y_err > 0
    freedom = np.sum(mask) - c
    chi2_red = np.sum(((y_data-y_inter)[mask]/y_err[mask])**2)/(freedom)**2
    return x, y, peaks, err, freedom, y_data, y_err, chi2_red
    
################################################# take data 
f = h.File('test.ADQData.h5')
charge=f['STAT/CHARGE/C'][:]
f.close()
################################################# Define model
mu = 3.25
std = 1.33
spe = 3.15 
std_ele = 0.18/2.355  
num_bins = 112
c = 3
   
# show chi2 in dependency of binning
if 0:
    chi2red = []
    for num_bins in np.arange(50,500,1):
        x, y, peaks, err, freedom,y_data,y_err, chi2_red = comparison(charge, num_bins)
        chi2red.append(chi2_red)
        print(num_bins)
    plt.plot(binning,chi2red)
    plt.ion()
    plt.show()
    
    
if 1:
    x, y, peaks, err, freedom,y_data,y_err, chi2_red = comparison(charge, num_bins)
    for p in peaks:
        plt.plot(x,p, ls=":")
    plt.plot(x,y)
    plt.errorbar(inter(x),y_data,yerr=y_err, fmt='.', markersize=1, c='r')
    plt.text(20,0.1,'chi2 red: %.2f\nmu: %.2f\nspe: %.2f pC\nwidth_spe: %.2f pC'%(chi2_red, mu, spe, std))
    plt.axvline(spe, c='g', label='Spe')
    plt.xlabel('Charge in pC')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()



