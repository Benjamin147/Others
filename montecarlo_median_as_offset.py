import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def generateWVF(offset_mean, offset_width, noise_width, nwvf, sample):
    offset = np.random.normal(offset_mean, offset_width, nwvf)
    wvf = np.random.normal(0, noise_width, (nwvf, sample))
    wvf =(wvf.T + offset).T 
    return wvf, offset

def compMEDIAN(wvf, offset):
    median = np.median(wvf, axis=1)
    err = offset - median
    return err
    
def compMeanCUT(wvf, offset, cuts):
    wvf = np.sort(wvf, axis=1)
    mean = np.mean(wvf[:,cuts[0]:cuts[1]], axis=1)
    err = offset - mean
    return err


#sample = 100000
nwvf = 10000
offset_width = 2/5
offset_mean = 6
noise_width = 0.3/5

samples = []
std_median = []
std_meancut = []

for i in range(4):
    sample = 10**(1+i)
    wvf, offset = generateWVF(offset_mean, offset_width, noise_width, nwvf, sample)
    err_median = compMEDIAN(wvf, offset)
    cuts = (int(2*sample/10), -int(2*sample/10))
    err_meancut = compMeanCUT(wvf, offset, cuts)
    
    samples.append(sample)
    std_median.append(np.std(err_median))
    std_meancut.append(np.std(err_meancut))
    print(sample)
    
plt.scatter(samples, std_median, label='median')
plt.scatter(samples, std_meancut, label='meancut')
plt.yscale('log')
plt.legend()
    
    

