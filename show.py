import numpy as np
import matplotlib.pyplot as plt
import h5py as h
from time import time

plt.ion()
f = h.File('test.ADQData.h5')
n = 100000
b = np.linspace(-1,300,400)
b_ = (b[:-1] + b[1:])/2
x = range(200)

if 0:
    off=f['STAT/OFFSET/C'][:]
    plt.title('Offset from minimum offset')
    plt.hist(off, bins=np.linspace(4,6,1000))
    plt.xlabel('ADQ units')
    plt.ylabel('#')
    del offset

wvf=f['WVF/C/RUN_0'][:]
f.close()    

if 0:
    offset = np.zeros(n)
    t0 = time()
    for i, w in enumerate(wvf):
        num,btmp = np.histogram(w, bins=b)
        offset[i] = b_[np.argmax(num)]
        if i%500==0: print(i, end='\r')
        if offset[i] > 3: plt.plot(b_,num)
    out=h.File('offset.h5', 'w')
    out.create_dataset('offset', data=offset)
   
if 0:
    out=h.File('offset.h5', 'r')
    offset=out['offset'][:]

#plt.hist(offset, bins=np.linspace(-1,1,100) )

if 0:
    for i,w in enumerate(wvf):
        plt.plot(x,w,alpha=0.1)
        if i%50==0: print(i)
        if i==10000: break
        
# get electronic noise        
if 1:
    pretrigger = wvf[:20]
    std = np.std(pretrigger, axis=0)
    #plt.hist(std, bins=np.linspace(0,1,100))
    print(np.mean(std[std<0.8]))



