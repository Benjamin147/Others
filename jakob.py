import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
plt.ion()


f = open('pmtlog_file.txt','r')
lines = f.readlines()
n = len(lines)
data = []
for i,line in enumerate(lines):
    data.append(np.fromstring(line, dtype=float, sep=' '))
    print(i, 'of', n, end='\r')
data = np.array(data)

data = data.T

data[-1] = data[-1] - data[-1][0]
time = data[-1]/60**2
counts = data[0]

plt.figure()
#plt.scatter(time, counts, s=1)
plt.yscale('log')
#for i in range(2,6):
#    plt.figure()
#    plt.ylim(data[i].min(), data[i].max())
#    plt.scatter(time, data[i], s=1)


m = counts < 2850

mt = time > 38
mt &= time < 51
mc = counts > 2400
m &= ~(mc & mt)
mt = time > 51
mt &= time < 53
mc = counts > 366
m &= ~(mc & mt)

dc = counts[m]
tdc = time[m]
plt.scatter(tdc, dc, s=1, color='b', label='dark counts')

c = counts[~m]
tc = time[~m]
#plt.scatter(tc, c, s=1, color='r', label='counts')

m1 = c > 10**6
c1 = c[m1]
t1 = tc[m1]
plt.scatter(t1, c1, s=1, color='c', label='OD1')
m2 = ~m1 & (c > 10**5)
c2 = c[m2]
t2 = tc[m2]
plt.scatter(t2, c2, s=1, color='y', label='OD2')
m3 = ~m1 & ~m2 & (c > 10**4)
c3 = c[m3]
t3 = tc[m3]
plt.scatter(t3, c3, s=1, color='b', label='OD3')
m4 = ~m1 & ~m2 & ~m3 & (c > 10**3)
c4 = c[m4]
t4 = tc[m4]
plt.scatter(t4, c4, s=1, color='r', label='OD4')
m5 = ~m1 & ~m2 & ~m3 & ~m4 & (c > 10**2)
c5 = c[m5]
t5 = tc[m5]
plt.scatter(t5, c5, s=1, color='g', label='OD5')

plt.legend()

#plt.figure()
#plt.title('OD1')
#plt.hist(c1, bins='auto')
#plt.figure()
#plt.title('OD2')
#plt.hist(c2, bins='auto')
#plt.figure()
#plt.title('OD3')
#plt.hist(c3, bins='auto')
#plt.figure()
#plt.title('OD4')
#plt.hist(c4, bins='auto')
#plt.figure()
#plt.title('OD5')
#plt.hist(c5, bins='auto')

std=np.array((np.std(c1),np.std(c2),np.std(c3),np.std(c4),np.std(c5)))
mean=np.array((np.mean(c1),np.mean(c2),np.mean(c3),np.mean(c4),np.mean(c5)))
plt.figure()
plt.scatter(-1, np.std(dc)/len(dc)**0.5/np.mean(dc), label='dc')
plt.scatter(range(1,6), std/mean, label='std')
plt.scatter(range(1,6), 1/mean**0.5, label='stat')
plt.xlabel('OD')
plt.ylabel('Relative error')
plt.legend()






