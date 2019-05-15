import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.ion()

tol = 1e-7

def polygonUmfang(radius, n):
    '''
    Calculates the circumference of an n-polygon in a circel of radius radius. 
    '''
    return 2*radius*n*np.sin(np.pi/n)

def norm(p):
    '''
    gives back the length of an array
    '''
    return np.sum(p**2, axis=0)**0.5       
      
def sproduct(a,b):
    '''
    Returns the skalar product of a and b
    '''
    return np.sum(a*b, axis=0)  
    
def angle(a,b):
    if not isinstance(a, np.ndarray): a = np.array(a)
    if not isinstance(b, np.ndarray): b = np.array(b)
    return np.rad2deg(np.arccos(sproduct(a,b)/norm(a)/norm(b)))
      
def getPolar(p):
    p = np.array(p).T
    r = norm(p)      
    mask = p[1] < 0
    phi = np.rad2deg(np.arccos(p[0]/norm(p)))
    if p.ndim > 1:
        phi[mask] = 360 - phi[mask] 
    elif mask:
        phi = 360 - phi
    return np.array([r, phi]).T

def getCart(p):
    p = np.array(p).T
    x = p[0]*np.cos(np.deg2rad(p[1]))
    y = p[0]*np.sin(np.deg2rad(p[1]))
    return np.array([x,y]).T
    
def intersection(t, *p):
    l = p[0] # line object
    r = p[1] # outer radius
    return (sum(l.cart(t)**2) - r**2)**2

def calculateIntersection(l,radius, guess):
    g = np.array([guess])
    args = (l, radius)
    res = minimize(intersection, g, args, bounds=None)
    t_max = res.x[0]
    end = l.cart(t_max)
    length = norm(l.start-end)
    hit = (intersection(t_max, l,radius) < tol) & (length > tol)
    return end, l.start, length, hit


class line():
    def __init__(self, p0, angle):
        self.start = np.array(p0)
        self.angle = np.deg2rad(angle%360)
        self.n = np.array([np.cos(self.angle),np.sin(self.angle)])
        self.intersection_calculated = False
        
    def cart(self, t):
        return self.start + t*self.n
    
    def getIntersection(self,radius_outer, radius_inner=None):
        if not self.intersection_calculated:
            end1, start1, l1, hit1 = calculateIntersection(self,radius_outer, 2*radius_outer)
            if radius_inner is not None:
                end2, start2, l2, hit2 = calculateIntersection(self,radius_inner, 0)
                self.end = end1 if not hit2 else end2
                self.length = l1 if not hit2 else l2
            else:
                self.end = end1 
                self.length = l1
            self.intersection_calculated = True
        

    def show(self, n=None):
        a = np.array([self.start,self.end]).T
        plt.plot(*a)
        if n is not None:
            ctr = np.mean([self.start,self.end], axis=0)
            plt.text(ctr[0], ctr[1], '%d'%n)
        plt.scatter(*a)

class circle():
    def __init__(self, phi0, radius_outer, wall, penetration=0, plot=False):
        self.lines = []
        self.radius_outer = radius_outer
        if wall is not None:
            self.radius_inner = radius_outer - wall
        else:
            self.radius_inner = None
        self.plot = plot
        p0 = (self.getCart(0)[0] - penetration, self.getCart(0)[1])
        self.lines.append(line(p0, 180-phi0))
        self.lines[-1].h = 1 if self.lines[-1].n.ravel()[1] >= 0 else -1
        self.lines[-1].getIntersection(self.radius_outer, self.radius_inner)
        if self.plot: self.showCircle()        
        if self.plot: self.lines[-1].show(0)
        self.len = self.lines[-1].length
        self.ref = 0
        
        
    def showCircle(self, n=200):
        f = plt.figure()
        self.fignum = f.number
        theta = np.linspace(0,360,n)
        r = np.full(n,self.radius_outer)
        p = np.array([r, theta]).T
        p = getCart(p)
        plt.plot(*p.T, c='r')
        if self.radius_inner is not None:
            r = np.full(n,self.radius_inner)
            p = np.array([r, theta]).T
            p = getCart(p)
            plt.plot(*p.T, c='r')
        
    def getCart(self, angle):
        return getCart((self.radius_outer, angle))
        
    def getN(self, angle):
        return getCart((1, angle))
         
    def appendLine(self, p, angle):   
        self.lines.append(line(p, angle))
       
    def getNewPhi(self,l):
        n_line = l.n.ravel()
        l.getIntersection(self.radius_outer, self.radius_inner)
        phi_intersection = getPolar(l.end)[1]
        n_circle = -1*np.array(self.getN(phi_intersection))
        new_n_line = n_line - 2*sproduct(n_circle, n_line)*n_circle
        reflection_angle = angle(n_circle, n_line)%90
        return getPolar(new_n_line)[1], reflection_angle
        
    def getNewLine(self):
        l = self.lines[-1]
        if norm(l.start) - self.radius_inner**2 < tol: # to find out whether it is reflected on the inner radius 
            l.getIntersection(self.radius_outer, None)
        else:
            l.getIntersection(self.radius_outer, self.radius_inner)
        phi_new, reflection_angle = self.getNewPhi(l)
        p_new = l.end
        self.appendLine(p_new, phi_new)
        self.lines[-1].h = l.h
        self.lines[-1].getIntersection(self.radius_outer, self.radius_inner)
        self.len += self.lines[-1].length
        self.ref += 1
        if self.plot: 
            plt.figure(self.fignum)
            self.lines[-1].show(self.ref) 
        return reflection_angle
        
    def getline(self, n=-1):
        return self.lines[n]
    
    def frac(self):
        return self.len/self.ref
    
    def fracLimes(self, precision_=0.005, Nmax=100):
        self.getNewLine()
        for n in range(Nmax): 
            f0 = self.frac()
            self.getNewLine()
            f1 = self.frac()
            precision = abs(f0/f1-1)
            if n%5 == 0: print('precision: ',precision, end=' \r') 
            if precision_ > precision: break
        print('')
        return f1

class photon():
    def __init__(self):
        self.theta
        self.phi
        self.z
        self.penetration
        self.n_reflection
        self.l
        

class tube():
    def __init__(self, radius=22.5, wall=6, length = 300):
        self.radius = radius
        self.length = length
        self.wall = wall
        self.radius = radius
    
    def getPropagationLength(self, z, theta): # 90 - angle between light and radial axis
        l = z/np.sin(np.deg2rad(theta))
        l_XYprojection = z/np.tan(np.deg2rad(theta))
        return l, l_XYprojection
     
         
    def shot(self, z, phi, theta, penetration=0, iteration=10000): 
        '''
        theta: lot angle on surface aligned with the radial axis
        phi: lot angle  on surface perpendicular with the radial axis
        '''
        self.l, self.l_XYprojection = self.getPropagationLength(z, theta)
        self.c = circle(phi,self.radius, wall=self.wall, penetration=penetration, plot=1)
        
        for i in range(iteration):
            if i%5 == 0:print(i, end='\r')
            self.l_before = self.c.len
            if self.c.len > self.l_XYprojection: break
            reflection_angle = self.c.getNewLine()
        
        if self.c.len < self.l_XYprojection:
            print('Iteration depth reached')     
            return None
        
        self.dl = self.l - self.l_before 
        p = self.c.lines[-1].cart(self.dl)        # position at the end 
        n = self.c.ref                       # number of reflections
        return p, n
        
        
# plots der fraction propagtion length / N scattering in dependence on the angle
if 0:   
    a=[]
    for angle in range(0,90,1):
        c = circle(angle, 45/2, wall=3, penetration=0, plot=False)
        for n in range(30): 
            c.getNewLine()
        if angle%5==0: print(angle, end=' \r');
        a.append((angle, c.frac()))    
    a = np.array(a).T
    plt.plot(*a)

    a=[]
    for angle in range(0,90,1):
        c = circle(angle, 45/2, wall=6, penetration=0, plot=False)
        for n in range(30): 
            c.getNewLine()
        if angle%5==0: print(angle, end=' \r');
        a.append((angle, c.frac()))    
    a = np.array(a).T
    plt.plot(*a)
else:
    c = circle(50, 45/2, wall=3, penetration=0, plot=False)
    t = tube()
    


