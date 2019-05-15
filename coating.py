#!/usr/local/anaconda3/bin/python -i

import numpy as np
import pyperclip

d_stange = 4.5
d_stange_err = 0.5
d_paint = 90
d_paint_err = 0.1

A_paint = np.pi*d_paint**2/4
A_paint_err = np.pi*2*d_paint*d_paint_err/4

roh_paint = 0.88/1000 #[g/mm^3]


A_dipper = 3*d_stange*np.pi*300 + 54**2/4*np.pi + 54*np.pi*18 + 3*(10*8*2 + 11*8 + 11*12*2)
A_dipper_err = A_dipper*0.05


def getPaintOnQuartz(d_glas, d_glas_err, diff):
    '''
    d_glas: durchmesser glas in mm
    d_glas_err
    diff: differenz paint höhe in mm
    return: masse paint auf glas in g, err
    '''

    diff_err = np.sqrt(2)*0.5
    A_glas = np.pi*d_glas*300
    A_glas_err = np.pi*d_glas_err*300
    
    dV = diff*A_paint
    dV_err = ((diff_err*A_paint)**2 + (diff*A_paint_err)**2)**0.5
    nenner = A_dipper/A_glas + 1
    dV_glas = dV/nenner
    term1 = dV_err/nenner
    term2 = dV/nenner**2*A_dipper_err/A_paint
    term3 = dV/nenner**2*A_dipper/A_paint**2*A_paint_err
    dV_glas_err = (term1**2 + term2**2 + term3**2)**0.5
    
    return dV_glas*roh_paint, dV_glas_err*roh_paint
    
def getG(d_glas, d_glas_err, w_glas=None, w_glas_err=0):
    '''
    d_glas: durchmesser glas in mm
    d_glas_err
    return: gemetric factor
    '''
    global g, g_err
    
    d_glas_in = 0 if w_glas is None else d_glas - w_glas*2
    d_glas_in_err = 0 if w_glas is None else (d_glas_err**2 + (2*w_glas_err)**2)**0.5
    
    diff2 = d_paint**2 - d_glas**2 - 3*d_stange**2 + d_glas_in**2 
    g = d_paint**2 / diff2 
    term1 = 2*d_paint*d_paint_err/diff2 + 2*d_paint**3*d_paint_err/diff2**2
    term2 = 2*d_paint**2*d_glas*d_glas_err/diff2**2
    term3 = 2*d_paint**2*d_stange*d_stange_err/diff2**2
    term4 = 2*d_paint**2*d_glas_in*d_glas_in_err/diff2**2
    
    g_err = (term1**2 + term2**2 + (3*term3)**2 + term4**2)**0.5
    
    return g, g_err 

def getG_rod(d_glas, d_glas_err, n):
    '''
    d_glas: durchmesser glas in mm
    d_glas_err
    n: number of rods per coating
    return: gemetric factor
    '''
    global g, g_err
    
    
    diff2 = d_paint**2 - n*d_glas**2 
    g = d_paint**2 / diff2 
    term1 = 2*d_paint*d_paint_err/diff2 + 2*d_paint**3*d_paint_err/diff2**2
    term2 = 2*d_paint**2*n*d_glas*d_glas_err/diff2**2
    
    g_err = (term1**2 + term2**2)**0.5
    
    return g, g_err 
    
def getVreal(v_eff, g, g_err):
    v_real = v_eff/g
    v_real_err = v_eff/g**2*g_err
    return v_real, v_real_err
    
def getVeff(v_real,  g, g_err):
    v_eff = v_real*g
    v_eff_err = v_real*g_err
    return v_eff, v_eff_err
    
def addsquare(*p):
    p = np.array(p)
    return (np.sum(p**2))**0.5

def lowerPosition(l = 300, higher_pos = 0, free=30, thickness_stamp=0):
    '''
    l: length of the tube
    higher_pos: position at which the lower side of the stamp touches the paint
    free: length the uncoated range
    thickness_stamp: thickness of the stamp or distance between tube and paint, when tube at higher position
    '''
    if 'g' not in globals():
        print('you have to specify g with getG()')
    else:
        lower_postion = higher_pos + (thickness_stamp + l - free)/g
        print('l: %d mm\nhigher_position: %.1f mm\nfree: %d mm\nthickness_stamp: %.1f\nlower_postion: %.1f mm'%(l, higher_pos, free, thickness_stamp, lower_postion))

def stat(*l):
    global wheight_list
    wheight_list = np.array(l)
    m = np.mean(l)
    std = np.std(l) if np.std(l) > 0.005 else 0.005
    s = '$%.3f\pm%.3f$'%(m,std)
    print('"'+s+'"'+' saved in clipboard.')
    pyperclip.copy(s)
    return m, std 
