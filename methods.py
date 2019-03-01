import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
from scipy import fftpack
import time

# AU to m
AU_to_M = 1.496*10**11
# AU/y to m/s
AUpY_to_MpS = 4743.183
# solar mass to kg
SM_to_kg = 1.988435*10**30
# year to seconds
Year_to_s = 3.154*10**7
# gravitational constant G[m^3/kg*s^2]
G = 6.67428*10**-11
# gravitational parameter mu
mu = 1.327137415503024*10**20

# number of bodies
n_b = 9
batchsize = 11

name_label = ['Sun', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
oe_label = ['Semimajor Axis [AU]', 'Eccentricity', 'Inclination [Deg]', 'A. of periapsis [Rad]', 'L. of ascending node [Rad]']
sin_labels = ['Period [Years]', 'Amplitude', 'Phase', 'Offset']
planetperiods = np.array([0.6152, 1, 1.8808, 11.862, 29.447, 84.012, 164.789, 249.589]) # Ur = 84, Sa = 29.4475
ratios = [3/4, 1/4, 2/3, 1/3, 1/2, 1, 2/1, 3/1, 3/2, 4/3, 4/1]
cols = ['k', 'r', 'g', 'b', 'c']



def CartesianToKepler(mp, r, v):
    # Input (two-body parameters)
    # mp = planet mass [kg]
    # r = planet position vector [m]
    # v = planet velocity vector [m/s]
    
    # output:
    #0: semimajor axis a [m]
    #1: vector of eccentricity e
    #2: eccentricity e
    #3: inclination i [degree]
    #4: argument of periapsis omega [radians]
    #5: longitude of ascending node Omega [radians]
    #6: mean anomaly M [radians]
    #7: eccentric anomaly E [radians]

    # orbital momentum vector h
    h = np.cross(r, v)
    #print ('r, v, h:', r, v, h)

    # eccentricity vector vec_e
    vec_e = (np.cross(v, h)/mu - r/np.linalg.norm(r))
    # eccentricity e
    e = np.linalg.norm(vec_e)
    #print (e)
    #print ('eccentricity vector vec_e:', vec_e)
    #print ('eccentricity e:', e)

    # vector n to ascending node
    vec_n = np.array((-h[1], h[0], 0))
    n = np.linalg.norm(vec_n)

    # true anomaly nu
    if (np.dot(r, v) >= 0):
        nu = np.arccos(np.dot(vec_e, r)/(e*np.linalg.norm(r)))
    else:
        nu = 2*np.pi - np.arccos(np.dot(vec_e, r)/(e*np.linalg.norm(r)))
    #print ('true anomaly nu [rad]:', nu)

    # inclination i
    i = np.degrees( np.arccos(h[2]/np.linalg.norm(h)) )
    #print ('inclination i:', i)

    # eccentric anomaly E
    E = 2*np.arctan( np.tan(nu/2.)/np.sqrt((1+e)/(1-e)) )
    #print ('eccentric anomaly E:', E)

    # longitude of ascending node Omega
    if vec_n[1] >= 0.:
        Omega = np.arccos(vec_n[0]/n)
    else:
        Omega = 2*np.pi - np.arccos(vec_n[0]/n)
    #print ('longitude of ascending node Omega:', Omega)

    # argument of periapsis omega
    if vec_e[0] >= 0.:
        omega = np.arccos(np.dot(vec_n, vec_e)/(n*e))
    else:
        omega = 2*np.pi - np.arccos(np.dot(vec_n, vec_e)/(n*e))
    #print ('argument of periapsis omega:', omega)

    # mean anomaly M
    M = E - e*np.sin(E)
    #print ('mean anomaly M:', M)

    # semi major axis a
    a = 1/(2/np.linalg.norm(r) - np.linalg.norm(v)**2/mu)
    #print ('semi major axis a:', a)
    return (a, vec_e, e, i, omega, Omega, M, E)



def SynPeriod(I1, I2):
    P1 = planetperiods[I1-1]
    P2 = planetperiods[I2-1]
    if P1 == P2:
        PSyn = P1
    else:
        FSyn = abs(1/P1-1/P2)
        PSyn = 1/(FSyn)
    return PSyn



def Goodness(Y, Fit):
    SS_Res = np.sum((Y-Fit)**2)
    SS_Tot = np.sum((Y-np.mean(Y))**2)
    R2 = 1 - SS_Res/SS_Tot
    return R2

def Adj_I(I_in, N_in):
    if I_in < N_in:
        I_0 = 0
    else:
        I_0 = I_in - N_in
    I_1 = I_in + N_in
    return I_0, I_1

def Names(Body, Coordinate):
    B = str(name_label[Body])
    C = str(oe_label[Coordinate])
    print (B, C)
    return

def DetrendLinear(NewData, Tau, N):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[:,0]
    Y = NewData[:,1]
    Init_vals = FirstGuessLinear(NewData, Tau, N)
    Res, Cov = curve_fit(xLin, X, Y, p0=Init_vals) #, bounds=bounds)
    New_Y = Y - Res[1]*X
    return New_Y, Res

def FirstGuessLinear(NewData, Tau, N):
#     Tau: time in question in yr, NewData: new simulation data, N: number of snapshots +-tau for getting First Guess
    N_offset = 5
#     Calc i nearest to Tau and I_0:
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(Tau, N)
#     Calc slope from I_tau-N to I_tau+N
    if (I_tau < N):
        Slope = (NewData[I_tau+N, 1] - NewData[I_tau, 1])/((N+I_tau)*0.5)
    else:
        Slope = (NewData[I_tau+N, 1] - NewData[I_tau-N, 1])/(2*N*0.5)
#     Calc mean of 2*N_offset data points:
    if (I_tau < N_offset):
        Mean_y = np.mean(NewData[0:I_tau+N_offset, 1])
    else:
        Mean_y = np.mean(NewData[I_tau-N_offset:I_tau+N_offset, 1])
    Offset = Mean_y - Slope*Tau
    return Offset, Slope

def Resonances(Ax, B, Res, Ymin, C):
    Lw = .7
    Period = planetperiods[B-1]
    for R in Res:
        F = R/Period
        Ax.axvline(F, lw=Lw, c=C, label=str(round(R,2)) + '-line of ' + str(name_label[B]) )
        Ax.text(F, Ymin, str(round(R,2)), fontsize=16)
        print (str(round(R,2)) + '-line of ' + str(name_label[B]) + ' at year ' + str(round(1/F,1)))

def PlotResonances(Ax, B, Res, Ymin, C):
    Lw = .7
    Period = planetperiods[B-1]
    for R in Res:
        F = R/Period
        Ax.axvline(F, lw=Lw, c=C, label=str(round(R,2)) + '-line of ' + str(name_label[B]) )
        Ax.text(F, Ymin, str(round(R,2)), fontsize=16)
        print (str(round(R,2)) + '-line of ' + str(name_label[B]) + ' at year ' + str(round(1/F,1)))

def PlotTestRatios(Ax, B, Ratios, Ymin, C):
    Lw = 0.7
    Period = planetperiods[B-1]
    R = Ratios[0]
    Freq = R/Period
    Ax.axvline(x=Freq, lw=Lw, c=C, label="Possible resonance freq's with " + name_label[B])
    Ax.text(Freq, Ymin, str(round(R,2)), fontsize=16)
    for R in Ratios[1:]:
        Freq = R/Period
        Ax.axvline(x=Freq, lw=Lw, c=C)
        Ax.text(Freq, Ymin, str(round(R,2)), fontsize=16)
    return

def Linear(X, Args):
    A, B = Args
    F = A + B*X
    return F

def xLin(X, A, B):
    F = A + B*X
    return F

def Sin(X, Args):
    Period, Amplitude, Phase, Offset = Args
    F = Offset + Amplitude*np.sin(Phase + 1/Period*X*(2*np.pi))
    return F

def xSine(X, Period, Amplitude, Phase, Offset):
    F = Offset + Amplitude*np.sin(Phase + (2*np.pi/Period)*X)
    return F

def FitSin(NewData, Tau, N, Init_vals, Maxfev):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[I_0:I_1, 0]
    Y = NewData[I_0:I_1, 1]
    Res, Cov = curve_fit(xSine, X, Y, p0=Init_vals, maxfev=Maxfev) #, bounds=bounds)
    return Res

def AddedSines(X, Args):
    Per_0, Amp_0, Phi_0, Offset, Per_1, Amp_1, Phi_1 = Args
    F1 = Amp_0*np.sin(Phi_0 + 1/Per_0*X*(2*np.pi))
    F2 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    return Offset + F1 + F2

def xTwoAddedSines(X, Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2):
    Sine1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    Sine2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    return Offset + Sine1 + Sine2

def FitTwoAddedSines(NewData, Tau, N, Init_vals, Maxfev):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[I_0:I_1, 0]
    Y = NewData[I_0:I_1, 1]
    Res, Cov = curve_fit(xTwoAddedSines, X, Y, p0=Init_vals, maxfev=Maxfev) #, bounds=bounds)
    return Res

def ThreeAddedSines(X, Args):
    Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3 = Args
    F1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    F2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    F3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    return Offset + F1 + F2 + F3

def xThreeAddedSines(X, Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3):
    Sine1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    Sine2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    Sine3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    return Offset + Sine1 + Sine2 + Sine3

def FitThreeAddedSines(NewData, Tau, N, Init_vals, Maxfev):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[I_0:I_1, 0]
    Y = NewData[I_0:I_1, 1]
    Res, Cov = curve_fit(xThreeAddedSines, X, Y, p0=Init_vals, maxfev=Maxfev) #, bounds=bounds)
    return Res

def FourAddedSines(X, Args):
    Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3, Per_4, Amp_4, Phi_4 = Args
    F1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    F2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    F3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    F4 = Amp_4*np.sin(Phi_4 + 1/Per_4*X*(2*np.pi))
    return Offset + F1 + F2 + F3 + F4

def xFourAddedSines(X, Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3, Per_4, Amp_4, Phi_4):
    Sine1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    Sine2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    Sine3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    Sine4 = Amp_4*np.sin(Phi_4 + 1/Per_4*X*(2*np.pi))
    return Offset + Sine1 + Sine2 + Sine3 + Sine4

def FitFourAddedSines(NewData, Tau, N, Init_vals, Maxfev):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[I_0:I_1, 0]
    Y = NewData[I_0:I_1, 1]
    Res, Cov = curve_fit(xFourAddedSines, X, Y, p0=Init_vals, maxfev=Maxfev) #, bounds=bounds)
    return Res

def FiveAddedSines(X, Args):
    Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3, Per_4, Amp_4, Phi_4, Per_5, Amp_5, Phi_5 = Args
    F1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    F2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    F3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    F4 = Amp_4*np.sin(Phi_4 + 1/Per_4*X*(2*np.pi))
    F5 = Amp_5*np.sin(Phi_5 + 1/Per_5*X*(2*np.pi))
    return Offset + F1 + F2 + F3 + F4

def xFiveAddedSines(X, Per_1, Amp_1, Phi_1, Offset, Per_2, Amp_2, Phi_2, Per_3, Amp_3, Phi_3, Per_4, Amp_4, Phi_4, Per_5, Amp_5, Phi_5):
    Sine1 = Amp_1*np.sin(Phi_1 + 1/Per_1*X*(2*np.pi))
    Sine2 = Amp_2*np.sin(Phi_2 + 1/Per_2*X*(2*np.pi))
    Sine3 = Amp_3*np.sin(Phi_3 + 1/Per_3*X*(2*np.pi))
    Sine4 = Amp_4*np.sin(Phi_4 + 1/Per_4*X*(2*np.pi))
    Sine5 = Amp_5*np.sin(Phi_5 + 1/Per_5*X*(2*np.pi))
    return Offset + Sine1 + Sine2 + Sine3 + Sine4 + Sine5

def FitFiveAddedSines(NewData, Tau, N, Init_vals, Maxfev):
    I_tau = int(2*Tau)
    I_0, I_1 = Adj_I(I_tau, N)
    X = NewData[I_0:I_1, 0]
    Y = NewData[I_0:I_1, 1]
    Res, Cov = curve_fit(xFiveAddedSines, X, Y, p0=Init_vals, maxfev=Maxfev) #, bounds=bounds)
    return Res
