import numpy as np
import matplotlib as mpl
#from importetFunctions import *
import time
import pickle as pl
import matlab.engine
from functions import *
#from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
#import tikzplotlib
plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#mpl.rc('text.latex', preamble=r"\boldmath")

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
fraction = 1.5
dpi = 300
PgWidthPt = 245
defBack = mpl.get_backend()

""" for plotting histogram and averaging over lambda """
n_bins = 20

""" for MwG"""
burnIn = 50

betaG = 1e-4# 1e-18#
betaD = 1e3#9e3#1e-3#1e-10#1e-22#  # 1e-4

""" for B_inve"""
tol = 1e-8

df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
#O3[42:51] = np.mean(O3[-4::])
minInd = 2
maxInd = 45#54
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3

def height_to_pressure(p0, x, dx):
    R = constants.gas_constant
    R_Earth = 6371  # earth radiusin km
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp(x)
    return p0 * np.exp(-28.97 * grav / temp / R * dx  )
##
calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_press[1:] = press.reshape((len(press),1)) #hPa
actual_heights = np.zeros(len(press)+1)
try_heights = np.logspace(0,2.2,1000)
try_heights[0] = 0

for i in range(1,len(calc_press)):
    #k = 0
    for j in range(0, len(try_heights)-1):
        curr_press = height_to_pressure(calc_press[i-1], actual_heights[i-1], try_heights[j] - actual_heights[i-1])
        next_press = height_to_pressure(calc_press[i-1], actual_heights[i-1], try_heights[j+1] - actual_heights[i-1])
        #print(curr_press)
        if abs(calc_press[i]-curr_press) < abs(calc_press[i]-next_press):
            next_press = height_to_pressure(calc_press[i - 1], actual_heights[i - 1],
                                            try_heights[j - 1] - actual_heights[i - 1])

            if abs(calc_press[i]-curr_press) > abs(calc_press[i]-next_press):
                actual_heights[i] = try_heights[j-1]
                k = j-1
            else:
                actual_heights[i] = try_heights[j]
                k = j
            break

print('got heights')



'''fit pressure'''
#efit, dfit, cfit,
cfit, bfit, afit = np.polyfit(actual_heights, np.log(calc_press), 2)


def pressFunc(a,b,c,d,e,x):
    #a[0] = pressure_values[0]*1.75e1
    return np.exp( e * x**4 + d * x**3 + c * x**2 + b * x + a)


# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(calc_press,actual_heights)
# plt.plot(pressFunc(afit, bfit, cfit, 0, 0,actual_heights),actual_heights)
# plt.show()
heights = actual_heights[1:]
##
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
#heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

SpecNumLayers = len(VMR_O3)
height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
np.savetxt('height_values.txt',height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt',VMR_O3, fmt = '%.15f',delimiter= '\t')
np.savetxt('pressure_values.txt',pressure_values, fmt = '%.15f', delimiter= '\t')



temp_values = get_temp_values(height_values)
""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R_Earth = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 45
SpecNumLayers = len(height_values)

n = SpecNumLayers
m = SpecNumMeas

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))

#find best configuration of layers and num_meas
#so that cond(A) is not inf
# coeff = 1/np.log(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * coeff * np.log( np.linspace(1, int(SpecNumMeas) , SpecNumMeas ))

# coeff = 1/(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * np.exp(- coeff *4* np.linspace(0, int(SpecNumMeas) -1 , SpecNumMeas ))
# meas_ang = np.flip(meas_ang)

meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], 0.0009))
#meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], 0.00045))
SpecNumMeas = len(meas_ang)
m = SpecNumMeas

A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R_Earth)
np.savetxt('tang_heights_lin.txt',tang_heights_lin, fmt = '%.15f', delimiter= '\t')


# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.scatter(range(0,SpecNumMeas), tang_heights_lin)
# plt.show()

ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros((SpecNumMeas,1))
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2 * (np.sqrt( ( extraHeight + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T,0), tot_r[:,0])))





#taylor exapnsion for f to do so we need y (data)

##
''' load data and pick wavenumber/frequency'''
#check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1



mol_M = 48 #g/mol for Ozone
#ind = 293
ind = 623
#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * temp_values )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((SpecNumLayers,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]




'''weighted absorption cross section according to Hitran and MIPAS instrument description
S is: The spectral line intensity (cm^−1/(molecule cm^−2))
f_broad in (1/cm^-1) is the broadening due to pressure and doppler effect,
 usually one can describe this as the convolution of Lorentz profile and Gaussian profile
 VMR_O3 is the ozone profile in units of molecule (unitless)
 has to be extended if multiple gases are to be monitored
 I multiply with 1e-4 to go from cm^2 to m^2
 '''
f_broad = 1
w_cross =   f_broad * 1e-4 * VMR_O3#np.mean(VMR_O3) * np.ones((SpecNumLayers,1))
#w_cross[0], w_cross[-1] = 0, 0

#from : https://hitran.org/docs/definitions-and-units/
HitrConst2 = 1.4387769 # in cm K

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ temp_values)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineInt = S[ind,0] * Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa


scalingConst = 1e11
# A_scal_T = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm * num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]
#
# theta_O3 = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]


A_scal_O3 = 1e2 * LineIntScal  * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0] * num_mole / temp_values.reshape((SpecNumLayers,1))
#scalingConst = 1e11

theta_P = pressure_values.reshape((SpecNumLayers,1))

""" plot forward model values """


A = A_lin * A_scal_O3.T
np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))


Ax = np.matmul(A, theta_P)
SNR = 275
#convolve measurements and add noise
y, gamma  = add_noise(Ax, SNR)#90 works fine
#np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')

#gamma = 3.1120138500473094e-10
#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
#y = np.loadtxt('dataYtest022.txt').reshape((SpecNumMeas,1))
ATy = np.matmul(A.T,y)
# gamma = 7.6e-5
#SNR = np.mean(Ax**2)/np.var(y)
#SNR = np.mean(np.abs(Ax) ** 2)*gamma
#print(SNR)
#gamma = 1/(np.max(Ax) * 0.1)**2

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''


##

# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))

for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')

A, theta_scale_O3= composeAforO3(A_lin, temp_values, pressure_values, ind)
ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T, A)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
def MinLogMargPostFirst(params):#, coeff):
    tol = 1e-8
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    #ATA = np.matmul(A.T,A)
    Bp = ATA + lamb * L

    #y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
    #ATy = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)


gamma0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma,(np.var(VMR_O3) * theta_scale_O3) /gamma ])
mu0 = 0
print(lam0)
print('delta:' + str(lam0*gamma0))
##

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
#plt.show()
#print(1/np.var(y))
print("gamma:" + str(gamma))

##
"""update A so that O3 profile is constant"""
#O3_Prof = np.mean(VMR_O3) * np.ones(SpecNumLayers)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
#ax1.plot(O3_Prof, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o')
#ax1.plot(O3, heights, linewidth = 2.5, label = 'true profile', marker = 'o')

ax1.set_ylabel('Height in km')
ax1.set_xlabel('Volume Mixing Ratio of Ozone')
# ax2 = ax1.twiny()
# ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
# ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,

ax1.legend()
# plt.savefig('DataStartTrueProfile.png')
plt.show()



##

def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return -b * (x - h0) + np.log(p0)

popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], np.log(pressure_values), p0=[-2e-2,-2e-2, 18, 15])

#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.plot(pressure_values,height_values, linewidth = 2)
# #ax1.plot(np.exp(pressFunc(height_values[:,0], -0.12,-0.2)), height_values, linewidth = 2)
# ax1.plot(np.exp(pressFunc(height_values[:,0], *popt)), height_values[:,0], linewidth = 2)
# ax1.axhline(y=popt[2])
# ax1.axvline(x=popt[3])
# ax1.set_xlabel(r'Pressure in hPa ')
# ax1.set_ylabel('Height in km')
# #ax1.set_xscale('log')
# plt.savefig('samplesPressure.png')
# plt.show()

##
''' t-walk for temperature start'''


h0 = 11
h1 = 20
h2 = 32
h3 = 47
h4 = 51
h5 = 71


a0 = -6.5
a1 = 1
a2 = 2.8
a3 = -2.8
a4 = -2
b0 = 288.15
# #b1 = 288.15 + h0 * a0
# #
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
# ax1.plot(temp_func(height_values,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0), height_values, linewidth=2, label='reconst', color='red', zorder=1)
# #plt.savefig('TemperatureSamp.png')
# plt.show()
# #
# #
# A, theta_scale_T = composeAforTemp(A_lin, pressure_values, VMR_O3, ind, temp_values)
#
# def log_post_temp(Params):
#
#     n = SpecNumLayers
#     m = SpecNumMeas
#     h0 = Params[0]
#     h1 = Params[1]
#     h2 = Params[2]
#     h3 = Params[3]
#     h4 = Params[4]
#     h5 = Params[5]
#     a0 = Params[6]
#     a1 = Params[7]
#     a2 = Params[8]
#     a3 = Params[9]
#     a4 = Params[10]
#     b0 = Params[11]
#
#     h0Mean = 11
#     h0Sigm = 0.5
#
#     h1Mean = 20
#     h1Sigm = 3
#
#     h2Mean = 32
#     h2Sigm = 1
#
#     h3Mean = 47
#     h3Sigm = 2
#
#     h4Mean = 51
#     h4Sigm = 2
#
#     h5Mean = 71
#     h5Sigm = 2
#
#     a0Mean = -6.5
#     a0Sigm = 0.01
#
#     a1Mean = 1
#     a1Sigm = 0.01
#
#     a2Mean = 2.8
#     a2Sigm = 0.1
#
#     a3Mean = -2.8
#     a3Sigm = 0.01
#
#     a4Mean = -2
#     a4Sigm = 0.01
#
#     b0Mean = 288.15
#     b0Sigm = 2
#
#
#     return gamma * np.sum((y - A @ (1/temp_func(height_values,*Params).reshape((n,1)))) ** 2) + ((h0-h0Mean)/h0Sigm)**2 + ((h1-h1Mean)/h1Sigm)**2 + ((h2-h2Mean)/h2Sigm)**2 +  ((h3-h3Mean)/h3Sigm)**2+  ((h4-h4Mean)/h4Sigm)**2+ ((h5-h5Mean)/h5Sigm)**2+  ((a0-a0Mean)/a0Sigm)**2+  ((a1-a1Mean)/a1Sigm)**2+ ((a2-a2Mean)/a2Sigm)**2\
#         + ((a3-a3Mean)/a3Sigm)**2 + ((a4-a4Mean)/a4Sigm)**2+ ((b0-b0Mean)/b0Sigm)**2
#
#
#
# def MargPostSupp_temp(Params):
#     list = []
#     return all(list)


# MargPost = pytwalk.pytwalk(n=10, U=log_post_temp, Supp=MargPostSupp_temp)
# x0 = np.array([h0,h1,h2,h3,h4,a0,a1,a2,a3,b0])
# xp0 = 1.01 * x0
# TempBurnIn = 5000
# TempWalkSampNum = 100000
# MargPost.Run(T=TempWalkSampNum + TempBurnIn, x0=x0, xp0=xp0)
##
# TempSamps = MargPost.Output
# paraSamp = 100#
# TempResults = np.zeros((paraSamp,n))
# randInd = np.random.randint(low = burnIn, high= burnIn+TempWalkSampNum, size = paraSamp)
#
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
#
# for p in range(0,paraSamp):
#     TempResults[p] = temp_func(height_values[:,0], *TempSamps[randInd[p],0:-1])
#     ax1.plot(TempResults[p], height_values[:,0], linewidth=0.2, label='reconst', zorder=1)
#
# temp_Prof = np.mean(TempResults,0)
# ax1.plot(temp_Prof, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,markersize=5)
# plt.show()

''' t-walk for temperature end'''
## accept data set

def gamDist(x, mean, sigma):
    return (2* np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * (x - mean) ** 2 / sigma** 2)

def twoNormDist(X, Y, meanX, sigX, meanY, sigY):
    Mat = np.zeros((len(Y),len(X)))
    for i in range(0,len(X)):
        for j in range(0,len(Y)):
             Mat[j,i] = 1/(2 * np.pi * sigX * sigY) * np.exp(
                -0.5 * (X[i] - meanX) ** 2 / sigX ** 2) * np.exp(-0.5 * (Y[j] - meanY) ** 2 / sigY ** 2)
    return Mat


def SingtwoNormDist(X, Y, meanX, sigX, meanY, sigY):
    return 1/(2 * np.pi * sigX * sigY) * np.exp(
                -0.5 * (X - meanX) ** 2 / sigX ** 2) * np.exp(-0.5 * (Y - meanY) ** 2 / sigY ** 2)

Y = np.linspace(gamma*0.5, gamma*1.5,100)

X = np.linspace(1e-4, 3e-4)

Z = twoNormDist(X, Y, 1.9e-4, 2e-5, gamma, gamma * 0.05)/ np.sum(twoNormDist(X, Y, 1.9e-4, 2e-5, gamma, gamma * 0.05))
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# plt.pcolormesh(X,Y,Z)
# #plt.imshow(Mat, cmap=mpl.cm.hot)
# plt.colorbar()
# plt.show()
# x = np.linspace(0, 3e-6)
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
# #ax1.plot(x,gamDist(x, gamma, gamma*0.2)/np.sum(gamDist(x, gamma, gamma*0.2)), linewidth=5, color='green', zorder=0)
# #ax1.plot(xdel,gamDist(xdel, 1.8e-4,2.5e-6)/np.sum(gamDist(xdel, 1.8e-4,2.5e-6)), linewidth=5, color='green', zorder=0)
# ax1.plot(x,gamDist(x, 9e-7,3.5e-7), linewidth=5, color='green', zorder=0)
#
# plt.show()
##
'''do the sampling'''
def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x>h0] = b2
    b[x<=h0] = b1
    return np.exp(-b * (x - h0) + np.log(p0))

def Parabel(x, h0, a0, d0):

    return a0 * np.power((h0-x),2 )+ d0
##
# tests = 30
# for t in range(0,tests):
#
#     A, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
#     Ax = np.matmul(A, VMR_O3 * theta_scale_O3)
#     y, gamma = add_noise(Ax, SNR)  # 90 works fine
#     y = y.reshape((m,1))
#     #y = np.loadtxt('dataYtest002.txt').reshape((SpecNumMeas, 1))
#     ATy = np.matmul(A.T, y)
#     ATA = np.matmul(A.T, A)
#     print(1/np.var(y[0:12]))

##

tests = 100
for t in range(0,tests):

    A, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
    Ax = np.matmul(A, VMR_O3 * theta_scale_O3)
    y, gamma = add_noise(Ax, SNR)  # 90 works fine
    y = y.reshape((m,1))
    #y = np.loadtxt('dataYtest003.txt').reshape((SpecNumMeas, 1))
    ATy = np.matmul(A.T, y)
    ATA = np.matmul(A.T, A)


    def MinLogMargPostFirst(params):  # , coeff):
        tol = 1e-8
        # gamma = params[0]
        # delta = params[1]
        gam = params[0]
        lamb = params[1]
        if lamb < 0 or gam < 0:
            return np.nan

        # ATA = np.matmul(A.T,A)
        Bp = ATA + lamb * L

        # y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
        # ATy = np.matmul(A.T, y)
        B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:, 0], tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

        G = g(A, L, lamb)
        F = f(ATy, y, B_inv_A_trans_y)

        return -n / 2 * np.log(lamb) - (m / 2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F + (
                    betaD * lamb * gam + betaG * gam)


    gamma0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma, (np.var(VMR_O3) * theta_scale_O3) / gamma])

    #don't accept if gamma0 is unlikey according to prio
    #while np.random.uniform() > SingtwoNormDist(gamma0 * lam0, gamma, 1.9e-4, 2e-5, gamma, gamma *0.05) / np.sum(twoNormDist(X, Y, 1.9e-4, 2e-5, gamma, gamma * 0.05)):

    # while 1/np.var(y[20:]) <5e-10 or 1/np.var(y[20:]) > 6.5e-10 or 1/np.var(y[0:15]) <1.9e-10 or 1/np.var(y[0:15]) > 2.1e-10:
    # #while 1 / np.var(y[20:]) < 3e-10 or 1 / np.var(y[20:]) > 4e-10 or 1 / np.var(y[0:15]) < 1e-10 or 1 / np.var(y[0:15]) > 1.3e-10:
    #
    #     print("sim again")
    #     y, gamma = add_noise(Ax, SNR)  # 90 works fine
    #     y = y.reshape((m, 1))
    #     ATy = np.matmul(A.T, y)
    #     ATA = np.matmul(A.T, A)


    #while gamma0 * lam0 < 1.6e-4 or gamma0 * lam0 > 1.8e-4 or gamma0 < 3.6e-10 or gamma0 > 3.7e-10:
        # y, gamma = add_noise(Ax, SNR)  # 90 works fine
        # y = y.reshape((m, 1))
        # ATy = np.matmul(A.T, y)
        # ATA = np.matmul(A.T, A)
        #
        #
        # def MinLogMargPostFirst(params):  # , coeff):
        #     tol = 1e-8
        #     # gamma = params[0]
        #     # delta = params[1]
        #     gam = params[0]
        #     lamb = params[1]
        #     if lamb < 0 or gam < 0:
        #         return np.nan
        #
        #     # ATA = np.matmul(A.T,A)
        #     Bp = ATA + lamb * L
        #
        #     # y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
        #     # ATy = np.matmul(A.T, y)
        #     B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:, 0], tol=tol, restart=25)
        #     if exitCode != 0:
        #         print(exitCode)
        #
        #     G = g(A, L, lamb)
        #     F = f(ATy, y, B_inv_A_trans_y)
        #
        #     return -n / 2 * np.log(lamb) - (m / 2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F + (
        #             betaD * lamb * gam + betaG * gam)
        #
        #
        # gamma0, lam0 = optimize.fmin(MinLogMargPostFirst, [gamma, (np.var(VMR_O3) * theta_scale_O3) / gamma])

    np.savetxt('dataYtest' + str(t).zfill(3) + '.txt', y, header = 'Data y including noise', fmt = '%.15f')



    SampleRounds = 255

    print(np.mean(VMR_O3))

    number_samples =1500
    recov_temp_fit = temp_values#np.mean(temp_values) * np.ones((SpecNumLayers,1))
    recov_press = pressure_values#np.mean(pressure_values) * np.ones((SpecNumLayers,1))#1013 * np.exp(-np.mean(grad) * height_values[:,0])
    Results = np.zeros((SampleRounds-1, len(VMR_O3)))
    TempResults = np.zeros((SampleRounds, len(VMR_O3)))
    PressResults = np.zeros((SampleRounds, len(VMR_O3)))
    deltRes = np.zeros((SampleRounds,3))
    gamRes = np.zeros(SampleRounds)
    round = 0
    burnInDel = 1500
    tWalkSampNumDel = 20000

    tWalkSampNum = 5000
    burnInT =100
    burnInMH =100

    deltRes[0,:] = np.array([ 30,1e-6, 1e-4])#lam0 * gamma0*0.4])
    gamRes[0] = gamma
    SetDelta = Parabel(height_values,*deltRes[0,:])
    SetGamma =  gamRes[0]
    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]

    B0 = (ATA + 1 / gamRes[0] *  L_d)
    B_inv_A_trans_y0, exitCode = gmres(B0, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    PressResults[0, :] = pressure_values
    TempResults[0,:] = temp_values.reshape(n)

    def MargPostSupp(Params):
        list = []
        list.append(Params[0] > 0)
        list.append(height_values[-1] > Params[1] > height_values[0])
        list.append(Params[2] > 0)
        list.append( Params[3] > 0)
        return all(list)

    def log_post(Params):
        tol = 1e-8
        n = SpecNumLayers
        m = SpecNumMeas

        gam = Params[0]
        h1 = Params[1]
        a0 = Params[2]
        d0 = Params[3]

        delta = Parabel(height_values,h1, a0, d0)
        TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
        TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
        Diag = np.eye(n) * np.sum(TriU + TriL, 0)

        L_d = -TriU + Diag - TriL
        L_d[0, 0] = 2 * L_d[0, 0]
        L_d[-1, -1] = 2 * L_d[-1, -1]

        try:
            L_du, L_ds, L_dvh = np.linalg.svd(L_d)
            detL = np.sum(np.log(L_ds))
        except np.linalg.LinAlgError:
            print("SVD did not converge, use scipy.linalg.det()")
            detL = np.log(scy.linalg.det(L_d))

        Bp = ATA + 1/gam * L_d

        B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], x0 = B_inv_A_trans_y0, tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

        G = g(A, L_d,  1/gam)
        F = f(ATy, y,  B_inv_A_trans_y)
        alphaD =  1
        alphaG = 1
        hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])]
        # hMean = 25
        alphaA1 = (lam0 * gamma0) / (hMean - np.min(height_values)) ** 2
        alphaA2 = (lam0 * gamma0) / (hMean - np.max(height_values)) ** 2
        if alphaA2 > alphaA1:
            alphaA = alphaA2
        else:
            alphaA = alphaA1
            #d0-0.75e-4 9e-6
            d0Mean =0.8e-4
            #((a0 - 2e-7) / 1.25e-8) ** 2
            #0.5 * ((a0 - 4e-7) / 3e-7) ** 2
        return - (m / 2 - n / 2) * np.log(gam) - 0.5 * detL+ 0.5 * ((d0-d0Mean)/(1e-5))**2 + 0.5 * G + 0.5 * gam * F  + 0.5 * ((gam-gamma)/(gamma*0.01))**2  + 0.5 * ((Params[1] - hMean) / 1) ** 2 + 1e-7 * a0

    A, theta_scale_O3 = composeAforO3(A_lin, TempResults[round, :].reshape((n, 1)), PressResults[round, :], ind)
    ATy = np.matmul(A.T, y)
    ATA = np.matmul(A.T, A)

    MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
    x0 = np.array([SetGamma, *deltRes[round, :]])
    xp0 = 1.0001 * x0
    MargPost.Run(T=tWalkSampNumDel + burnInDel, x0=x0, xp0=xp0)

    Samps = MargPost.Output

    while round < SampleRounds-1:

        MWGRand = burnIn + np.random.randint(low=0, high=tWalkSampNumDel)
        SetGamma = Samps[MWGRand,0]
        SetDelta = Parabel(height_values, *Samps[MWGRand,1:-1])

        TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
        TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
        Diag = np.eye(n) * np.sum(TriU + TriL, 0)

        L_d = -TriU + Diag - TriL
        L_d[0, 0] = 2 * L_d[0, 0]
        L_d[-1, -1] = 2 * L_d[-1, -1]
        SetB = SetGamma * ATA +  L_d

        W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
        v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
        W2 = np.random.multivariate_normal(np.zeros(len(L)), L_d )
        v_2 = W2.reshape((n,1))

        RandX = (SetGamma * ATy + v_1 + v_2)
        Results[round, :], exitCode = gmres(SetB, RandX[0::, 0], tol=tol)
        O3_Prof = Results[round, :] / theta_scale_O3
        deltRes[round+1, :] = np.array([Samps[MWGRand, 1:-1]])
        gamRes[round+1] = SetGamma

        print(np.mean(O3_Prof))

        # A, theta_scale = composeAforPress(A_lin, TempResults[round, :].reshape((n,1)), O3_Prof, ind)
        # SampParas = tWalkPress(height_values, A, y, popt, tWalkSampNum, burnInT, SetGamma)
        # randInd = np.random.randint(low=0, high=tWalkSampNum)
        #
        # sampB1 = SampParas[burnInT + randInd,0]
        # sampB2 = SampParas[burnInT + randInd, 1]
        # sampA1 = SampParas[burnInT + randInd, 2]
        # sampA2 = SampParas[burnInT + randInd, 3]
        #
        # PressResults[round+1, :] = pressFunc(height_values[:,0], sampB1, sampB2, sampA1, sampA2)

        PressResults[round+1, :] = pressure_values

        # A, theta_scale_T = composeAforTemp(A_lin, PressResults[round+1,:], O3_Prof, ind, temp_values)
        #
        # TempBurnIn = 2500
        # TempWalkSampNum = 25000
        # TempSamps = tWalkTemp(height_values, A, y, TempWalkSampNum, TempBurnIn, SetGamma, SpecNumLayers, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3,a4, b0)
        # randInd = np.random.randint(low=0, high=TempWalkSampNum)
        #
        # h0 = TempSamps[TempBurnIn + randInd, 0]
        # h1 = TempSamps[TempBurnIn + randInd, 1]
        # h2 = TempSamps[TempBurnIn + randInd, 2]
        # h3 = TempSamps[TempBurnIn + randInd, 3]
        # h4 = TempSamps[TempBurnIn + randInd, 4]
        # h5 = TempSamps[TempBurnIn + randInd, 5]
        # a0 = TempSamps[TempBurnIn + randInd, 6]
        # a1 = TempSamps[TempBurnIn + randInd, 7]
        # a2 = TempSamps[TempBurnIn + randInd, 8]
        # a3 = TempSamps[TempBurnIn + randInd, 9]
        # a4 = TempSamps[TempBurnIn + randInd, 10]
        # b0 = TempSamps[TempBurnIn + randInd, 11]
        #
        #
        # TempResults[round+1, :] = temp_func(height_values,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0).reshape(n)

        TempResults[round + 1, :] = temp_values.reshape(n)
        round += 1
        print('Round ' + str(round))

    np.savetxt('deltRes'+ str(t).zfill(3) +'.txt', deltRes, fmt = '%.15f', delimiter= '\t')
    np.savetxt('gamRes'+ str(t).zfill(3) +'.txt', gamRes, fmt = '%.15f', delimiter= '\t')
    np.savetxt('O3Res'+ str(t).zfill(3) +'.txt', Results/theta_scale_O3, fmt = '%.15f', delimiter= '\t')
    np.savetxt('PressRes'+ str(t).zfill(3) +'.txt', PressResults, fmt = '%.15f', delimiter= '\t')
    np.savetxt('TempRes'+ str(t).zfill(3) +'.txt', TempResults, fmt = '%.15f', delimiter= '\t')

print('finished')
##

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)

# fig, axs = plt.subplots()#figsize = (7,  2))
# # We can set the number of bins with the *bins* keyword argument.
# axs.hist(lamRes,bins=200, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
# axs.axvline(x=lam0, color = "r", linewidth = 5)
# axs.set_title('$\lambda$ samples')
# #axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
# #axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')

#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()
fig, axs = plt.subplots()#figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(gamRes,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
axs.set_title('$\gamma$ samples')
#axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')
axs.axvline(x=gamma, color = 'r')
#tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
#plt.close()
plt.show()

##

# deltRes = np.loadtxt('deltRes.txt', delimiter= '\t')
# gamRes = np.loadtxt('gamRes.txt', delimiter= '\t')
# VMR_O3 = np.loadtxt('VMR_O3.txt', delimiter= '\t')
# O3Res = np.loadtxt('O3Res.txt', delimiter= '\t')
# PressResults = np.loadtxt('PressRes.txt', delimiter= '\t')
# Results = O3Res  * theta_scale_O3
# SampleRounds = len(gamRes)

##
plt.close('all')
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]

mpl.use(defBack)

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.serif"] = "cmr"

fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()

ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=1 ,linewidth = 1.5, markersize =7)

for r in range(1,SampleRounds-1):
    Sol = Results[r, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5)
    # with open('Samp' + str(n) +'.txt', 'w') as f:
    #     for k in range(0, len(Sol)):
    #         f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
    #         f.write('\n')
O3_Prof = np.mean(Results[0:],0)/ (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

ax1.plot(O3_Prof, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
             markersize=5)

ax1.set_xlabel(r'Ozone volume mixing ratio ')

ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.set_ylim([heights[minInd-1], heights[maxInd-1]])

#ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W } \text{cm}}{\text{m}^2 \text{ sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)
ax1.legend()
plt.savefig('O3Results.png')
plt.show()
##
fig3, ax1 = plt.subplots(tight_layout=True, figsize=set_size(245, fraction=fraction))
#ax1.plot(press, heights, label='true press.')
ax1.plot(pressure_values, height_values, label='true pressure', color = TrueCol, marker ='o', zorder =1, markersize=10)
#ax1.plot(recov_press, height_values, linewidth=2.5, label='samp. press. fit')  #
for r in range(0, SampleRounds):
    Sol = PressResults[r, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5,
             markersize=5)
PressProf = np.mean(PressResults[0:],0)
ax1.plot(PressProf, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
         markersize=5)

#ax1.plot(2500 * np.exp(-np.mean(grad) * height_values[:,0]),height_values[:,0])
ax1.set_xlabel(r'Pressure in hPa ')
ax1.set_ylabel('Height in km')
ax1.legend()
plt.savefig('samplesPressure.png')
plt.show()
##

fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))

for r in range(0, SampleRounds):
    Sol = Parabel(height_values, *deltRes[r, :])

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5)
ax1.set_xlabel(r'$\delta$ ')
ax1.set_ylabel('Height in km')
plt.savefig('DeltaSamp.png')
plt.show()

##
fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
for r in range(0, SampleRounds):
    Sol = TempResults[r, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5,
             markersize=5)

TempProf = np.mean(TempResults[0:], 0)
ax1.plot(TempProf, height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
         markersize=5)

ax1.plot(temp_values, height_values, linewidth=5, label='true T', color='green', zorder=0)
ax1.legend()
plt.savefig('TemperatureSamp.png')
plt.show()

#tikzplotlib.save("FirstRecRes.pgf")
print('done')



# def Parabel(x, h0, a0, d0):
#
#     return a0 * np.power((h0-x),2 )+ d0
#
#
#
# def oneParabeltoConst(x, h0, a0, d0):
#     a = np.ones(x.shape)
#     a[x <= h0] = a0
#     a[x > h0] = 0#-a1
#     return a * (h0 -x)**2 + d0
#
#
#
# B0 = ATA + lam0 * L
# B_inv_A_trans_y0, exitCode = gmres(B0, ATy[:,0], tol=tol, restart=25)
#
# B0u, B0s, B0vh = np.linalg.svd(B0)
# cond_B0 = np.max(B0s)/np.min(B0s)
# print("Condition Number B0: " + str(orderOfMagnitude(cond_B0)))
#
# def log_post(Params):
#     tol = 1e-8
#     n = SpecNumLayers
#     m = SpecNumMeas
#     # gamma = params[0]
#     # delta = params[1]
#     gam = Params[0]
#     h1 = Params[1]
#     a0 = Params[2]
#     # h0 = Params[2]
#
#
#     # mean = Params[1]
#     # w = Params[2]
#     # skewP = Params[3]
#     # scale = Params[4]
#     d0 = Params[3]
#     #a1 = Params[4]
#     delta = Parabel(height_values,h1, a0, d0)
#     #delta = oneParabeltoConst(height_values,h1, a0, d0)
#     #delta = simpleDFunc(height_values, h1,a0, d0)
#     #delta = twoParabel(height_values, a0, 0, h1, 0)
#     #delta = skew_norm_pdf(height_values, 16, 50, 8, 9.5e-05, 3.7e-03)
#     #delta = skew_norm_pdf(height_values[:,0],mean,w,skewP, scale, d0)
#     TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
#     TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
#     Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
#     L_d = -TriU + Diag - TriL
#     L_d[0, 0] = 2 * L_d[0, 0]
#     L_d[-1, -1] = 2 * L_d[-1, -1]
#
#     try:
#         L_du, L_ds, L_dvh = np.linalg.svd(L_d)
#         detL = np.sum(np.log(L_ds))
#     except np.linalg.LinAlgError:
#         print("SVD did not converge, use scipy.linalg.det()")
#         detL = np.log(scy.linalg.det(L_d))
#
#     Bp = ATA + 1/gam * L_d
#
#     B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], x0= B_inv_A_trans_y0, tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, L_d,  1/gam)
#     F = f(ATy, y,  B_inv_A_trans_y)
#     alphaD = 1
#     alphaG = 1
#     #hMean = tang_heights_lin[y[:,0] == np.max(y[:,0])]
#     #hMean = tang_heights_lin[Ax == np.max(Ax)]
#     hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])]
#     #hMean = 25
#     alphaA1 = (lam0 * gamma0*0.75) / (hMean - np.min(height_values))**2
#     alphaA2 = (lam0 * gamma0*0.75) / (hMean - np.max(height_values)) ** 2
#     if alphaA2 < alphaA1:
#         alphaA = 1/alphaA2
#     else:
#         alphaA = 1/alphaA1
#     #sigmaP = 100
#     #return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e1 *  np.sum(delta) + 1e2 *gam)+ ((8 - mean)/sigmaP) ** 2 + (( 1.7e-03 - d0)/1e-3) ** 2 + (( 5 - skewP)/10) ** 2 +(( 4.2e-05 - scale)/1e-4) ** 2 +(( 50 - w)/20) ** 2
#     #return - (0.5 + alphaD - 1 ) * np.sum(np.log(delta/gam))  - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e4 *  np.sum(delta)/n + betaG *gam)+ 0.5 * ((20 -Params[1])/25) ** 2 + 0.5* (( 1e-4 - Params[2])/2e-4) ** 2
#     #return - (0.5* n)  * np.log(1/gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 1e4 * d0 + betaG *gam) - 0 * np.log(Params[2]) + 1e3* Params[2] - 0.1*  np.log(Params[1]) + 1e-4* Params[1]
#     #return - (0.5* n)  * np.log(1/gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) - (m/2+1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( 3e4 * d0 + 1e5 *gam) - 11 * np.log(Params[1]) + 5e-1* Params[1] - 0.2*  np.log(Params[2]) + 5e7* Params[2]
#     #return - (m/2 - n/2 + alphaG -1) * np.log(gam) - 0.5 * np.sum(np.log(L_ds)) - (alphaD - 1) * np.log(d0) + 0.5 * G + 0.5 * gam * F +  (1/(lam0 * gamma0*1e-1) * d0 +7e9 *gam) - 0.3 * np.log(Params[1]) +1e-3 * Params[1] #- 0*  np.log(Params[2]) + 1e7* Params[2]
#     return - (m/2 - n/2 + alphaG -1) * np.log(gam) - 0.5 * detL - (alphaD - 1) * np.log(d0) + 0.5 * G + 0.5 * gam * F +  (1/(gamma0*lam0*0.4) * d0 + betaG *gam)  - 0*  np.log(Params[2]) + alphaA* Params[2]- 0 * np.log(Params[1]) + 0.5* ((Params[1]-hMean)/2)**2
#
#
#
# def MargPostSupp(Params):
#     list = []
#     list.append(Params[0] > 0)
#     list.append(height_values[-1]> Params[1] >height_values[0])
#     list.append(Params[2] > 0)
#     list.append(lam0 * gamma0 >Params[3] > 0)
#     return all(list)
#
#
# MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
# # startTime = time.time()
# #x0 = np.array([gamma, 8, 50, 5, 4.2e-05,1.7e-03])
# x0 = np.array([gamma,29, 5e-7, lam0 * gamma0*0.4])
# xp0 = 1.01 * x0
# burnIn = 500
# tWalkSampNum = 20000
# MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
#
# Samps = MargPost.Output
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,0], bins = 50)
# ax1.axvline(x=gamma, color = 'r')
# plt.show()
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,1], bins = 50)
# #ax1.axvline(x=popt[1], color = 'r')
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,2], bins = 50)
#
# plt.show()
#
#
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# ax1.hist(Samps[:,3], bins = 50)
# plt.show()
# ##
# fig3, ax1 = plt.subplots(figsize=set_size(245, fraction=fraction))
#
# for p in range(burnIn, tWalkSampNum,500):
#     Sol = Parabel(height_values, *Samps[p, 1:-1])
#
#     ax1.plot(Sol, height_values, linewidth=0.5)
# ax1.set_xlabel(r'$\delta$ ')
# ax1.set_ylabel('Height in km')
#
#
# plt.show()
#
# ##
# xm = np.mean(Samps[:,2])
#
# hMean = 27.5  # tang_heights_lin[y[:,0] == np.max(y[:,0])]
# alphaA1 = (lam0 * gamma0 * 0.6) / (hMean - np.min(height_values)) ** 2
# alphaA2 = (lam0 * gamma0 * 0.6) / (hMean - np.max(height_values)) ** 2
# if alphaA2 < alphaA1:
#     alphaA = 1 / alphaA2
# else:
#     alphaA = 1 / alphaA1
# def normalprior(x):
#     sigma =2
#
#
#     return 1/sigma * np.exp(-0.5 * ((x - xm)/(sigma))**2)
#
# def expDelta(x, a,b,d0):
#     # a = 4
#     # b = 1e-1
#     # d0 = 50
#     return x**a * np.exp(-b * x) + d0
# xTry = np.linspace(0,3*(xm),100)
# fig3, ax1 = plt.subplots()
# #ax1.scatter(xTry, normalprior(xTry) , color = 'r')
# ax1.scatter(xTry, expDelta(xTry,0,alphaA,0) , color = 'r')
# ax1.axvline(x=xm, color = 'r')
# #ax1.scatter(expDelta(height_values,4,1e-1,50), height_values, color = 'r')
# plt.show()
#
# ##
# #ds = oneParabeltoConst(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2])-1.6e-7,np.mean(Samps[:,3])+1e-5)
# ds = Parabel(height_values,np.mean(Samps[:,1]),np.mean(Samps[:,2]),np.mean(Samps[:,3]))
#
# fig3, ax1 = plt.subplots()
# ax1.scatter(ds,height_values, color = 'r')
# #ax1.scatter(paraDs,height_values, color = 'b')
# plt.show()
#
#
#
# n = SpecNumLayers
# m = SpecNumMeas
# paraSamp = 100#n_bins
# NewResults = np.zeros((paraSamp,n))
# #SetDelta = skewDsTry #ds
# SetGamma = gamma
# randInd = np.random.randint(low=burnIn, high=tWalkSampNum+burnIn, size = paraSamp)
# for p in range(paraSamp):
#     SetGamma = Samps[randInd[p],0]
#     #SetDelta = twoParabel(height_values,Samps[randInd[p],1], 0, Samps[randInd[p],2],0)
#     #SetDelta = simpleDFunc(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 2])
#     #SetDelta = oneParabeltoConst(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 3])
#     SetDelta = Parabel(height_values, Samps[randInd[p], 1], Samps[randInd[p], 2],  Samps[randInd[p], 3])
#     #SetDelta = ds
#     Mu = np.zeros((n,1))
#     #Mu = 0.3e-6 * theta_scale_O3
#     TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
#     TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
#     Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
#     L_d = -TriU + Diag - TriL
#     L_d[0, 0] = 2 * L_d[0, 0]
#     L_d[-1, -1] = 2 * L_d[-1, -1]
#     SetB = SetGamma * ATA +  L_d
#
#     W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)) )
#     v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m,1))
#     W2 = np.random.multivariate_normal(np.zeros(len(L)), L_d )
#     v_2 = W2.reshape((n,1))
#
#     RandX = (SetGamma * ATy + L_d @ Mu + v_1 + v_2)
#     NewResults[p,:], exitCode = gmres(SetB, RandX[0::, 0], tol=tol)
#     print(np.mean(NewResults[p,:]))
# ResCol = "#1E88E5"
# fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
# #ax1.plot(Res/theta_scale_O3, height_values, linewidth = 2.5, label = 'my guess', marker = 'o')
#
# for p in range(0, paraSamp):
#     Sol = NewResults[p, :] / theta_scale_O3
#     ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=1, linewidth=0.5, markersize=5)
#
# ax1.plot(VMR_O3, height_values, linewidth = 2.5, label = 'true profile', marker = 'o', color = "k")
# O3_Prof = np.mean(NewResults,0)/ theta_scale_O3
#
# ax1.plot(O3_Prof, height_values, marker='>', color="k", zorder=2, linewidth=0.5,
#              markersize=5)
# ax1.set_ylabel('Height in km')
# ax1.set_xlabel('Volume Mixing Ratio of Ozone')
# ax2 = ax1.twiny()
# ax2.scatter(y, tang_heights_lin ,linewidth = 2, marker =  'x', label = 'data' , color = 'k')
# ax2.set_xlabel(r'Spectral radiance in $\frac{W cm}{m^2  sr} $',labelpad=10)# color =dataCol,
# ax1.legend()
# #plt.savefig('DataStartTrueProfile.png')
# plt.show()
#
#
# print('bla')
