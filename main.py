import numpy as np
import os
import matplotlib as mpl
from functions import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy


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
np.savetxt('temp_values.txt',temp_values, fmt = '%.15f', delimiter= '\t')


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



#meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
pointAcc = np.linspace(0.00025, 0.0009,15)
SNRs = np.linspace(5,281,35)
# pointAcc = np.linspace(0.00025, 0.0009,1)
# SNRs = np.linspace(4,300,1)
for s in  range(0,len(SNRs)):
    SNR = SNRs[s]
    for i in range(0,len(pointAcc)):

        parent_dir= '/home/lennartgolks/PycharmProjects/testSNRandRes/'
        filepath = 'SNR' + str(int(SNR)) + '/pointAcc' + str(pointAcc[i]) + '/'

        # Path
        path = os.path.join(parent_dir, filepath)

        try:
            os.makedirs(path, exist_ok=True)
            print("Directory '%s' created successfully" % filepath)
        except OSError as error:
            print("Directory '%s' can not be created" % filepath)

        meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc[i]))
        SpecNumMeas = len(meas_ang)
        m = SpecNumMeas
        print(m)

        A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R_Earth)
        np.savetxt(filepath + 'tang_heights_lin.txt',tang_heights_lin, fmt = '%.15f', delimiter= '\t')

        tot_r = np.zeros((SpecNumMeas,1))
        #calculate total length
        for j in range(0, SpecNumMeas):
            tot_r[j] = 2 * (np.sqrt( ( extraHeight + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2) )
        print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T,0), tot_r[:,0])))
        def Parabel(x, h0, a0, d0):
            return a0 * np.power((h0-x),2 )+ d0
        ##

        tests = 50
        for t in range(0,tests):

            A, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
            np.savetxt(filepath + 'AMat.txt', A, fmt='%.15f', delimiter='\t')
            Ax = np.matmul(A, VMR_O3 * theta_scale_O3)
            y, gamma = add_noise(Ax, SNR)  # 90 works fine
            y = y.reshape((m,1))
            #y = np.loadtxt('dataYtest003.txt').reshape((SpecNumMeas, 1))
            ATy = np.matmul(A.T, y)
            ATA = np.matmul(A.T, A)

            np.savetxt(filepath + 'dataYtest' + str(t).zfill(3) + '.txt', y, header = 'Data y including noise', fmt = '%.15f')

            SampleRounds = 500

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
            tWalkSampNumDel = 40000

            tWalkSampNum = 5000
            burnInT =100
            burnInMH =100

            deltRes[0,:] = np.array([ 30,1e-7, 1e-4])#lam0 * gamma0*0.4])
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
                hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])]
                d0Mean =0.8e-4

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


                PressResults[round+1, :] = pressure_values

                TempResults[round + 1, :] = temp_values.reshape(n)
                round += 1
                print('Round ' + str(round))

            np.savetxt(filepath + 'deltRes'+ str(t).zfill(3) +'.txt', deltRes, fmt = '%.15f', delimiter= '\t')
            np.savetxt(filepath +'gamRes'+ str(t).zfill(3) +'.txt', gamRes, fmt = '%.15f', delimiter= '\t')
            np.savetxt(filepath +'O3Res'+ str(t).zfill(3) +'.txt', Results/theta_scale_O3, fmt = '%.15f', delimiter= '\t')
            #np.savetxt(filepath +'PressRes'+ str(t).zfill(3) +'.txt', PressResults, fmt = '%.15f', delimiter= '\t')
            #np.savetxt(filepath +'TempRes'+ str(t).zfill(3) +'.txt', TempResults, fmt = '%.15f', delimiter= '\t')

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


