import math
import numpy as np
from numpy.random import uniform, normal, gamma
from scipy.sparse.linalg import gmres
import time, pytwalk
from scipy import constants
import pandas as pd
def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

'''Generate Forward Map A
where each collum is one measurment defined by a tangent height
every entry is length in km each measurement goes through
first non-zero entry of each row is the lowest layer (which should have a Ozone amount of 0)
last entry of each row is highest layer (also 0 Ozone)'''
def gen_sing_map(meas_ang, height, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # add one extra layer so that the difference in height can be calculated
    layers = np.zeros((len(height)+1,1))
    layers[0:-1] = height
    layers[-1] = height[-1] + (height[-1] - height[-2])/2


    A_height = np.zeros((num_meas, len(layers)-1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:

            t += 1

        # first dr
        A_height[m, t - 1] = np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]

    return 2 * A_height, tang_height, layers[-1]


def generate_L(neigbours):
    #Dirichlet Boundaries
    siz = int(np.size(neigbours, 0))
    neig = np.size(neigbours, 1)
    L = np.zeros((siz, siz))

    for i in range(0, siz):
        L[i, i] = neig
        for j in range(0, neig):
            if ~np.isnan(neigbours[i, j]):
                L[i, int(neigbours[i, j])] = -1
    #non periodic boundaries Neumann
    # L[0,0] = 1
    # L[-1,-1] = 1
    #periodic boundaires
    # L[0,-1] = -1
    # L[-1,0] = -1

    return L


def get_temp_values(height_values):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    temp_values = np.zeros(len(height_values))
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    ###calculate temp values
    for i in range(0, len(height_values)):
        if height_values[i] <= 11:
            temp_values[i] = np.around(- height_values[i] * 6.49 + 288.15,2)
        if 11 < height_values[i] <= 25:
            temp_values[i] = np.around(216.76,2)
        if 20 < height_values[i] <= 32:
            temp_values[i] = np.around(216.76 + (height_values[i] - 20) * 1, 2)
        if 32 < height_values[i] <= 47:
            temp_values[i] = np.around(228.76 + (height_values[i] - 32) * 2.8, 2)
        if 47 < height_values[i] <= 51:
            temp_values[i] = np.around(270.76, 2)
        if 51 < height_values[i] <= 71:
            temp_values[i] = np.around(270.76 - (height_values[i] - 51) * 2.8, 2)
        if 71 < height_values[i] <= 85:
            temp_values[i] = np.around(214.76 - (height_values[i] - 71) * 2.0, 2)
        if 85 < height_values[i]:
            temp_values[i] = 186.8

    return temp_values.reshape((len(height_values),1))


def get_temp(height_value):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    ###calculate temp values

    if height_value < 11:
        temp_value = np.around(- height_value * 6.49 + 288.15,2)
    if 11 <= height_value < 20:
        temp_value = np.around(216.76 ,2)
    if 20 <= height_value < 32:
        temp_value = np.around(216.76 + (height_value - 20) * 1,2)
    if 32 <= height_value < 47:
        temp_value = np.around(228.76 + (height_value - 32)* 2.8, 2)
    if 47 <= height_value < 51:
        temp_value = np.around(270.76, 2)
    if 51 <= height_value < 71:
        temp_value = np.around(270.76 - (height_value - 51) * 2.8 ,2)
    if 71 <= height_value < 85:
        temp_value = np.around(214.76 - (height_value -71) * 2.0 ,2)
    if  85 <= height_value:
        temp_value = 186.8

    return temp_value


# def add_noise(Ax, percent):
#     return Ax + np.random.normal(0, percent * np.max(Ax), (len(Ax), 1))


def add_noise(signal, snr_percent):
    """
    Add noise to a signal based on the specified SNR (in percent).

    Parameters:
        signal: numpy array
            The original signal.
        snr_percent: float
            The desired signal-to-noise ratio in percent.

    Returns:
        numpy array
            The signal with added noise.
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise power based on SNR (in percent)
    #noise_power = signal_power / (100 / snr_percent)
    noise_power = signal_power / snr_percent

    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, 1/noise_power

def gaussian(x, mu, sigma):
    """
    Compute the value of the Gaussian (normal) distribution at point x.

    Parameters:
        x: float or numpy array
            The point(s) at which to evaluate the Gaussian function.
        mu: float
            The mean (average) of the Gaussian distribution.
        sigma: float
            The standard deviation (spread) of the Gaussian distribution.

    Returns:
        float or numpy array
            The value(s) of the Gaussian function at point x.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
def calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, theta, w_cross, AscalConstKmToCm, SpecNumLayers, SpecNumMeas ):

    ConcVal = - pressure_values.reshape(
        (SpecNumLayers, 1)) * 1e2 * LineIntScal * AscalConstKmToCm / temp_values * theta * w_cross.reshape(
        (SpecNumLayers, 1))


    mask = A_lin * np.ones((SpecNumMeas, SpecNumLayers))
    mask[mask != 0] = 1

    ConcValMat = np.tril(np.ones(len(ConcVal))) * ConcVal
    ValPerLayPre = mask * (A_lin @ ConcValMat)
    preTrans = np.exp(ValPerLayPre)

    ConcValMatAft = np.triu(np.ones(len(ConcVal))) * ConcVal
    ValPerLayAft = mask * (A_lin @ ConcValMatAft)
    BasPreTrans = (A_lin @ ConcValMat)[0::, 0].reshape((SpecNumMeas, 1)) @ np.ones((1, SpecNumLayers))
    afterTrans = np.exp( (BasPreTrans + ValPerLayAft) * mask )


    return preTrans + afterTrans



def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    Bu, Bs, Bvh = np.linalg.svd(B)
    # np.log(np.prod(Bs))
    return np.sum(np.log(Bs))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)


def g_tayl(delta_lam, g_0, trace_B_inv_L_1, trace_B_inv_L_2, trace_B_inv_L_3, trace_B_inv_L_4, trace_B_inv_L_5, trace_B_inv_L_6):
    return g_0 + trace_B_inv_L_1 * delta_lam + trace_B_inv_L_2 * delta_lam**2 + trace_B_inv_L_3 * delta_lam**3 + trace_B_inv_L_4 * delta_lam**4 + trace_B_inv_L_5 * delta_lam**5 + trace_B_inv_L_6 * delta_lam**6


def f_tayl( delta_lam, f_0, f_1, f_2, f_3, f_4):
    return f_0 + f_1 * delta_lam + f_2 * delta_lam**2 + f_3 * delta_lam**3 + f_4 * delta_lam**4# + f_5 * delta_lam**5 #- f_6 * delta_lam**6


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def MHwG(number_samples,A ,burnIn, lambda0, gamma0, y, ATA, Prec, B_inv_A_trans_y0, ATy, tol, betaG, betaD, B0):
    wLam = lambda0/1.5
    SpecNumMeas, SpecNumLayers = np.shape(A)
    B_inv_L = np.zeros(np.shape(B0))

    for i in range(SpecNumLayers):
        B_inv_L[:, i], exitCode = gmres(B0, Prec[:, i], tol=tol, restart=25)
        if exitCode != 0:
            print('B_inv_L ' + str(exitCode))

    B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
    B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)


    f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y0)
    f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
    f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_3), B_inv_A_trans_y0)

    g_0_1 = np.trace(B_inv_L)
    g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
    g_0_3 = 1 / 6 * np.trace(B_inv_L_3)



    alphaG = 1
    alphaD = 1.001
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0



    shape = SpecNumMeas / 2 + alphaD + alphaG
    #print(f(ATy, y, B_inv_A_trans_y0))
    rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]
        # B = (ATA + lam_p * L)
        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)


        # f_new = f(ATy, y,  B_inv_A_trans_y)
        # g_new = g(A, L,  lam_p)
        #
        # delta_f = f_new - f_old
        # delta_g = g_new - g_old

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/2 + alphaD - 1) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            B = (ATA + lam_p * Prec)
            B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0,tol=tol, restart=25)
            #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

            # if exitCode != 0:
            #         print(exitCode)

            f_new = f(ATy, y,  B_inv_A_trans_y)
            #print(f(ATy, y, B_inv_A_trans_y))
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])


        #print(1/rate)
        gammas[t+1] = gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k



def tWalkPress(x, A, y, popt, tWalkSampNum, burnIn, gamma):
    def pressFunc(x, b1, b2, h0, p0):
        b = np.ones(len(x))
        b[x > h0] = b2
        b[x <= h0] = b1
        return np.exp(-b * (x - h0) + np.log(p0))

    SpecNumMeas, SpecNumLayers  = np.shape(A)
    def log_post(Params):
        b1 = Params[0]
        b2 = Params[1]
        h0 = Params[2]
        p0 = Params[3]
        #return gamma * np.sum((y - A @ pressFunc(x[:, 0], b1, b2, h0, p0).reshape((SpecNumLayers, 1))) ** 2) + 1e-4 * p0 + 1e-5 * h0 + 1e-5 * (b1 + b2)
        sigmaP = 0.25
        sigmaH = 0.5
        sigmaGrad1 = 0.005
        sigmaGrad2 = 0.01
        #return gamma * np.sum((y - A @ pressFunc(x[:, 0], b1, b2, h0, p0).reshape((SpecNumLayers, 1))) ** 2) + ((popt[3] - p0)/sigmaP) ** 2 + ((popt[2] - h0)/sigmaH) ** 2 + 1/sigmaGrad**2 * ((np.mean(popt[0:2]) - b1) ** 2 + (np.mean(popt[0:2]) - b2) ** 2)
        return gamma * np.sum((y - A @ pressFunc(x[:, 0], b1, b2, h0, p0).reshape((SpecNumLayers, 1))) ** 2) + ( (popt[3] - p0) / sigmaP) ** 2 + ((popt[2] - h0) / sigmaH) ** 2 + ((popt[0] - b1)/sigmaGrad1) ** 2 + ((popt[1] - b2)/sigmaGrad2) ** 2



    def MargPostSupp(Params):
        list = []
        list.append(Params[0] > 0)
        list.append(Params[1] > 0)
        list.append(Params[2] > 0)  # 6.5)
        list.append(Params[3] > 0)  # 5.5)
        #list.append(Params[0] > Params[1])
        return all(list)

    MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
    #startTime = time.time()
    x0 = popt * 1.1
    xp0 = 1.01 * x0
    #print(" Support of Starting points:" + str(MargPostSupp(x0)) + str(MargPostSupp(xp0)))
    MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
    #elapsedtWalkTime = time.time() - startTime
    #print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
    #MargPost.Ana()
    #MargPost.SavetwalkOutput("MargPostDat.txt")
    return MargPost.Output

def updateTemp(x, t, p):
    R_Earth = 6371
    grav = 9.81 * ((R_Earth) / (R_Earth + x)) ** 2
    R = constants.gas_constant

    del_height = x[1:,0] - x[:-1,0]
    recov_temp = np.zeros((len(del_height), 1))

    for i in range(0, len(del_height)):

        recov_temp[i] = -28.97 * grav[i] / np.log(p[i + 1] / p[i]) / R * del_height[i]

    recov_temp[recov_temp < 0.1 * np.mean(t)] = np.nan
    recov_temp[recov_temp > 2 * np.mean(t)] = np.nan
    idx = np.isfinite(recov_temp)

    if np.isnan(recov_temp[-1]):
        idx[-1] = True
        recov_temp[-1] = np.mean(t)

    fit_heights = x[1:]
    eTempSamp, dTempSamp, cTempSamp, bTempSamp, aTempSamp = np.polyfit(fit_heights[idx], recov_temp[idx], 4)


    def temp(a, b, c, d, e, x):
        return e * x ** 4 + d * x ** 3 + c * x ** 2 + b * x + a

    tempmask = np.zeros((len(del_height)+1, 1))
    tempmask[0:-1] = recov_temp
    tempmask[-1] = recov_temp[-1]
    return temp(aTempSamp,bTempSamp,cTempSamp, dTempSamp, eTempSamp,x), tempmask


def MinLogMargPost(params):#, coeff):
    tol = 1e-8
    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    betaG = 1e-4
    betaD = 1e-10
    A = getA()
    SpecNumMeas, SpecNumLayers = np.shape(A)

    L = getPrec()
    n = SpecNumLayers
    m = SpecNumMeas
    ATA = np.matmul(A.T,A)
    Bp = ATA + lamb * L

    y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
    ATy = np.matmul(A.T, y)
    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[:,0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

def getA():
    return np.loadtxt("AMat.txt")

def getPrec():
    return np.loadtxt('GraphLaplacian.txt', skiprows = 1, delimiter= '\t')

def composeAforPress(A_lin, temp, O3, ind):
    files = '634f1dc4.par'  # /home/lennartgolks/Python /Users/lennart/PycharmProjects

    my_data = pd.read_csv(files, header=None)
    data_set = my_data.values

    size = data_set.shape
    wvnmbr = np.zeros((size[0], 1))
    S = np.zeros((size[0], 1))
    F = np.zeros((size[0], 1))
    g_air = np.zeros((size[0], 1))
    g_self = np.zeros((size[0], 1))
    E = np.zeros((size[0], 1))
    n_air = np.zeros((size[0], 1))
    g_doub_prime = np.zeros((size[0], 1))

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    w_cross = f_broad * 1e-4 * O3
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    # 1e2 for pressure values from hPa to Pa
    A_scal = 1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers, 1)) * scalingConst * S[ind, 0] * num_mole / temp

    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, 1

def composeAforO3(A_lin, temp, press, ind):

    files = '634f1dc4.par'  # /home/lennartgolks/Python /Users/lennart/PycharmProjects

    my_data = pd.read_csv(files, header=None)
    data_set = my_data.values

    size = data_set.shape
    wvnmbr = np.zeros((size[0], 1))
    S = np.zeros((size[0], 1))
    F = np.zeros((size[0], 1))
    g_air = np.zeros((size[0], 1))
    g_self = np.zeros((size[0], 1))
    E = np.zeros((size[0], 1))
    n_air = np.zeros((size[0], 1))
    g_doub_prime = np.zeros((size[0], 1))

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))

    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # 1e2 for pressure values from hPa to Pa
    A_scal = press.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm / (temp)
    theta_scale = num_mole *  f_broad * 1e-4 * scalingConst * S[ind, 0]
    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, theta_scale


def composeAforTemp(A_lin, press, O3, ind, old_temp):
    files = '634f1dc4.par'  # /home/lennartgolks/Python /Users/lennart/PycharmProjects

    my_data = pd.read_csv(files, header=None)
    data_set = my_data.values

    size = data_set.shape
    wvnmbr = np.zeros((size[0], 1))
    S = np.zeros((size[0], 1))
    F = np.zeros((size[0], 1))
    g_air = np.zeros((size[0], 1))
    g_self = np.zeros((size[0], 1))
    E = np.zeros((size[0], 1))
    n_air = np.zeros((size[0], 1))
    g_doub_prime = np.zeros((size[0], 1))

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    w_cross = f_broad * 1e-4 * O3
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / old_temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / old_temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / old_temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * old_temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    # 1e2 for pressure values from hPa to Pa
    A_scal = 1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers, 1)) * scalingConst * S[ind, 0] * num_mole * press.reshape((SpecNumLayers, 1))

    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, 1

def temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0):
    a = np.ones(x.shape)
    b = np.ones(x.shape)
    a[x < h0] = a0
    a[h0 <= x] = 0
    a[h1 <= x] = a1
    a[h2 <= x] = a2
    a[h3 <= x] = 0
    a[h4 <= x ] = a3
    a[h5 <= x ] = a4
    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + h0 * a0
    b[h2 <= x] = a1 * (h2-h1) + b0 + h0 * a0
    b[h3 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h4 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h5 <= x ] = a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    return a * (x - h) + b

def tWalkTemp(x, A, y, TempWalkSampNum, TempBurnIn, gamma, SpecNumLayers, h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0):
    def log_post_temp(Params):
        n = SpecNumLayers
        h0 = Params[0]
        h1 = Params[1]
        h2 = Params[2]
        h3 = Params[3]
        h4 = Params[4]
        h5 = Params[5]
        a0 = Params[6]
        a1 = Params[7]
        a2 = Params[8]
        a3 = Params[9]
        a4 = Params[10]
        b0 = Params[11]

        h0Mean = 11
        h0Sigm = 0.5

        h1Mean = 20
        h1Sigm = 3

        h2Mean = 32
        h2Sigm = 1

        h3Mean = 47
        h3Sigm = 2

        h4Mean = 51
        h4Sigm = 2

        h5Mean = 71
        h5Sigm = 2

        a0Mean = -6.5
        a0Sigm = 0.01

        a1Mean = 1
        a1Sigm = 0.01

        a2Mean = 2.8
        a2Sigm = 0.1

        a3Mean = -2.8
        a3Sigm = 0.01

        a4Mean = -2
        a4Sigm = 0.01

        b0Mean = 288.15
        b0Sigm = 2

        return gamma * np.sum((y - A @ (1 / temp_func(x, *Params).reshape((n, 1)))) ** 2) + (
                    (h0 - h0Mean) / h0Sigm) ** 2 + ((h1 - h1Mean) / h1Sigm) ** 2 + ((h2 - h2Mean) / h2Sigm) ** 2 + (
                           (h3 - h3Mean) / h3Sigm) ** 2 + ((h4 - h4Mean) / h4Sigm) ** 2 + (
                           (h5 - h5Mean) / h5Sigm) ** 2 + ((a0 - a0Mean) / a0Sigm) ** 2 + (
                           (a1 - a1Mean) / a1Sigm) ** 2 + ((a2 - a2Mean) / a2Sigm) ** 2 \
               + ((a3 - a3Mean) / a3Sigm) ** 2 + ((a4 - a4Mean) / a4Sigm) ** 2 + ((b0 - b0Mean) / b0Sigm) ** 2

    def MargPostSupp_temp(Params):
        list = []
        return all(list)

    MargPost = pytwalk.pytwalk(n=12, U=log_post_temp, Supp=MargPostSupp_temp)
    x0 = np.array([h0, h1, h2, h3, h4, h5, a0, a1, a2, a3, a4, b0])
    xp0 = 1.01 * x0

    MargPost.Run(T=TempWalkSampNum + TempBurnIn, x0=x0, xp0=xp0)

    return MargPost.Output