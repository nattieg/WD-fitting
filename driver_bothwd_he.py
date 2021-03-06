from MCMCfit import fitfunc
from astropy.io import ascii
from astropy import units
import numpy
import corner
import pylab as plt
import numpy as np
import numpy.ma as ma
from numpy import sum
import seaborn

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator 
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
import sys

Rsun = units.Rsun.to(units.cm) #6.957e10 #cm
Msun = units.Msun.to(units.g)  #1.9891e+33 #g
parsec = units.pc.to(units.cm) #3.086e18 #cm
Grav = 6.6743e-8 #cgs

#Eclipsing binary distance (Meibom+2009)
dobs = 1770.0 #pc
dobserr = 75.0

# this function interpolates in the WD grid of log g and Teff, then normalizes 
# based on the radius and distance

#this function fits BS1 with a He-core grid and BS2 with a CO-core grid
#He-core mass/radius relationship comes from Althaus et al: http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_heliumcore.html
#CO-core mass/radius relationship comes from Bergeron: http://www.astro.umontreal.ca/~bergeron/CoolingModels/

def wdparams(wave, ps, ifunc):
    modg, modT, dist = ps
    wdmod = np.array([modg,modT])
    #findM = griddata((temp, logg), Mo, (modT, modg), method='linear', fill_value=1e8)
    #modrad = np.sqrt( (Grav * findM * Msun ) / 10**modg )
    realdist = dist * parsec
    if (ifunc==interpfunc1):
    	findR_He = interpRo_He(np.array([modg,modT]))[0]
    	modrad = findR_He * Rsun
    elif (ifunc==interpfunc2):
    	findR = interpRo(np.array([modg,modT]))[0]
    	modrad = findR * Rsun
    fitflux = ifunc(wdmod)[0] * np.pi * (modrad**2 / realdist**2) 
    return fitflux

#NOTE: interpfunc1 and interpfunc2 are defined below
#this is our new function that will do the interpolation on each WD separately, but using the same distance

def wdparams2(waves, ps):
    modg1, modT1, modg2, modT2, dist = ps
    wave1, wave2 = waves
    fitflux1 = wdparams(wave1, (modg1, modT1, dist), interpfunc1)
    fitflux2 = wdparams(wave2, (modg2, modT2, dist), interpfunc2)
    return fitflux1, fitflux2

#we're not really fitting radius, radius is defined by log g and temp, we're fitting for distance...

#read in table of CO-core WD values:
temp, logg, Mo, Ro = np.loadtxt('../Natalie/wdtable.txt', unpack=True)

#griddata can do this all in fewer lines, but it takes SO MUCH longer
#findM = griddata((temp,logg), Mo, (15000,7.5), method='linear')

#let's make some tables...
glist = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
tlist = temp[0:60]

#we'll only be looking up either M or R at a time. 
tableMo = np.zeros([len(glist), len(tlist)])
tableRo = np.zeros([len(glist), len(tlist)])

for i in xrange(0, len(glist)):
	for j in xrange(0, len(tlist)):
		tableMo[i][j] = Mo[i*(len(tlist))+j]
		tableRo[i][j] = Ro[i*(len(tlist))+j]

interpMo = RegularGridInterpolator((glist,tlist), tableMo, bounds_error=False, fill_value=None)
interpRo = RegularGridInterpolator((glist,tlist), tableRo, bounds_error=False, fill_value=None)

#read in table of He-core WD values:
t_He, logg_He, Mo_He, Ro_He, age = np.loadtxt('../Natalie/mygrid.txt', unpack=True)
temp_He = 10.0**(t_He)

tlist_He = temp_He[0:99]
glist_He = logg_He[0::100]

tableMo_He = np.zeros([len(glist_He), len(tlist_He)])
tableRo_He = np.zeros([len(glist_He), len(tlist_He)])

for i in xrange(0, len(glist_He)):
	for j in xrange(0, len(tlist_He)):
		tableMo_He[i][j] = Mo_He[i*(len(tlist_He))+j]
		tableRo_He[i][j] = Ro_He[i*(len(tlist_He))+j]

interpMo_He = RegularGridInterpolator((glist_He,tlist_He), tableMo_He, bounds_error=False, fill_value=None)
interpRo_He = RegularGridInterpolator((glist_He,tlist_He), tableRo_He, bounds_error=False, fill_value=None)

#read in the BSS data and ...

#CREATE WAVELENGTH MASK
#edges of masked regions adjusted by hand in TestMasks.ipynb to 
#block out the geocoronal lines. 

#w1, f1, err1 = np.loadtxt('lcb201010_x1dsum.txt.bin30.unred', unpack=True)
w1, f1, stdev1, err1 = np.loadtxt('../Natalie/lcb201010_x1dsum.txt.bin30.unred.newerrors', unpack=True)
add_disp = 9.1e-18
#adding intrinsic dispersion
toterr1 = np.sqrt(err1**2.0 + add_disp**2.0)
w1_1 = ma.masked_less(w1, 1141)
w1_2 = ma.masked_inside(w1_1, 1178., 1250.)
w1_3 = ma.masked_inside(w1_2, 1292., 1318.)
w1_4 = ma.masked_inside(w1_3, 1348., 1367.)
w1_5 = ma.masked_inside(w1_4, 1332., 1340.)
dataw1 = ma.masked_greater(w1_5, 1725.)
dataw1c = 1.0*dataw1.compressed()

#w2, f2, err2 = np.loadtxt('lcb202010_x1dsum.txt.bin30.unred', unpack=True)
w2, f2, stdev2, err2 = np.loadtxt('../Natalie/lcb202010_x1dsum.txt.bin30.unred.newerrors', unpack=True)
#adding intrinsic dispersion 
toterr2 = np.sqrt(err2**2.0 + add_disp**2.0)
w2_1 = ma.masked_less(w2, 1142)
w2_2 = ma.masked_inside(w2_1, 1182., 1250.)
w2_3 = ma.masked_inside(w2_2, 1290., 1318.)
w2_4 = ma.masked_inside(w2_3, 1348., 1368.)
w2_5 = ma.masked_inside(w2_4, 1330., 1340.)
dataw2 = ma.masked_greater(w2_5, 1518.)
dataw2c = 1.0*dataw2.compressed()

#We're not going to fit the BSS. Worried that a bad line list in the UV would
#mess everything up. So the mask cuts off at the red end to remove
#contribution from the BSS. BSS2 is brighter, so the cutoff is bluer.

#transfer mask to flux and error
dataf1 = ma.masked_array(f1, mask=dataw1.mask)
dataerr1 = ma.masked_array(toterr1, mask=dataw1.mask)
dataf1c = 1.0*dataf1.compressed()
dataerr1c = 1.0*dataerr1.compressed()

dataf2 = ma.masked_array(f2, mask=dataw2.mask)
dataerr2 = ma.masked_array(toterr2, mask=dataw2.mask)
dataf2c = 1.0*dataf2.compressed()
dataerr2c = 1.0*dataerr2.compressed()

wave1 = dataw1c
ydata1 = dataf1c
sigma1 = dataerr1c

wave2 = dataw2c
ydata2 = dataf2c
sigma2 = dataerr2c

#set up data dict
data = dict()

data['x1'] = wave1
data['y1'] = ydata1
data['y1e'] = sigma1

data['x2'] = wave2
data['y2'] = ydata2
data['y2e'] = sigma2

#initialize the class  
soln = fitfunc()

#send the data
soln.vars1 = data['x1'] #x-axis
soln.vals1 = data['y1'] #y-axis
soln.evals1 = data['y1e'] #error on the y axis

soln.vars2 = data['x2'] #x-axis
soln.vals2 = data['y2'] #y-axis
soln.evals2 = data['y2e'] #error on the y axis

#find the distance constraint
distcm = dobs #* parsec
#dobserr is 1sig
distcmerr = dobserr*3. #* parsec
distmin = (distcm - distcmerr) 
distmax = (distcm + distcmerr) 
print("This is the minimum distance:")
print(distmin)
print("This is the maximum distance:")
print(distmax)

#define the function
soln.func = wdparams2
#make guesses at parameters : 
#NOTE: these are now used as Gaussian priors
#log g1, T1, logg2, T2, dist
soln.params = (7.5, 16000., 7.5, 17000., dobs) 
soln.eparams = (1.0, 1000., 1.0, 1000., dobserr) 
#distance fit in units of pc 
#give the parameter names for plotting (can use latex symbols)
soln.params_name = [r'log$_{10}$(g$_1$)',r'Temp$_1$(K)',r'log$_{10}$(g$_2$)',r'Temp$_2$(K)',r'dist(pc)']
#provide bounds on the parameters used to draw the initial guesses and walkers and possibly to limit the priors
soln.bnds = ( (6., 9.), (14000., 20000.), (6., 9.), (14000., 20000.), (distmin, distmax))

#these are various contols for the MCMC
#these values may still need to be adjusted

soln.Nemcee = 10000
soln.Nthin = 10
soln.Nburn = 200
soln.Nwalkers = 30
soln.bndsprior = True
#I also set bndsprior = True in MCMCfit.py. 

#make lists of g and T values so we can read in the models.
glist = ["600","625","650","675","700","725","750","775","800","825",
         "850","875","900"]

garray100 = np.array((glist), dtype=np.float)
garray = [g/100.0 for g in garray100]
Tlist = ["11000","12000","13000","14000","16000","18000","20000",
         "22000","24000","26000","28000","30000","35000"]
Tarray = np.array((Tlist), dtype=np.float)

modinfo = np.array((glist, Tlist))
modflux1 = np.zeros((len(glist), len(Tlist), len(dataf1c)))
modflux2 = np.zeros((len(glist), len(Tlist), len(dataf2c)))

#read in the models for BS1
for i in xrange(0, len(glist)):
	for j in xrange(0, len(Tlist)):
		file = "../Natalie/da" + str(Tlist[j]) +"_" + \
                       str(glist[i]) +".dk.rebin1.lsf.bin30"
		modw1, modf1, moderr1 = np.loadtxt(file, unpack=True)
		dummy = ma.masked_array(modf1, mask=dataw1.mask) 
		modflux1[i][j] = 1.0e-8*dummy.compressed()
                #the model fluxes are per cm. the above
                #line changes this to per A.

interpfunc1 = RegularGridInterpolator((garray,Tarray), modflux1, 
                                     bounds_error=False, fill_value=None)

#read in the models for BS2
for i in xrange(0, len(glist)):
	for j in xrange(0, len(Tlist)):
		file = "../Natalie/da" + str(Tlist[j]) +"_" + \
                       str(glist[i]) +".dk.rebin2.lsf.bin30"
		modw2, modf2, moderr2 = np.loadtxt(file, unpack=True)
		dummy = ma.masked_array(modf2, mask=dataw2.mask) 
		modflux2[i][j] = 1.0e-8*dummy.compressed()
                #the model fluxes are per cm. the above
                #line changes this to per A.

interpfunc2 = RegularGridInterpolator((garray,Tarray), modflux2, 
                                     bounds_error=False, fill_value=None)

#dummy figure to see what the initial guess looks like
yval_mi1, yval_mi2 = soln.func((soln.vars1, soln.vars2), soln.params)

f2 = plt.figure(figsize=(5,7))

ax1 = plt.subplot(211)
ax1.set_xlabel('wavelength')
ax1.set_ylabel('flux_1')
ax1.errorbar(soln.vars1, soln.vals1, yerr=soln.evals1, fmt='r.')
#ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposy='clip')
ax1.plot(soln.vars1,yval_mi1,'b')

ax2 = plt.subplot(212)
ax2.set_xlabel('wavelength')
ax2.set_ylabel('flux_2')
ax2.errorbar(soln.vars2, soln.vals2, yerr=soln.evals2, fmt='r.')
#ax2.set_xscale("log", nonposx='clip')
ax2.set_yscale("log", nonposy='clip')
ax2.plot(soln.vars2, yval_mi2,'b')

plt.subplots_adjust(hspace = 0.4, top = 0.95, bottom = 0.07, left = 0.15, right = 0.95)

f2.savefig('initial_guess_fit_wd.png')

#now run the fit
soln.dofit()

#now print the results to the terminal
print soln.results

logg1 = soln.results[0][0]
logg1err_plus = soln.results[0][1]
logg1err_minus = soln.results[0][2]

temp1 = soln.results[1][0]
temp1err_plus = soln.results[1][1]
temp1err_minus = soln.results[1][2]

logg2 = soln.results[2][0]
logg2err_plus = soln.results[2][1]
logg2err_minus = soln.results[2][2]

temp2 = soln.results[3][0]
temp2err_plus = soln.results[3][1]
temp2err_minus = soln.results[3][2]

wd1values = np.array([logg1,temp1])
wd1plusval = np.array([( logg1 + logg1err_plus, temp1 + temp1err_plus)])
wd1minusval = np.array([( logg1 - logg1err_minus, temp1 - temp1err_minus)])

wd2values = np.array([logg2,temp2])
wd2plusval = np.array([( logg2 + logg2err_plus, temp2 + temp2err_plus)])
wd2minusval = np.array([( logg2 - logg2err_minus, temp2 - temp2err_minus)])

findR1 = interpRo_He(wd1values)[0]
findR1_plus = interpRo_He(wd1plusval)[0]
findR1_minus = interpRo_He(wd1minusval)[0]

findM1 = interpMo_He(wd1values)[0]
findM1_plus = interpMo_He(wd1plusval)[0]
findM1_minus = interpMo_He(wd1minusval)[0]

print "WD1 radius and errors"
print findR1
print (findR1 - findR1_plus)
print (findR1_minus - findR1)

print "WD1 mass and errors"
print findM1
print (findM1_plus - findM1)
print (findM1 - findM1_minus)

findR2 = interpRo(wd2values)[0]
findR2_plus = interpRo(wd2plusval)[0]
findR2_minus = interpRo(wd2minusval)[0]

findM2 = interpMo(wd2values)[0]
findM2_plus = interpMo(wd2plusval)[0]
findM2_minus = interpMo(wd2minusval)[0]

print "WD2 radius and errors"
print findR2
print (findR2 - findR2_plus)
print (findR2_minus - findR2)

print "WD2 mass and errors"
print findM2
print (findM2_plus - findM2)
print (findM2 - findM2_minus)
