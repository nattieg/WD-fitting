from astropy.io import ascii
from astropy import units
import numpy
import pylab as plt
import numpy as np
import numpy.ma as ma
from numpy import sum
from matplotlib import pyplot
import seaborn as sns
sns.set_style("white")

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
import sys

#matplotlib.rcParams.update({'font.size': 20})

import warnings
warnings.filterwarnings("ignore")

Rsun = units.Rsun.to(units.cm) #6.957e10 #cm
Msun = units.Msun.to(units.g)  #1.9891e+33 #g
parsec = units.pc.to(units.cm) #3.086e18 #cm
Grav = 6.6743e-8 #cgs

##Gaia DR2 distance from Christian Knigge:
#dobs = 1942.0 #pc
#dobserr = 30.0

#Gaia distance from Phill Cargile
dobs = 1955.0 #pc
pradius = 11.2 #pc - scale radius for the Plummer sphere

'''
def wdparams(wave, ps, ifunc):
    modg, modT = ps
    wdmod = np.array([modg,modT])
    #findM = griddata((temp, logg), Mo, (modT, modg), method='linear', fill_value=1e8)
    #modrad = np.sqrt( (Grav * findM * Msun ) / 10**modg )
    realdist = dist * parsec
    if (ifunc==interpfunc1):
        findR_He = interpRo_He(np.array([modg,modT]))[0]
        modrad = findR_He * Rsun
        if (findR_He==0.):
            print("WE HAVE A PROBLEM")
    elif (ifunc==interpfunc2):
        findR = interpRo(np.array([modg,modT]))[0]
        modrad = findR * Rsun
    fitflux = ifunc(wdmod)[0] * np.pi * (modrad**2 / realdist**2)
    return fitflux
'''

def wdparams(wave, ps, ifunc):
    modg, modT = ps
    wdmod = np.array([modg,modT])
    #dist = np.random.normal(dobs, dobserr)
    #findM = griddata((temp, logg), Mo, (modT, modg), method='linear', fill_value=1e8)
    #modrad = np.sqrt( (Grav * findM * Msun ) / 10**modg )
    #Sample Plummer sphere to randomly assign distance:
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(np.random.uniform(-1,1))
    r = pradius / np.sqrt(np.random.uniform(0,1)**(-2.0/3.0) - 1)
    x = (r * np.sin(theta) * np.cos(phi)) + dobs
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    dist = np.sqrt (x**2.0 + y**2.0 + z**2.0)
    realdist = dist * parsec
    if (ifunc==interpfunc1):
        logmodT = np.log10(modT)
        #findM_He = interpMo_He(np.array([modg,modT]))[0]
        #if (findM_He>=0.44):
            #findR_He = interpRo(np.array([modg,modT]))[0]
        #elif (findM_He<0.44):
        findR_He = interpRo_He(np.array([modg,logmodT]))[0]
        modrad = findR_He * Rsun
        norm = (modrad**2 / realdist**2)
        if (findR_He==0.):
            print("WE HAVE A PROBLEM")

    elif (ifunc==interpfunc2):
        findM = interpMo(np.array([modg,modT]))[0]
        findR = interpRo(np.array([modg,modT]))[0]
        modrad = findR * Rsun
        norm = (modrad**2 / realdist**2)

        #prevent CO fits below the CO mass limit

        #logmodT = np.log10(modT)
        #findM_He = interpMo_He(np.array([modg,modT]))[0]
        #if (findM_He>=0.44):
            #findR_He = interpRo(np.array([modg,modT]))[0]
        #elif (findM_He<0.44):
        #findR_He = interpRo_He(np.array([modg,logmodT]))[0]
        #modrad = findR_He * Rsun
        #if (findR_He==0.):
        #    print "WE HAVE A PROBLEM"
    fitflux = ifunc(wdmod)[0] * np.pi * norm
    return fitflux

#NOTE: interpfunc1 and interpfunc2 are defined below
#this is our new function that will do the interpolation on each WD separately, but using the same distance

def wdparams2(waves, ps):
    modg1, modT1, modg2, modT2 = ps
    wave1, wave2 = waves
    fitflux1 = wdparams(wave1, (modg1, modT1), interpfunc1)
    fitflux2 = wdparams(wave2, (modg2, modT2), interpfunc2)
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

for i in range(0, len(glist)):
	for j in range(0, len(tlist)):
		tableMo[i][j] = Mo[i*(len(tlist))+j]
		tableRo[i][j] = Ro[i*(len(tlist))+j]

interpMo = RegularGridInterpolator((glist,tlist), tableMo, bounds_error=False, fill_value=None)
interpRo = RegularGridInterpolator((glist,tlist), tableRo, bounds_error=False, fill_value=None)

#read in table of He-core WD values:
t_He, logg_He, Mo_He, Ro_He, age = np.loadtxt('../Natalie/mygrid.txt', unpack=True)
#t_He, logg_He, Mo_He, Ro_He = np.loadtxt('mygrid.txt', unpack=True)
temp_He = 10.0**(t_He)

tlist_He = temp_He[0:299]
glist_He = logg_He[0::300]

tableMo_He = np.zeros([len(glist_He), len(tlist_He)])
tableRo_He = np.zeros([len(glist_He), len(tlist_He)])

for i in range(0, len(glist_He)):
	for j in range(0, len(tlist_He)):
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
#dataw2 = ma.masked_greater(w2_5, 1533.)
dataw2 = ma.masked_greater(w2_5, 1600.)
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

glist = ["600","625","650","675","700","725","750","775","800","825",
         "850","875","900"]

garray100 = np.array((glist), dtype=np.float)
garray = [g/100.0 for g in garray100]
Tlist = ["11000","12000","13000","14000","16000","18000","20000",
         "22000","24000","26000","28000","30000","35000"]
Tarray = np.array((Tlist), dtype=np.float)

#read in one model to initialize array sizes
samplewave1, sampleflux, sampleerr = np.loadtxt("../Natalie/da11000_600.dk.rebin1.lsf.bin30", unpack=True)
modmask1 = ma.masked_greater(samplewave1, 1800.)
modwave1 = 1.0*modmask1.compressed()
samplewave2, sampleflux, sampleerr = np.loadtxt("../Natalie/da11000_600.dk.rebin2.lsf.bin30", unpack=True)
modmask2 = ma.masked_greater(samplewave2, 1800.)
modwave2 = 1.0*modmask2.compressed()

modinfo = np.array((glist, Tlist))
modflux1 = np.zeros((len(glist), len(Tlist), len(modwave1)))
modflux2 = np.zeros((len(glist), len(Tlist), len(modwave2)))

#read in the models for BS1
for i in range(0, len(glist)):
	for j in range(0, len(Tlist)):
		file = "../Natalie/da" + str(Tlist[j]) +"_" + \
                       str(glist[i]) +".dk.rebin1.lsf.bin30"
		modw1, modf1, moderr1 = np.loadtxt(file, unpack=True)
		dummy = ma.masked_array(modf1, mask=modmask1.mask)
		modflux1[i][j] = 1.0e-8*dummy.compressed()
                #the model fluxes are per cm. the above
                #line changes this to per A.

interpfunc1 = RegularGridInterpolator((garray,Tarray), modflux1,
                                     bounds_error=False, fill_value=None)

#read in the models for BS2
for i in range(0, len(glist)):
	for j in range(0, len(Tlist)):
		file = "../Natalie/da" + str(Tlist[j]) +"_" + \
                       str(glist[i]) +".dk.rebin2.lsf.bin30"
		modw2, modf2, moderr2 = np.loadtxt(file, unpack=True)
		dummy = ma.masked_array(modf2, mask=modmask2.mask)
		modflux2[i][j] = 1.0e-8*dummy.compressed()
                #the model fluxes are per cm. the above
                #line changes this to per A.

interpfunc2 = RegularGridInterpolator((garray,Tarray), modflux2,
                                     bounds_error=False, fill_value=None)

print("Reading in posterior values...")
p1, p2, p3, p4 = np.loadtxt('posterior_setdist_plummer.txt', unpack=True)

#print "logg1, .10:"
#print(np.nanpercentile(p1, 10, axis=0))
#print "logg1, .90:"
#print(np.nanpercentile(p1, 90, axis=0))
#doall = True

#if not doall:
nsize = 500
print("Plotting using {} samples of posterior".format(nsize))
index = np.random.randint(0, len(p1), size=1)
#print(index)
#temp1 = t1
#temp2 = t2
#R1vals = np.zeros(len(index))
#M1vals = np.zeros(len(index))
#R2vals = np.zeros(len(index))
#M2vals = np.zeros(len(index))

if (nsize > len(p1)):
	print("Whoops, trying to sample more values than exist!")
	print("Setting sample size to total size...")
	nsize = len(postvalues)


pyplot.rc('font', size=20)          # controls default text sizes
pyplot.rc('axes', titlesize=20)     # fontsize of the axes title
pyplot.rc('axes', labelsize=24)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=20)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=20)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=14)    # legend fontsize
pyplot.rc('figure', titlesize=24)  # fontsize of the figure title

fig1 = pyplot.figure(figsize=(12,7))
fig2 = pyplot.figure(figsize=(12,7))


f1name = "WD1spectrum_hst_setdist_plummer.pdf"
f2name = "WD2spectrum_hst_setdist_plummer.pdf"

linecolor = "#9ca8b5"
alphaval = 0.1
width = 2.0
xlimits1 = (1125,1750)
xlimits2 = (1125,1625)
#xlimits2 = (1125,1750)
ylimits = (0,2.5e-16)
modwidth = 1.25

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

ax1.errorbar(w1, f1, yerr=toterr1, fmt='.', color='#7A93BE', linewidth=0.5, markersize=4, capsize=0)
ax2.errorbar(w2, f2, yerr=toterr2, fmt='.', color='#7A93BE', linewidth=0.5, markersize=4, capsize=0)


for i in range(0, nsize):
	index = np.random.randint(0, len(p1), size=2)
	#print(index)
#	postflux1, postflux2 = wdparams2((modwave1,modwave2), (p1[np.int(index[0])],p2[np.int(index[1])],p3[np.int(index[2])],p4[np.int(index[3])],p5[np.int(index[4])]))
	postflux1, postflux2 = wdparams2((modwave1,modwave2), (p1[np.int(index[0])],p2[np.int(index[0])],p3[np.int(index[1])],p4[np.int(index[1])]))
	#print(postvalues[i])
	ax1.plot(modwave1, postflux1, alpha=alphaval, zorder=1, linewidth=width, color=linecolor)
	ax2.plot(modwave2, postflux2, alpha=alphaval, zorder=1, linewidth=width, color=linecolor)

ax1.errorbar(wave1, ydata1, yerr=sigma1, fmt='.', zorder=2, color="#3B5489", capsize=0)
ax2.errorbar(wave2, ydata2, yerr=sigma2, fmt='.', zorder=2, color='#3B5489', capsize=0)

logg1min, logg1, logg1max = np.percentile(p1, (16, 50, 84))
logg2min, logg2, logg2max = np.percentile(p3, (16, 50, 84))
t1min, t1, t1max = np.percentile(p2, (16, 50, 84))
t2min, t2, t2max = np.percentile(p4, (16, 50, 84))

#pout = [(7.473348507185858, 0.05717583720890396, 0.053703275192185984), (15522.355072927421, 250.1830047650801, 250.5042511204647), (7.806522800494475, 0.04012449615397795, 0.026563630932838755), (17280.010989355796, 178.49753383865755, 121.99026411373052)]
#yval_mi1, yval_mi2 = wdparams2((modwave1,modwave2), [pout[x][0] for x in range(len(pout))])
#yval_lo1, yval_lo2 = wdparams2((modwave1,modwave2), [pout[x][0] - pout[x][1] for x in range(len(pout))])
#yval_hi1, yval_hi2 = wdparams2((modwave1,modwave2), [pout[x][0] + pout[x][2] for x in range(len(pout))])
yval_mi1, yval_mi2 = wdparams2((modwave1,modwave2), (logg1, t1, logg2, t2))
yval_lo1, yval_lo2 = wdparams2((modwave1,modwave2), (logg1min, t1min, logg2min, t2min))
yval_hi1, yval_hi2 = wdparams2((modwave1,modwave2), (logg1max, t1max, logg2max, t2max))

#ax1.plot(modwave1, yval_mi1, color='#3a516c', linewidth=modwidth)
#ax1.plot(modwave1, yval_lo1,'--', color='#3a516c', linewidth=modwidth)
#ax1.plot(modwave1, yval_hi1,'--', color='#3a516c', linewidth=modwidth)

#ax2.plot(modwave2, yval_mi2, color='#3a516c', linewidth=modwidth)
#ax2.plot(modwave2, yval_lo2,'--', color='#3a516c', linewidth=modwidth)
#ax2.plot(modwave2, yval_hi2,'--', color='#3a516c', linewidth=modwidth)

ax1.set_xlim(xlimits1)
ax2.set_xlim(xlimits2)
ax1.set_ylim(ylimits)
ax2.set_ylim(ylimits)

ax1.set_xlabel(r'Wavelength ($\AA$)')
ax2.set_xlabel(r'Wavelength ($\AA$)')

ax1.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
ax2.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')

#ax1.set_yscale("log", nonposy='clip')
#ax2.set_yscale("log", nonposy='clip')
#ax1.set_ylim(1e-17,1e-15)
#ax2.set_ylim(1e-17,1e-15)

fig1.tight_layout()
fig2.tight_layout()

fig1.savefig(f1name, dpi=1000)
fig2.savefig(f2name, dpi=1000)


#post_logg1, post_t1, post_logg2, post_t2, post_dist = np.loadtxt('posterior_wd2.txt', unpack=True)
#print(np.nanpercentile(post_logg1, 16, axis=0))
#print(np.nanpercentile(post_logg1, 84, axis=0))
#print "Trying to find models for a random posterior selection:"
#post = [7.5803974e+00,   1.5899308e+04,   7.6925516e+00,   1.6637355e+04,   1.8577406e+03]
#postflux1, postflux2 = wdparams2((modwave1,modwave2), postvalues[10])

#print postflux1[100:150]
