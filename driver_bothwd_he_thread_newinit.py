from astropy.io import ascii
from astropy import units
import numpy
import corner
import emcee
import pylab as plt
import numpy as np
import numpy.ma as ma
from numpy import sum
from matplotlib import pyplot
import seaborn
import multiprocessing as multi
#Nthreads = multi.cpu_count() - 2
Nthreads = 5

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator 
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
from scipy.optimize import leastsq
import sys

Rsun = units.Rsun.to(units.cm) #6.957e10 #cm
Msun = units.Msun.to(units.g)  #1.9891e+33 #g
parsec = units.pc.to(units.cm) #3.086e18 #cm
Grav = 6.6743e-8 #cgs

#these are various contols for the MCMC
Nemcee = 700
Nthin = 1
Nburn = 50
Nwalkers = 300
bndsprior = True
#I also set bndsprior = True in MCMCfit.py. 

print "Nemcee = ", Nemcee
print "Nwalkers = ", Nwalkers
print "Nthin = ", Nthin
print "Nburn = ", Nburn
if (Nthreads > Nwalkers):
    print "   WARNING: you set Nthreads > Nwalkers.  This is probably not wise.  I will set Nthreads = Nwalkers"
    Nthreads = Nwalkers
print "Nthreads = ", Nthreads


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
#t_He, logg_He, Mo_He, Ro_He = np.loadtxt('mygrid.txt', unpack=True)
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
dataw2 = ma.masked_greater(w2_5, 1533.)
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

#send the data
vars1 = data['x1'] #x-axis
vals1 = data['y1'] #y-axis
evals1 = data['y1e'] #error on the y axis

vars2 = data['x2'] #x-axis
vals2 = data['y2'] #y-axis
evals2 = data['y2e'] #error on the y axis

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
func = wdparams2
#make guesses at parameters : 
#NOTE: these are now used as Gaussian priors
#log g1, T1, logg2, T2, dist
params = (7.5, 15600., 7.6, 16600., dobs) 
eparams = (1.0, 1000., 1.0, 1000., dobserr) 
#distance fit in units of pc 
#give the parameter names for plotting (can use latex symbols)
params_name = [r'log$_{10}$(g$_1$)',r'Temp$_1$(K)',r'log$_{10}$(g$_2$)',r'Temp$_2$(K)',r'dist(pc)']
#provide bounds on the parameters used to draw the initial guesses and walkers and possibly to limit the priors
bnds = ( (6., 9.), (14000., 20000.), (6., 9.), (14000., 20000.), (distmin, distmax))



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
yval_mi1, yval_mi2 = func((vars1, vars2), params)

f2 = plt.figure(figsize=(5,7))

ax1 = plt.subplot(211)
ax1.set_xlabel('wavelength')
ax1.set_ylabel('flux_1')
ax1.errorbar(vars1, vals1, yerr=evals1, fmt='r.')
#ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposy='clip')
ax1.plot(vars1,yval_mi1,'b')

ax2 = plt.subplot(212)
ax2.set_xlabel('wavelength')
ax2.set_ylabel('flux_2')
ax2.errorbar(vars2, vals2, yerr=evals2, fmt='r.')
#ax2.set_xscale("log", nonposx='clip')
ax2.set_yscale("log", nonposy='clip')
ax2.plot(vars2, yval_mi2,'b')

plt.subplots_adjust(hspace = 0.4, top = 0.95, bottom = 0.07, left = 0.15, right = 0.95)

f2.savefig('initial_guess_fit_wd.png')


####################################################################################################
#to enable threading in emcee, we need the MCMCfit.py methods to be here, instead of having this all in a class
#it's less elegant, but this works.  (And apparently this is because emcee must "pickle" the data and can't do that inside a class)
#parameters for fit
#emcee parameters
doMCMC = True
pr_lo = 16.
pr_mi = 50.
pr_hi = 84.
#for plots
doplots = True
f1name = "walkers_wd2.png"
f2name = "fit_MCMC_wd2.png"
f3name = "chains_wd2.png"
f4name = "posterior_wd2.png"
#f5name = "fit_leastsq_wd2.png"
#output 
results = []
print_posterior = True
posterior_file = "posterior_wd2.txt"


#flatten a tuple of tuples into a list
#also works for lists
def flatten(tupoftup):
    return [element for foo in tupoftup for element in foo]

#normal chi^2
def chi2(ps, args):
    x,obs,sigma,func = args
    ob = flatten(obs)
    si = flatten(sigma)
    fu = flatten(func(x,ps))
    OC = [o - f for (o,f) in zip(ob,fu)]
    return np.sum([o**2./s**2. for (o,s) in zip(OC,si)])
    
#reduced chi^2
def chi2_red(ps, args):
    x,obs,sigma,func = args
    ob = flatten(obs)
    si = flatten(sigma)
    fu = flatten(func(x,ps))
    OC = [o - f for (o,f) in zip(ob,fu)]
    dof = len(ob) - len(ps)
    return np.sum([o**2./s**2. for (o,s) in zip(OC,si)])/dof

#chi^2 as a list for the least squares fit
def chi2list(ps, args):
    x,obs,sigma,func = args
    ob = flatten(obs)
    si = flatten(sigma)
    fu = flatten(func(x,ps))
    OC = [o - f for (o,f) in zip(ob,fu)]
    return [o**2./s**2. for (o,s) in zip(OC,si)]
    
#draw initial walkers from the covariance matrix given by the least squares fit
def getwalkers(fit_results, walkers, n=1000):
    best,cov=fit_results[:2]
    pInits = np.random.multivariate_normal(best,cov*n,Nwalkers)
    for j,p in enumerate(pInits):
        for i,a in enumerate(p):
            if (a > bnds[i][1] or a < bnds[i][0]):
                pInits[j][i] = walkers[j][i]
    return pInits

#draw random initial guesses for the least-squares fitting
def drawguesses():
    initPosT = []
# I found it easier to select the parameters first this way...
    for i,p in enumerate(params):
        initPosT.append([x*(bnds[i][1]-bnds[i][0]) + bnds[i][0] for x in np.random.random(Nwalkers)])
#...and then transpose this matrix so that I can get an array of guesses
    initPos = np.matrix.transpose(np.array(initPosT))
    return initPos

    
#for emcee...
#likelihood        
def lnlike(ps, args):
    return -0.5*chi2(ps,args)


def gauss(x,m,s):
#        return 1./(s*(2.*np.pi)**2.)*np.exp(-1.*(x - m)**2./(2.*s**2.))
    return np.exp(-1.*(x - m)**2./(2.*s**2.)) #don't want the normalization because we want a peak at 1
def lnprior(ps):
    if (bndsprior):
        lp = 0. #ln(1)
        for i,p in enumerate(ps):
            if (p < bnds[i][0] or p > bnds[i][1]):
#doing this doesn't allow some walkers to advance.  
                    return -np.inf #ln(0)
#it works better if we have some very small number
#                    return np.log(1.e-3)
#though probably the most realistic thing would be to have some Gaussian priors
#we'd want to create inputs in this class and set them in the driver, but just an example here
#                mn = (bnds[i][0] + bnds[i][1])/2.
#                si = abs(bnds[i][0] - bnds[i][1])/5. #5 is arbitrary
#only apply a Gaussian prior to the distance (the rest are just flat within the bounds)
            if (i == 4):
                mn = params[i]
                si = eparams[i]
                g = gauss(p, mn, si)
                if (g > 0): #safety check
                    lp += np.log(g)
                else:
                    return -np.inf

    return 0. #ln(1)

#priors 
#this assume flat priors  (can make this more informed, given whatever prior information we think we know)
#    def lnprior(ps):
#        if (bndsprior):
#            for i,p in enumerate(ps):
#                if (p < bnds[i][0] or p > bnds[0][1]):
#                    return -np.inf #ln(0)
#        return 0. #ln(1)

#now construct the probability as priors*likelihood (but again in the log)
def lnprob(ps, args):
    lp = lnprior(ps)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(ps, args)

#plot the chains
def plotchains(sampler):
    subn = len(params)*100+11
    for i,p in enumerate(params):
        for w in range(sampler.chain.shape[0]):
            ax = pyplot.subplot(subn+i)
            ax.plot(sampler.chain[w,:,i],color='black', linewidth=2, alpha=0.2)
            if (len(params_name) >= i):
                ax.set_ylabel(params_name[i])
        ax.plot([Nburn,Nburn],[min(flatten(sampler.chain[:,:,i])),max(flatten(sampler.chain[:,:,i]))],'r--', linewidth=2)

#need another method to plot the fitted surface

#this is the main method that runs the fitting routine
def dofit():

#used for the fitting
    vars = (vars1, vars2)
    vals = (vals1, vals2)
    evals = (evals1, evals2)
    args = (vars,  vals, evals, func)

    if (doMCMC):

#OK now choose the walkers for the MCMC
        p0 = np.zeros([Nwalkers,len(params)])
        for i in xrange(0, Nwalkers):
            for j in xrange(0, len(params)):
                #p0[i][j] = params[j]*np.random.uniform(0.95,1.05,1)
                p0[i][j] = np.random.normal(loc=params[j],scale=params[j]*0.05,size=1)
        #inbounds = False
        #if (inbounds):
        #    walkers = getwalkers(fres, inits)
        #else:
        #    walkers = inits
        walkers = p0
        if (doplots):
            f1 = corner.corner(walkers, labels = params_name, truths = params, range = [(0.999*min(walkers[:,i]),1.001*max(walkers[:,i])) for i in range(len(params))])
            f1.savefig(f1name, dpi=300)


#now run emcee
        print "Running emcee ..."
        sampler = emcee.EnsembleSampler(Nwalkers, len(params), lnprob,  args = (args,), threads = Nthreads)
        sampler.reset()
        sampler.run_mcmc(walkers, Nemcee, thin = Nthin)

#remove the burn-in and reshape the samples
        samples = sampler.chain[:, Nburn:, :].reshape((-1, len(params)))

#print the results
        pout = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [pr_lo, pr_mi, pr_hi], axis=0)))
        print "Acceptance fractions:"
        print(sampler.acceptance_fraction)
        print "Median and mean acceptance fraction:"
        print(np.median(sampler.acceptance_fraction))
        print(np.mean(sampler.acceptance_fraction))
        print "Estimated autocorrelation time:"
        print(sampler.acor)
        for i,p in enumerate(pout):
            name = " "
            if (len(params_name) >= i):
                name = params_name[i]
            print '%-10s : %10f  +%10f  -%10f' % ((name,)+tuple(p))
            results.append(tuple(p))

        if (print_posterior):
            ofile = open(posterior_file, 'w')
            for s in samples:
                for p in s:
                    ofile.write("%15.7e " % p,)
                ofile.write("\n")
            ofile.close()


#plot the results
        if (doplots):

#fit 
#This one is not general
#set up shared axes and different plot sizes (not sure if this is the best way, but I think it works OK here)
            gs = pyplot.GridSpec(7, 1, hspace=0, height_ratios = [2, 1, 0.5, 2, 1, 0.5, 2] )

            f2 = pyplot.figure(figsize=(5,12))
 
            yval_mi1, yval_mi2 = func(vars, [pout[x][0] for x in range(len(pout))])
            yval_lo1, yval_lo2 = func(vars, [pout[x][0] - pout[x][1] for x in range(len(pout))])
            yval_hi1, yval_hi2 = func(vars, [pout[x][0] + pout[x][2] for x in range(len(pout))])

#                ax1 = pyplot.subplot(511)
            ax1 = f2.add_subplot(gs[0,:])
            ax1.set_ylabel('flux$_1$')
            ax1.errorbar(vars1, vals1, yerr=evals1, fmt='r.')
            ax1.set_yscale("log", nonposy='clip')
            ax1.plot(vars1, yval_mi1,'b')
            ax1.plot(vars1, yval_lo1,'b--')
            ax1.plot(vars1, yval_hi1,'b--')  
            ax1.set_xlim((1100,1700))

#                ax2 = pyplot.subplot(512)
            ax2 = f2.add_subplot(gs[1,:], sharex=ax1)
            ax2.set_xlabel('Wavelength')
            ax2.set_ylabel('Residuals')
            resid = [ (y - yf) / ye for (y, yf, ye) in zip(vals1, yval_mi1, evals1)]
            pyplot.plot(vars1, resid,'o',linestyle="None")
            ax2.set_ylim((-5, 5))
            pyplot.setp(ax1.get_xticklabels(), visible=False)
  
#                ax3 = pyplot.subplot(513)
            ax3 = f2.add_subplot(gs[3,:])
            ax3.set_ylabel('flux$_2$')
            ax3.errorbar(vars2, vals2, yerr=evals2, fmt='r.')
            ax3.set_yscale("log", nonposy='clip')
            ax3.plot(vars2, yval_mi2,'b')
            ax3.plot(vars2, yval_lo2,'b--')
            ax3.plot(vars2, yval_hi2,'b--')  
            ax3.set_xlim((1100,1700))

#                ax4 = pyplot.subplot(514)
            ax4 = f2.add_subplot(gs[4,:], sharex=ax3)
            ax4.set_xlabel('Wavelength')
            ax4.set_ylabel('Residuals')
            resid = [ (y - yf) / ye for (y, yf, ye) in zip(vals2, yval_mi2, evals2)]
            pyplot.plot(vars2, resid,'.',linestyle="None")
            ax4.set_ylim((-5, 5))
            pyplot.setp(ax3.get_xticklabels(), visible=False)
            ax5 = pyplot.subplot(515)
            ax5 = f2.add_subplot(gs[6,:])
            ax5.set_xlabel(r'$\chi^2_{\rm red}$')
            ax5.set_ylabel('N')
            chi2a = [chi2_red(ps, args) for ps in samples]
            n, bins, patches = ax5.hist(chi2a, 50)
                
            pyplot.subplots_adjust(hspace = 0.4, top = 0.99, bottom = 0.07, left = 0.15, right = 0.95)
            f2.savefig(f2name, dpi=300)
                

#chains
            f3 = pyplot.figure(figsize=(5,10))
            plotchains(sampler)
            pyplot.subplots_adjust(hspace = 0.4, top = 0.98, bottom = 0.03, left = 0.2, right = 0.95)
            f3.savefig(f3name, dpi=300)
                
#posterior
            f4 = corner.corner(samples, labels = params_name, quantiles=[0.16, 0.5, 0.84])
            f4.savefig(f4name, dpi=300)

    return results

####################################################################################################



if __name__=="__main__":

#now print the results to the terminal
    results = dofit()

    print results

    logg1 = results[0][0]
    logg1err_plus = results[0][1]
    logg1err_minus = results[0][2]

    temp1 = results[1][0]
    temp1err_plus = results[1][1]
    temp1err_minus = results[1][2]
    
    logg2 = results[2][0]
    logg2err_plus = results[2][1]
    logg2err_minus = results[2][2]
    
    temp2 = results[3][0]
    temp2err_plus = results[3][1]
    temp2err_minus = results[3][2]
    
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
    
