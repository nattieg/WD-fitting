import numpy
from matplotlib import pyplot
import emcee
#import triangle
import corner
from scipy.optimize import leastsq
from scipy.misc import comb
#import multiprocessing as multi
#nthreads = multi.cpu_count()
#can't use multiprocessing with emcee while lnprob is method of a class!  (or maybe there's a different pool that i can use?)
nthreads = 1

def genfunc(x, ps):
#some function form for Pcoll(E,L,N)
    E,L,N = x
    alpha, beta = ps

#I will need to flatten this in some way, right?
    f = comb(N, 2.) * E**alpha * L**beta
    
    return f

#flatten a tuple of tuples into a list
#also works for lists
def flatten(tupoftup):
    return [element for foo in tupoftup for element in foo]

class fitfunc():

    def __init__(self):
#parameters for fit
        self.params = [] #[alpha, beta]
        self.eparams = [] #[alpha_error, beta_error]
        self.params_name = []
        self.bnds = () #bounds over which each parameter can be defined (as a tuple)
        self.bndsprior = True #force the MCMC to stay within the bounds?
        self.vars1 = ([])
        self.vals1 = []
        self.evals1 = []
        self.vars2 = ([])
        self.vals2 = []
        self.evals2 = []
        self.func = genfunc
#emcee parameters
        self.doMCMC = True
        self.Nwalkers = 50
        self.Nemcee = 10000
        self.Nthin = 10
        self.Nburn = 100
        self.pr_lo = 16.
        self.pr_mi = 50.
        self.pr_hi = 84.
#for plots
        self.doplots = True
        self.f1name = "walkers_wd.png"
        self.f2name = "fit_MCMC_wd.png"
        self.f3name = "chains_wd.png"
        self.f4name = "posterior_wd.png"
        self.f5name = "fit_leastsq_wd.png"
#output 
        self.results = []
        self.print_posterior = True
        self.posterior_file = "posterior_wd.txt"


#normal chi^2
    def chi2(self, ps, args):
        x,obs,sigma,func = args
        ob = flatten(obs)
        si = flatten(sigma)
        fu = flatten(func(x,ps))
        OC = [o - f for (o,f) in zip(ob,fu)]
        return numpy.sum([o**2./s**2. for (o,s) in zip(OC,si)])
    
#reduced chi^2
    def chi2_red(self, ps, args):
        x,obs,sigma,func = args
        ob = flatten(obs)
        si = flatten(sigma)
        fu = flatten(func(x,ps))
        OC = [o - f for (o,f) in zip(ob,fu)]
        dof = len(ob) - len(ps)
        return numpy.sum([o**2./s**2. for (o,s) in zip(OC,si)])/dof

#chi^2 as a list for the least squares fit
    def chi2list(self, ps, args):
        x,obs,sigma,func = args
        ob = flatten(obs)
        si = flatten(sigma)
        fu = flatten(func(x,ps))
        OC = [o - f for (o,f) in zip(ob,fu)]
        return [o**2./s**2. for (o,s) in zip(OC,si)]
    
#draw initial walkers from the covariance matrix given by the least squares fit
    def getwalkers(self, fit_results, walkers, n=1000):
        best,cov=fit_results[:2]
        pInits = numpy.random.multivariate_normal(best,cov*n,self.Nwalkers)
        for j,p in enumerate(pInits):
            for i,a in enumerate(p):
                if (a > self.bnds[i][1] or a < self.bnds[i][0]):
                    pInits[j][i] = walkers[j][i]
        return pInits

#draw random initial guesses for the least-squares fitting
    def drawguesses(self):
        initPosT = []
# I found it easier to select the parameters first this way...
        for i,p in enumerate(self.params):
            initPosT.append([x*(self.bnds[i][1]-self.bnds[i][0]) + self.bnds[i][0] for x in numpy.random.random(self.Nwalkers)])
#...and then transpose this matrix so that I can get an array of guesses
        initPos = numpy.matrix.transpose(numpy.array(initPosT))
        return initPos

    
#for emcee...
#likelihood        
    def lnlike(self, ps, args):
        return -0.5*self.chi2(ps,args)


    def gauss(self, x,m,s):
#        return 1./(s*(2.*numpy.pi)**2.)*numpy.exp(-1.*(x - m)**2./(2.*s**2.))
        return numpy.exp(-1.*(x - m)**2./(2.*s**2.)) #don't want the normalization because we want a peak at 1
    def lnprior(self, ps):
        if (self.bndsprior):
            lp = 0. #ln(1)
            for i,p in enumerate(ps):
                if (p < self.bnds[i][0] or p > self.bnds[i][1]):
#doing this doesn't allow some walkers to advance.  
                    return -numpy.inf #ln(0)
#it works better if we have some very small number
#                    return numpy.log(1.e-3)
#though probably the most realistic thing would be to have some Gaussian priors
#we'd want to create inputs in this class and set them in the driver, but just an example here
#                mn = (self.bnds[i][0] + self.bnds[i][1])/2.
#                si = abs(self.bnds[i][0] - self.bnds[i][1])/5. #5 is arbitrary
#only apply a Gaussian prior to the distance (the rest are just flat within the bounds)
                if (i == 4):
                    mn = self.params[i]
                    si = self.eparams[i]
                    g = self.gauss(p, mn, si)
                    if (g > 0): #safety check
                        lp += numpy.log(g)
                    else:
                        return -numpy.inf

        return 0. #ln(1)

#priors 
#this assume flat priors  (can make this more informed, given whatever prior information we think we know)
#    def lnprior(self, ps):
#        if (self.bndsprior):
#            for i,p in enumerate(ps):
#                if (p < self.bnds[i][0] or p > self.bnds[0][1]):
#                    return -numpy.inf #ln(0)
#        return 0. #ln(1)

#now construct the probability as priors*likelihood (but again in the log)
    def lnprob(self, ps, args):
        lp = self.lnprior(ps)
        if not numpy.isfinite(lp):
            return -numpy.inf
        return lp + self.lnlike(ps, args)

#plot the chains
    def plotchains(self, sampler):
        subn = len(self.params)*100+11
        for i,p in enumerate(self.params):
            for w in range(sampler.chain.shape[0]):
                ax = pyplot.subplot(subn+i)
                ax.plot(sampler.chain[w,:,i])
                if (len(self.params_name) >= i):
                    ax.set_ylabel(self.params_name[i])
            ax.plot([self.Nburn,self.Nburn],[min(flatten(sampler.chain[:,:,i])),max(flatten(sampler.chain[:,:,i]))],'k--', linewidth=2)

#need another method to plot the fitted surface

#this is the main method that runs the fitting routine
    def dofit(self):

#used for the fitting
        self.vars = (self.vars1, self.vars2)
        self.vals = (self.vals1, self.vals2)
        self.evals = (self.evals1, self.evals2)
        args = (self.vars,  self.vals, self.evals, self.func)


#first we do a least-squares fit to narrow in on the correct parameters (possibly unnecessary)
        print "Generating guesses of parameters and finding least-squares fit ..."
        inits = self.drawguesses()
        fits = {"chi2":[],"params":[],"chi2_bnds":[]}
        for i,w in enumerate(inits):
            fres = leastsq(self.chi2list, w, args=(args,), full_output=1, maxfev = 5000)

            havesolution = True
            if (fres[4] < 1 or fres[4] > 4):
                havesolution = False
#test print out results
            print 'guess',i,'[',
            for p0 in w: 
                print '%10f' % p0,
            print ']','[', 
            inbounds = True
            for i,p1 in enumerate(fres[0]): 
                print '%10f' % p1,
                if (p1 > self.bnds[i][1] or p1 < self.bnds[i][0]):
                    inbounds = False

            if (havesolution):
                print '] %10f' % self.chi2_red(fres[0],args)
                fits["chi2"].append(self.chi2(fres[0], args))
            else:
                print '] %10f' % 1.e20
                fits["chi2"].append(1.e20)

                
            if (inbounds and havesolution):
                fits["chi2_bnds"].append(self.chi2(fres[0], args))
            else:
                fits["chi2_bnds"].append(1.e20)

            fits["params"].append(fres[0])

        ibest = numpy.nanargmin(fits["chi2"])
#only select the best fit of these that fall within the bounds (if that exists)
#does this if statement really work?
        if (len(numpy.where(numpy.array(fits["chi2_bnds"]) < 1.e20)[0]) > 0):
            ibest = numpy.nanargmin(fits["chi2_bnds"])
            print ""
            print "found leastsq fit within bounds"
            inbounds = True
        else:
            ibest = numpy.nanargmin(fits["chi2"])
            print ""
            print "no leastsq fit within the bounds"
            inbounds = False



#now redo the best fit so that we can get the covariance matrix
#        pguess = fits["params"][ibest]
#        fres = leastsq(self.chi2list, pguess, args=(args,), full_output=1)
        fres = leastsq(self.chi2list, inits[ibest], args=(args,), full_output=1, maxfev = 5000)
        print ''
        print 'leastsq fit attempt [',
        for p0 in inits[ibest]: 
            print '%10f' % p0,
        print ']','[', 
        for p1 in fres[0]: 
            print '%10f' % p1,
        print '] %10f' % self.chi2_red(fres[0],args)

#get the errors from the covariance matrix, 
#taken from http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
        best, pcov = fres[:2]
        if (pcov.all() != None):
            pcov = pcov * self.chi2_red(fres[0],args)
            error = [] 
            for i in range(len(self.params)):
                try:
                    error.append( numpy.absolute(pcov[i][i])**0.5)
                except:
                    error.append( 0.00 )
        else:
            print ""
            print "WARNING: no covariance matrix for leastsq fit, so no uncertainties. Not using this!"
            print ""
            inbounds = False
            error = [0. for b in best]
#            print fres
#print the results from the leastsq fit
        for i,val in enumerate(best):
            name = " "
            if (len(self.params_name) >= i):
                name = self.params_name[i]
            print '%-10s : %10f  +/- %10f  ' % (name,val,error[i])
        print ''


        if (self.doplots):

            yval_mi1, yval_mi2 = self.func(self.vars, best)
#this may not really be the full range of values in the fit
            yval_lo1, yval_lo2 = self.func(self.vars, [x - xe for (x,xe) in zip(best,error)])
            yval_hi1, yval_hi2 = self.func(self.vars, [x + xe for (x,xe) in zip(best,error)])

            f5 = pyplot.figure(figsize=(5,7))
            ax1 = pyplot.subplot(211)
            ax1.set_xlabel('wavelength')
            ax1.set_ylabel('flux_1')
            ax1.errorbar(self.vars1, self.vals1, yerr=self.evals1, fmt='r.')
            ax1.set_yscale("log", nonposy='clip')
            ax1.plot(self.vars1, yval_mi1,'b')
            ax1.plot(self.vars1, yval_lo1,'b--')
            ax1.plot(self.vars1, yval_hi1,'b--')  

            ax2 = pyplot.subplot(212)
            ax2.set_xlabel('wavelength')
            ax2.set_ylabel('flux_2')
            ax2.errorbar(self.vars2, self.vals2, yerr=self.evals2, fmt='r.')
            ax2.set_yscale("log", nonposy='clip')
            ax2.plot(self.vars2, yval_mi2,'b')
            ax2.plot(self.vars2, yval_lo2,'b--')
            ax2.plot(self.vars2, yval_hi2,'b--')  

            pyplot.subplots_adjust(hspace = 0.4, top = 0.95, bottom = 0.07, left = 0.15, right = 0.95)

            f5.savefig(self.f5name)

        if (self.doMCMC):

#OK now choose the walkers for the MCMC
            inbounds = False
            if (inbounds):
                walkers = self.getwalkers(fres, inits)
            else:
                walkers = inits

            if (self.doplots):
                f1 = corner.corner(walkers, labels = self.params_name, truths = self.params, range = [(0.999*min(walkers[:,i]),1.001*max(walkers[:,i])) for i in range(len(self.params))])
                f1.savefig(self.f1name)


#now run emcee
            print "Running emcee ..."
            sampler = emcee.EnsembleSampler(self.Nwalkers, len(self.params), self.lnprob,  args = (args,), threads = nthreads)
            sampler.reset()
            sampler.run_mcmc(walkers, self.Nemcee, thin = self.Nthin)

#remove the burn-in and reshape the samples
            samples = sampler.chain[:, self.Nburn:, :].reshape((-1, len(self.params)))

#print the results
            pout = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*numpy.percentile(samples, [self.pr_lo, self.pr_mi, self.pr_hi], axis=0)))
            for i,p in enumerate(pout):
                name = " "
                if (len(self.params_name) >= i):
                    name = self.params_name[i]
                print '%-10s : %10f  +%10f  -%10f' % ((name,)+tuple(p))
                self.results.append(tuple(p))

            if (self.print_posterior):
                ofile = open(self.posterior_file, 'w')
                for s in samples:
                    for p in s:
                        ofile.write("%15.7e " % p,)
                    ofile.write("\n")
                ofile.close()

#plot the results
            if (self.doplots):

#fit 
#This one is not general
#set up shared axes and different plot sizes (not sure if this is the best way, but I think it works OK here)
                gs = pyplot.GridSpec(7, 1, hspace=0, height_ratios = [2, 1, 0.5, 2, 1, 0.5, 2] )

                f2 = pyplot.figure(figsize=(5,12))
 
                yval_mi1, yval_mi2 = self.func(self.vars, [pout[x][0] for x in range(len(pout))])
                yval_lo1, yval_lo2 = self.func(self.vars, [pout[x][0] - pout[x][1] for x in range(len(pout))])
                yval_hi1, yval_hi2 = self.func(self.vars, [pout[x][0] + pout[x][2] for x in range(len(pout))])

#                ax1 = pyplot.subplot(511)
                ax1 = f2.add_subplot(gs[0,:])
                ax1.set_ylabel('flux$_1$')
                ax1.errorbar(self.vars1, self.vals1, yerr=self.evals1, fmt='r.')
                ax1.set_yscale("log", nonposy='clip')
                ax1.plot(self.vars1, yval_mi1,'b')
                ax1.plot(self.vars1, yval_lo1,'b--')
                ax1.plot(self.vars1, yval_hi1,'b--')  
                ax1.set_xlim((1100,1700))

#                ax2 = pyplot.subplot(512)
                ax2 = f2.add_subplot(gs[1,:], sharex=ax1)
                ax2.set_xlabel('Wavelength')
                ax2.set_ylabel('Residuals')
                resid = [ (y - yf) / ye for (y, yf, ye) in zip(self.vals1, yval_mi1, self.evals1)]
                pyplot.plot(self.vars1, resid,'o',linestyle="None")
                ax2.set_ylim((-5, 5))
                pyplot.setp(ax1.get_xticklabels(), visible=False)
  
#                ax3 = pyplot.subplot(513)
                ax3 = f2.add_subplot(gs[3,:])
                ax3.set_ylabel('flux$_2$')
                ax3.errorbar(self.vars2, self.vals2, yerr=self.evals2, fmt='r.')
                ax3.set_yscale("log", nonposy='clip')
                ax3.plot(self.vars2, yval_mi2,'b')
                ax3.plot(self.vars2, yval_lo2,'b--')
                ax3.plot(self.vars2, yval_hi2,'b--')  
                ax3.set_xlim((1100,1700))

#                ax4 = pyplot.subplot(514)
                ax4 = f2.add_subplot(gs[4,:], sharex=ax3)
                ax4.set_xlabel('Wavelength')
                ax4.set_ylabel('Residuals')
                resid = [ (y - yf) / ye for (y, yf, ye) in zip(self.vals2, yval_mi2, self.evals2)]
                pyplot.plot(self.vars2, resid,'.',linestyle="None")
                ax4.set_ylim((-5, 5))
                pyplot.setp(ax3.get_xticklabels(), visible=False)

#                ax5 = pyplot.subplot(515)
                ax5 = f2.add_subplot(gs[6,:])
                ax5.set_xlabel(r'$\chi^2_{\rm red}$')
                ax5.set_ylabel('N')
                chi2a = [self.chi2_red(ps, args) for ps in samples]
                n, bins, patches = ax5.hist(chi2a, 50)
                
                pyplot.subplots_adjust(hspace = 0.4, top = 0.99, bottom = 0.07, left = 0.15, right = 0.95)
                f2.savefig(self.f2name)
                

#chains
                f3 = pyplot.figure(figsize=(5,10))
                self.plotchains(sampler)
                pyplot.subplots_adjust(hspace = 0.4, top = 0.98, bottom = 0.03, left = 0.2, right = 0.95)
                f3.savefig(self.f3name)
                
#posterior
                f4 = corner.corner(samples, labels = self.params_name, quantiles=[0.16, 0.5, 0.84])
                f4.savefig(self.f4name)
                

####################################################################################################
if __name__=="__main__":
    numpy.random.seed(seed = 1234567)



    def fPcoll(x, ps):
#some function form for Pcoll(E,L,N)
        E,L,N = x
        alpha, beta = ps

#I will need to flatten this in some way, right?
        f = comb(N, 2.) * E**alpha * L**beta
        
        return f

#as a test, generate some random data and try to fit it
    Nvals = 10.
    noise_lvl = 0.1

    alpha = 0.15
    beta = 0.57
    ps = (alpha, beta)

    E = []
    L = []
    N = []
    P = []
    eP = []
    for i,x, in enumerate(range(int(Nvals))):
        for j,y in enumerate(range(int(Nvals))):
            E.append((x+1.)/Nvals)
            L.append((y+1.)/Nvals)
            N.append(3.)
            nl = noise_lvl*numpy.random.randn()
            eP.append(nl)
            P.append( fPcoll( ((x+1.)/Nvals,(y+1.)/Nvals,3), ps) + nl )

#    f=pyplot.figure(figsize = (7,15))
#    pyplot.xlabel("E")
#    pyplot.ylabel("L")
#    pyplot.xlim([0,1])
#    pyplot.ylim([0,1])
#    cm = pyplot.cm.get_cmap("gist_rainbow")
#    pyplot.scatter(E, L, c=P, cmap=cm, edgecolors='none')
#    pyplot.show()


#initialize the class    
    soln = fitfunc()
    soln.params = ps
    soln.params_name = [r'$\alpha$',r'$\beta$']
    soln.vars = (E,L,N)
    soln.vals = P
    soln.evals = eP
    soln.bnds = ( (-1.,1.), (-1.,1.) )
    soln.func = fPcoll
#uncomment these to do a more accurate fit
    soln.Nemcee = 1000
    soln.Nthin = 1
    soln.burn = 0
    soln.Nwalkers = 10

#now run the fit
    soln.dofit()

