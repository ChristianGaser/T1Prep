import numpy as np
import math
import scipy.stats as stats
import multiprocessing as mp

def symGaussFiltFWHM (filtFWHM, filtbinwidth):
  sigma = filtFWHM / math.sqrt(8 * math.log(2))
  return symGaussFilt(sigma,filtbinwidth)


def symGaussFilt (filtsigma, filtbinwidth):
  filtsigmaNbin = filtsigma / filtbinwidth
  filtbins = int(int(np.ceil(filtsigmaNbin)*8)/2*2+1)
  filtmid = filtbins//2 # filtbins is odd...
  filtx = np.arange(-filtmid,filtmid+1) * filtbinwidth
  filtz = np.arange(-filtmid,filtmid+1) / filtsigmaNbin
  filtnorm = np.sqrt(2 * filtsigmaNbin ** 2 *np.pi)
  filt = np.exp ( -filtz**2 / 2) / filtnorm
  return filt, filtx, filtmid, filtbins


def symLorentz (FWHM, filtbinwidth):
  widthNbin = FWHM / filtbinwidth
  filtbins = int(int(np.ceil(widthNbin)*8)/2*2+1)
  filtmid = filtbins//2 # filtbins is odd...
  filtx = np.arange(-filtmid,filtmid+1) * filtbinwidth
  filtz = np.arange(-filtmid,filtmid+1) / widthNbin
  filtnum = 1 / np.pi / widthNbin
  filt = filtnum / (1 + filtz**2)
  return filt, filtx, filtmid, filtbins


def genGaussianFWHM (FWHM, filtbinwidth, beta):
  alpha = FWHM * math.pow(math.log(2), -1.0/beta)
  return genGaussian(alpha,filtbinwidth,beta)


def genGaussian (alpha, filtbinwidth, beta):
  alphaNbin = alpha / filtbinwidth
  filtbins = int(int(np.ceil(alphaNbin)*8)/2*2+1)
  filtmid = filtbins//2 # filtbins is odd...
  filtx = np.arange(-filtmid,filtmid+1) * filtbinwidth
  filtz = np.arange(-filtmid,filtmid+1) / alphaNbin
  num = beta / (2 * alphaNbin * math.gamma(1/beta))
  filt = num * np.exp(-np.power(np.abs(filtz),beta))
  return filt, filtx, filtmid, filtbins


def wiener_filter_withpad(signal, filt, filtmid, Z):
  # Apply wiener deconvolution filter to array
  # allows use of a non-causal filter (e.g. spatial symmetric)
  # by specifying the filter mid-point/zero point
  # also sufficiently pads the signal to be deconvolved to
  # avoid periodicity issues
  filtbins = len(filt)
  padval = 0 # in deconvolution doesn't make much sense to pad non-zero
  filtpadval = 0 # separate from signal pad value
  # We're deconvolving, so the shift the filter introduces is added
  # in the padding, don't need to remove it from the start afterwards
  # as deconvolution shifts the signal back
  # The entirety of the padding is removed from the end.
  sigpad = np.pad(signal, ((filtbins-(filtmid+1)),filtmid),
    'constant', constant_values=(padval, padval))
  filtpad = np.pad(filt, (0,sigpad.shape[0]-filt.shape[0]),
    'constant', constant_values=(0, 0))
  F = np.fft.fft(sigpad)
  H = np.fft.fft(filtpad)
  H2 = np.conj(H)*H
  G = (1/H) * (H2 / (H2+Z))
  Ffilt = F * G
  sigfilt = np.real(np.fft.ifft(Ffilt))
  #plt.subplot(2,1,1)
  #sigrange=np.arange(0,(histbinwidth*(sigpad.shape[0])),histbinwidth)
  #plt.plot(sigrange,sigpad)
  #plt.subplot(2,1,2)
  #plt.plot(filtpad)
  return sigfilt[slice(signal.shape[0])]


def Eu_v (distu, valsu, filt, distv, filtmid=None) :
  filtlen = filt.shape[0]
  if filtmid is None:
    filtmid = filtlen // 2
  u1 = distu * valsu
  conv1 = np.convolve(u1,filt,mode="full")[filtlen-1-filtmid:-filtmid]
  conv2 = np.convolve(distu,filt,mode="full")[filtlen-1-filtmid:-filtmid]
  uest = conv1/conv2
  #plt.subplot(4,1,1)
  #plt.plot(u1)
  #plt.subplot(4,1,2)
  #plt.plot(conv1)
  #plt.subplot(4,1,3)
  #plt.plot(conv2)
  #plt.subplot(4,1,4)
  #plt.plot(uest)
  return uest, u1, conv1, conv2


def map_Eu_v(histval,uest, v):
  # histval are bin centers, this is because uest
  # is estimated by convolution, so need to use E(v)
  # and E(u) values for each bin.
  # Could just use np.interp here, however it will result
  # in clipping at the max and min values, which loses a
  # bin either end, so a bit more elegant to do it ourselves
  vwidths = histval[1:] - histval[:-1]
  # Clipping means we project linearly past the ends of the
  # array based on the first and last intervals
  vbins = np.clip(np.searchsorted(histval,v)-1,0,histval.shape[0]-2)
  vbindist = (v - histval[vbins]) / vwidths[vbins]
  uwidths = uest[1:] - uest[:-1]
  u = uest[vbins] + uwidths[vbins] * vbindist
  return u


def map_quantu_v(binvals, distv, distu, v):
  # Use the idea the voxel value quantiles should remain the
  # same transferring between the distributions
  # Doesn't work properly, remapped histogram comes out very spiky
  # and CDF closer to the original than that of the filtered
  # distribution. If working correctly then the resulting CDF
  # should resembled that of the filtered distribution
  # Also, bin values are the bin centres, so there's a slight
  # offset issue to deal with, but minor so long as the bins
  # are narrow. Not entirely sure using correct binwidths for
  # the interpolation (might be shifted by one)

  # cumv are to the bin upper edge. fudge a bit and approximate
  # the middle value
  cumv = np.cumsum(distv)/float(np.sum(distv))
  cumv = np.hstack((0,cumv))
  cumv = (cumv[0:-1] + cumv[1:])/2
  cumvdiffs = cumv[1:] - cumv[:-1]

  vwidths = binvals[1:] - binvals[:-1]
  vbin = np.clip(np.searchsorted(binvals,v)-1,0,cumv.shape[0]-2)
  vbindist = (v - binvals[vbin])/ vwidths[vbin]
  
  vpercentile = cumv[vbin] + cumvdiffs[vbin] * vbindist

  # cumu, now dealing with the quantiles and project
  # back to histval
  cumu = np.cumsum(distu)/float(np.sum(distu))
  cumu = np.hstack((0,cumv))
  cumu = (cumu[0:-1] + cumu[1:])/2
  cumudiffs = cumu[1:] - cumu[:-1]

  with np.errstate(divide='ignore'):
    cumuquantscale = 1 / cumudiffs
  # This is wrong: it's saying that if the percentile is in the
  # next higher bin it gets stuck in the current one if the adjacent
  # bin is the same, should actually be a leap. Could be one cause of
  # spicky-ness
  cumuquantscale[np.isinf(cumuquantscale)] = 0
  quantbin = np.clip(np.searchsorted(cumu,vpercentile)-1,0,cumu.shape[0]-2)
  quantbindist = (vpercentile - cumu[quantbin]) * \
    cumuquantscale[quantbin]
  # vwidths are intervals for values bins represent for both
  # original and filtered histogram
  quantmappedu = binvals[quantbin] + \
    vwidths[quantbin] * quantbindist[quantbin]
  return quantmappedu


def kernelfntri(x,width):
    ax=np.fabs(x)
    return (width-ax).clip(0,None)/width**2


def kernelfnhat(x,width):
    ax=np.fabs(x)
    return (width>ax).clip(0,None)/(2*width)

  
def kernelfngauss(x,width):
    return np.exp(-x**2/(2*width**2))/math.sqrt(2*math.pi*width**2)


# Several times faster than the sm.nonparametric.KDEUnivariate equivalent
# which matters if we need to repeat it for optimisation
def kdepdf (x, data, bw, kernfn=kernelfngauss):
    pdf = np.zeros(x.shape)
    for N in range(x.shape[0]):
        searchwidth=3*bw
        thisset = np.argwhere(np.logical_and(data <= x[N]+searchwidth, data >= x[N]-searchwidth))
        pdf[N] = kernfn(data[thisset]-x[N],bw).sum()
    return pdf


def distrib_histo(data, Nbins):
    hist,histvaledge = np.histogram(data,Nbins)
    histwidth = histvaledge[-1] - histvaledge[0]
    histval = (histvaledge[0:-1] + histvaledge[1:])/2
    histbinwidth = histwidth / (histval.shape[0]-1)
    return hist,histvaledge,histval,histbinwidth


def distrib_kde(data,Nbins, bw=None, kernfn=kernelfngauss,
                binCentreLimits=False, binPastLimits=False):
    # The binCentreLimits==False behaviour produces the same
    # bins as numpy.histogram, with binCentreLimits==True
    # the first and last bin centres are on the minimum and
    # maximum values. For histogram sharpening this will
    # lead to slight divergence at the extrema.
    if binCentreLimits:
      histval = np.linspace(data.min(),data.max(),
          num=Nbins)
      histwidth = histval[-1] - histval[0]
      histbinwidth = histwidth / (histval.shape[0]-1)
      histvaledge = histval - histbinwidth/2
      histvaledge = np.append(histvaledge, histval[-1] - histbinwidth/2)
    else:
      if binPastLimits:
        histwidthshort = data.max() - data.min()
        histbinwidthshort = histwidthshort / (Nbins-2)
        histlim=(data.min()-histbinwidthshort,
                 data.max()+histbinwidthshort)
      else:
        histlim=(data.min(), data.max())
      histvaledge = np.linspace(histlim[0],
                                histlim[1],
                                num=Nbins+1)
      # We recalculate all to make sure they're consistent
      histwidth = histvaledge[-1] - histvaledge[0]
      histval = (histvaledge[0:-1] + histvaledge[1:])/2
      histbinwidth = histwidth / (histval.shape[0])
    if bw is None:
      bw = histbinwidth
    hist = kdepdf(histval, data, bw, kernfn=kernfn)
    return hist,histvaledge,histval,histbinwidth


def getblockstats(data, mask, blocksize=7, sampsize=100, debug=False):
  np.random.seed()
  masklist = np.argwhere(mask)
  blockneg=-blocksize//2
  blockpos=blocksize//2
  blockrange=np.arange(blockneg,blockpos)
  blockvar = []
  blockmean = []
  blockcohen2 = []
  blockn = []
  while (len(blockvar) < sampsize):
    choice=np.random.randint(0,masklist.shape[0]-1)
    if (len(blockvar)==0 and debug):
      print ("blocksize {} first block {}".format(blocksize,choice))
    blockcentre = masklist[choice]
    #includedallax = np.zeros(mask.shape)
    subidx = [slice(None)]*mask.ndim
    for ax in range(len(blockcentre)):
      els = range(max(0, blockcentre[ax]+blockneg),
                  min(mask.shape[ax],blockcentre[ax]+blockpos))
      subidx[ax] = els
    #includedallax[np.ix_(*subidx)] = 1
    #includedallax = np.logical_and(mask, includedallax)
    #blockvals = data[includedallax]
    blockmask = mask[np.ix_(*subidx)]
    if (np.sum(blockmask)<2):
      continue
    blockvals = data[np.ix_(*subidx)]
    blockvals = blockvals[blockmask]
    blockvar.append(np.var(blockvals,ddof=1))
    blockmean.append(np.average(blockvals))
    blockn.append(blockvals.shape[0])
  _m = np.average(blockmean, weights=blockn)
  _v = np.average((blockmean-_m)**2,weights=np.sqrt(blockn)) * \
    len(blockmean)/(len(blockmean)-1)
  blockmeanvar=_v
  return blocksize, np.mean(blockmean), blockmeanvar, np.mean(blockvar), np.var(blockvar)
#bs = range(3,20)#stats=[]
#for N in range(len(bs)):
#  stats.append(getblockstats(datalog, mask, bs[N]))
#bmm, bmv, bvm, bvv = zip(*stats)


def pargetblockstats(data, mask, blocksize=7, sampsize=100):
  print("blocksize {}".format(blocksize))
  masklist = np.argwhere(mask)
  blockneg=-blocksize//2                 
  blockpos=blocksize//2                  
  blockrange=np.arange(blockneg,blockpos)
  def getblockvar(blockchoice):
    print ("choice {}".format(blockchoice))
    try:
      blockcentre = masklist[blockchoice]            
      includedallax = np.ones(masklist.shape[0])                  
      for ax in range(len(blockcentre)):                          
        # thought where was tuple of index arrays, but if [choice] 
        # indexes then not right?                                  
        included = np.logical_and(masklist[:,ax] >= blockcentre[ax]      
                                  + blockneg,
                                  masklist[:,ax] < blockcentre[ax] + blockpos)
        includedallax = np.logical_and(included, includedallax)
      mget=masklist[includedallax]                    
      blockvals = data[mget[:,0],mget[:,1], mget[:,2]]
      if (np.sum(includedallax) <= 1):
        # Maybe should resample at this point till we get a good one?
        # Sparse masking voxels could be tricky, but that situation
        # shouldn't really occur with normal images
        # Would save worrying over presence of NaN later
        print ("Hit 1 vox block")
        return (np.NAN, np.NAN)
      return (np.mean(blockvals), np.var(blockvals,ddof=1))
    except KeyboardInterrupt:
      print ("<child processor> ignores Ctl-C")
      pass

  pool = mp.Pool(processes=mp.cpu_count()*2)
  try:
    poolres = [pool.apply_async(getblockvar, args=(choice,))
               for choice in np.random.choice(range(masklist.shape[0]),
                                              size=sampsize)]
  except KeyboardInterrupt:
    pool.terminate()
    print ("Loop cancelled")

  pool.close()
  reslist = [p.get() for p in poolres]
  meanvals, varvals = zip(*filter(lambda x: x!=None, reslist))
  blockmeanmean = np.nanmean(meanvals)
  blockvarmean = np.nanmean(varvals)
  blockmeanvar = np.nanvar(meanvals, ddof=1)
  blockvarvar = np.nanvar(varvals, ddof=1)
  return blocksize, blockmeanmean, blockmeanvar, blockvarmean, blockvarvar

 
def picksdremmeanvar(data, mask, sampsize=100, bw=4):
  import statsmodels.nonparametric.smoothers_lowess as ls
  import statsmodels.nonparametric.kernel_regression as kreg
  bsrange = np.arange(3,51,2)
  pool = mp.Pool(processes=mp.cpu_count()*2)
  try:
    poolres = [pool.apply_async(getblockstats, args=(data, mask, bs, sampsize))
               for bs in bsrange]
  except KeyboardInterrupt:
    pool.terminate()
    print ("Loop cancelled")
  pool.close()
  stats = [p.get() for p in poolres]

  msize, bmm, bmv, bvm, bvv = zip(*stats)
  print(msize)
  #bvvls = ls.lowess(bvv,bsrange, is_sorted=True, return_sorted=False,
  #                  frac=0.5)
  #bvmls = ls.lowess(bvm,bsrange, is_sorted=True, return_sorted=False)
  bvvKreg = kreg.KernelReg(bvv,bsrange,"c",bw=[bw])
  bvvls = bvvKreg.fit(bsrange)[0]
  bmvKreg = kreg.KernelReg(bmv,bsrange,"c",bw=[bw])
  bmvls = bmvKreg.fit(bsrange)[0]
  #import matplotlib
  ##matplotlib.use('TkAgg')
  #import matplotlib.pyplot as plt
  #plt.plot(bmvls,color="Blue")
  #plt.plot(np.sqrt(bvvls),color="Red")
  #plt.show()
  #bvvmaxind = np.argmax(bvvls)
  #searchind = (bvvmaxind+1)*2 - 1
  #s2 =  bmvls[searchind]
  s2 = bmvls[0] - np.sqrt(bvvls[0])
  print ("bmvls0 {} sqrt bvvls0 {} s2 {}".format(bmvls[0], np.sqrt(bvvls[0]), s2))
  s = math.sqrt(s2)
  print(s)
  return s


def singlepicksdremmeanvar(data, mask, sampsize=100, bw=4):
  import statsmodels.nonparametric.smoothers_lowess as ls
  import statsmodels.nonparametric.kernel_regression as kreg
  bsrange = np.arange(3,51,2)
  stats=[]
  for bs in bsrange:
    print ("bs {}".format(bs))
    stats.append(getblockstats(data, mask, bs, sampsize))
  msize, bmm, bmv, bvm, bvv = zip(*stats)
  print(msize)
  #bvvls = ls.lowess(bvv,bsrange, is_sorted=True, return_sorted=False,
  #                  frac=0.5)
  #bvmls = ls.lowess(bvm,bsrange, is_sorted=True, return_sorted=False)
  bvvKreg = kreg.KernelReg(bvv,bsrange,"c",bw=[bw])
  bvvls = bvvKreg.fit(bsrange)[0]
  bmvKreg = kreg.KernelReg(bmv,bsrange,"c",bw=[bw])
  bmvls = bmvKreg.fit(bsrange)[0]
  #import matplotlib
  ##matplotlib.use('TkAgg')
  #import matplotlib.pyplot as plt
  #plt.plot(bmvls,color="Blue")
  #plt.plot(np.sqrt(bvvls),color="Red")
  #plt.show()
  #bvvmaxind = np.argmax(bvvls)
  #searchind = (bvvmaxind+1)*2 - 1
  #s2 =  bmvls[searchind]
  s2 = bmvls[0] - np.sqrt(bvvls[0])
  print ("bmvls0 {} sqrt bvvls0 {} s2 {}".format(bmvls[0], np.sqrt(bvvls[0]), s2))
  s = math.sqrt(s2)
  print(s)
  return s



def picksdexcessvar(data, mask, sampsize=100):
  import statsmodels.nonparametric.smoothers_lowess as ls
  import statsmodels.nonparametric.kernel_regression as kreg
  while True:
    bsrange = np.arange(3,51,2)
    stats=[]
    for bs in bsrange:
      stats.append(getblockstats(data, mask, bs, sampsize))
    bmm, bmv, bvm, bvv = zip(*stats)
    #bvvls = ls.lowess(bvv,bsrange, is_sorted=True, return_sorted=False,
    #                  frac=0.5)
    #bvmls = ls.lowess(bvm,bsrange, is_sorted=True, return_sorted=False)
    bvvKreg = kreg.KernelReg(bvv,bsrange,"c")
    bvvls = bvvKreg.fit(bsrange)[0]
    bvmKreg = kreg.KernelReg(bvm,bsrange,"c")
    bvmls = bvmKreg.fit(bsrange)[0]
    bmvKreg = kreg.KernelReg(bmv,bsrange,"c")
    bmvls = bmvKreg.fit(bsrange)[0]
    bvvtgt = (bvvls.max() + bvvls[-1])/2
    bvvmaxind = np.argmax(bvvls)
    if np.isnan(bvvls.max()):
      print(bvv)
    print ("bvvlsmax {}, bvvlsend {}, bvvtgt {}, bvvmaxind {}".format(bvvls.max(), bvvls[-1],bvvtgt,bvvmaxind))
    inds = np.argwhere(bvvls <= bvvtgt)
    inds = inds[inds > bvvmaxind]
    firstind = inds.min()
    allvar = np.var(data[mask])
    #s2 = allvar - bvmls[firstind]
    bmvatmax = bmvls[bvvmaxind]
    bvmnoiseest = bvm[0]
    s2 = allvar - bmvatmax - bvmnoiseest
    print (bvvls.max(), bvvls[-1], bvvtgt, firstind, bvv[firstind], bvvls[firstind], bvm[firstind], bvmls[firstind], allvar, bmvatmax, bvmnoiseest, s2)
    if s2 > 0:
      s = math.sqrt(s2)
      print(s)
      return s
    else:
      print ("Negative bias field variance estimate {}, may be stuck".format(s2))
  


def picksd(data, mask, blocksize=7, sampsize=100):
  masklist = np.argwhere(mask)
  blockneg=-blocksize//2
  blockpos=blocksize//2
  blockrange=np.arange(blockneg,blockpos)
  blockvar = []
  blockmean = []
  blockcohen2 = []
  blockn = []
  for choice in np.random.choice(range(masklist.shape[0]), size=sampsize):
    blockcentre = masklist[choice]
    includedallax = np.ones(masklist.shape[0])
    for ax in range(len(blockcentre)):
      # thought where was tuple of index arrays, but if [choice]
      # indexes then not right?
      included = np.logical_and(masklist[:,ax] >= blockcentre[ax]
                                + blockneg,
                                masklist[:,ax] < blockcentre[ax] + blockpos)
      includedallax = np.logical_and(included, includedallax)
    mget=masklist[includedallax]
    blockvals = data[mget[:,0],mget[:,1], mget[:,2]]
    blockvar.append(np.var(blockvals,ddof=1))
    blockmean.append(np.average(blockvals))
    blockcohen2.append(blockvar[-1]/blockmean[-1]**2)
    blockn.append(includedallax.sum())
  _m = np.average(blockmean, weights=blockn)
  _v = np.average((blockmean-_m)**2,weights=np.sqrt(blockn)) * \
    len(blockmean)/(len(blockmean)-1)
  blockmeanvar=_v
  blockvaradj = np.array(blockcohen2)*_m**2
  s = (blockmeanvar - np.average(blockvaradj, weights=blockn))
  print(len(blockmean),_m,_v,blockmeanvar, np.average(blockvar, weights=blockn), np.average(blockvaradj, weights=blockn), s)
  return math.sqrt(s)


def optFWHM(hist, histbinwidth, Z=0.01, range=(0.01,1,0.005), filtfunc=symGaussFiltFWHM):
    varlist = []
    fwlist = []
    for FWHM in np.arange(*range) :
        filt, filtx, filtmid, filtbins = filtfunc(FWHM,histbinwidth)
        histfilt = wiener_filter_withpad(hist, filt, filtmid, Z)
        histfiltclip = np.clip(histfilt,0,None)
        histprod = np.sqrt(hist*histfiltclip)
        xvals = np.arange(0,hist.shape[0])
        mv = np.average(xvals, weights=histprod)
        var = np.average((xvals-mv)**2, weights=histprod)
        varlist.append(var)
        fwlist.append(FWHM)
    varmaxind = np.array(varlist).argmax()
    return(fwlist[varmaxind])

def entropyEval(hist, histbinwidth, histval, data, fwhm, Z=0.01, filtfunc=symGaussFiltFWHM, distrib="histo"):
    Nbins = hist.shape[0]
    mfilt, mfiltx, mfiltmid, mfiltbins = filtfunc(fwhm, histbinwidth)
    histfilt = wiener_filter_withpad(hist, mfilt, mfiltmid, Z)
    histfiltclip = np.clip(histfilt,0,None)
    uest, u1, conv1, conv2 = Eu_v(histfiltclip, histval, mfilt, hist)
    dataupd = map_Eu_v(histval, uest, data)
    if distrib == "histo":
      newhist,histedge = np.histogram(dataupd,Nbins,(data.min(),data.max()))
    elif distrib == "kde":
      newhist,histedge,_val,_width = distrib_kde(dataupd,Nbins,bw=histbinwidth)
    
    histfiltclipsm = (histfiltclip+1.0)/(histfiltclip.sum()+Nbins)
    newhistsm = (newhist + 1.0)/(newhist.sum()+Nbins)
    doplt=False
    if doplt:
        plt.plot(histval, newhist)
        plt.plot(histval, hist)
        plt.plot(histval, histfilt)
    rval=stats.entropy(newhistsm,histfiltclipsm)
    #rval = np.var(datalogmaskedupd)
    #print(rval)
    return rval

def optEntropyFWHMsing(hist, histbinwidth, histval, data, Z=0.01, fwrange=(0.01,1,0.005), filtfunc=symGaussFiltFWHM, distrib="histo"):
    varlist = []
    fwlist = np.arange(*fwrange)
    for fwhm in fwlist:
        varlist.append(entropyEval(hist,histbinwidth,histval,data,fwhm,Z,filtfunc,distrib))
    varminind = np.array(varlist).argmin()
    return(fwlist[varminind])


def optEntropyFWHMpar(hist, histbinwidth, histval, data, Z=0.01, fwrange=(0.01,1,0.005), filtfunc=symGaussFiltFWHM, distrib="histo"):
    fwlist = np.arange(*fwrange)
    pool = mp.Pool(processes=mp.cpu_count())
    poolres = [pool.apply_async(entropyEval,
      args=(hist,histbinwidth,histval,data,fwhm,Z,filtfunc,distrib))
               for fwhm in fwlist]
    pool.close()
    varlist = [p.get() for p in poolres]

    varminind = np.array(varlist).argmin()
    return(fwlist[varminind])

optEntropyFWHM = optEntropyFWHMpar
