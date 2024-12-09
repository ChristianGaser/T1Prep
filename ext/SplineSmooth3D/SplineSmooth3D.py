from builtins import *

import time

import numpy.linalg as linalg
import numpy as np
import scipy.ndimage as ndimage

try:
  from .SplineSupport import knots_over_domain, eval_nonzero_bspl
except ValueError:
  from SplineSupport import knots_over_domain, eval_nonzero_bspl


class SplineSmooth3D(object):
  """3D smoothing tensor B-spline class for fields defined as numpy
  arrays, with thin-plate spline bending regularisation.

  Intended to replace spline smoothing of Sled et al. TMI 17, 1998,
  follows some details of https://github.com/BIC-MNI/N3/ with regard
  to knot placement. Use of Cholesky decomp for solver, Green and
  Silverman, Nonparametric regression[...] (1994); Wahba, Spline
  Models for Observational Data (1990). Analytic form of
  thin-plate-spline penalty Shackleford et al. LNCS 7511 MICCAI 2012
  (with correction to match form used in Sled, matching Wahba and
  other sources).

  Parameters
  ----------
    file_loc : str
    The file location of the spreadsheet
    print_cols : bool, optional
    A flag used to print the columns to the console (default is False)

  data : ndarray
      3D array matching shape of data to smooth
  voxsizes : tuple
      tuple of floats for voxel size in each dimension
  spacing : float
      knot spacing to use, units should match those of `voxsizes`
  mask : ndarray or None, optional
      ndarray same shape as `data`, 0 for exclude, 1 for include. May
      be of any type, including boolean
  q : int, optional
      Spline degree, only q=3 (cubic) supported by most functions in
      this class
  domainMethod : str, optional
      Method to use for knot placement, see knots_over_domain, default
      'centre'. Certain domainMethod values will produce different
      spacings.
  Lambda : float or None, optional:
      Smoothing parameter. Default None: don't use smoothing. Images
      without full support (when using masks) may fail to solve.
  dofit : bool, optional
      Fit on initialisation (with supplied data), default True.


  Attributes
  ----------
  data : ndarray
      Last data passed to fit
  shape : tuple
      Shape of data, set at instantiation
  voxsizes : tuple
      Dimensions of voxels in each direction
  spacing : float
      Knot spacing
  mask : ndarray or None
      Set at instantion, array same shape as data to fit
  q : int
      Spline degree, only q=3 (cubic) supported by most
      functions in this class
  domainMethod : str
      Method to use for knot placement, see knots_over_domain
  Lambda : float
      Requested spacing (same units as voxsize) for knots/control
      points. Certain domainMethod values will produce different
      spacings.
  mincLambda : bool
      # Not implemented yet, see solve. Whether to use an adjstment
      that makes lambda values compatible with MINC spline_smooth
  costDerivative : int
      Derivative order for cost function associated with Lambda,
      default 2 is thin plate bending energy. 0 penalises square of
      coefficients. Must be between 0 and q-1
  P : ndarray
      Fitted parameters as vector
  Atx : ndarray
      data projected to parameter space
  AtA : ndarray
      defines inner product on spline coefficients
  Jsub : ndarray
      thin-plate-energy matrix for the coefficients supporting
      an interval
  J : ndarray
      thin-plate-energy matrix for all coefficients

  Methods
  -------
  fit(data,reportingLevel=0)
      Fit new data. Invalidates current parameter solution. First use
      in a new instance will also build inner product matrix, re-running
      fit for updated data is faster than creating a new instance to fit.
  solve(Lambda=None, mincLambda=True, reportingLevel=0):
      Solve for parameters. Will fit current data first if not yet done.
      Can be re-run with a different value of Lambda to change smoothing.
  predict(reportingLevel=0)
      Predict smoothed data from current parameters. Will run all prior
      steps if not already performed.

  """

  def __init__(self,data,voxsizes,spacing,mask=None,
               q=3,
               domainMethod="centre", knts=None,
               Lambda=None, mincLambda=True, voxelsLambda=False,
               costDerivative=2,
               dofit=True):
    #super(SplineSmooth3D,self).__init__()
    self.data = data
    self.shape = data.shape
    self.voxsizes = voxsizes
    self.spacing = spacing
    self.mask = mask
    self.q=q
    self.domainMethod = domainMethod
    self.mincLambda = mincLambda
    self.voxelsLambda= voxelsLambda
    self.costDerivative = costDerivative

    if Lambda is None:
      self.Lambda={}
    elif isinstance(Lambda, dict):
      self.Lambda=Lambda
    else:
      self.Lambda = {costDerivative:Lambda}
    Lambda = self.Lambda

    self.AtA=None
    self.Atx=None
    self.P=None
    self.Jsub=dict()
    self.J=dict()
    self.promotion = 0 # Tracker for spline promotion

    if ( np.min(self.shape) < 2 or len(self.shape) !=3 or
         voxsizes.size != 3 ):
      # Not too difficult to extend to 1D/2D, would
      # need to special case the support coefficients
      # a bit, the regularisation would also need some
      # work
      raise ValueError("SplineSmooth3D currently only "+
                       "works with 3D, non-flat arrays")
    if (q !=3):
      # Least squares spline fitting will work, but
      # regularisation needs extending to accomodate
      # non cubic-splines
      raise ValueError("SplineSmooth3D only cubic "+
                       "splines currently supported")
    if (not spacing > 0):
      raise ValueError("SplineSmooth3D, knot spacing "+
                       "must be > 0")

    if mask is None:
      self.mask = np.ones(self.shape,dtype=self.data.dtype)
    else:
      self.mask = mask.astype(self.data.dtype,copy=True)

    self.setupKCP(knts)
    for deriv in Lambda:
      self.Jsub[deriv] = self.buildJ(deriv)
    if(dofit):
      self.fit()
    return None


  def coefIntervalIter(self, reportingLevel=0):
    """Generator for the common iteration loop over supported intervals

    Parameters
    ----------
    reportingLevel : int, optional
    If >=2 then report timing information for each inner loop

    Yields
    -------
    cIndZ,cIndY,cIndX : int
        Coefficient starting indices in each dimension (mainly
        for debugging)
    coefsZ,coefsY,coefsX : ndarray
        matrices of control point coefficients by direction for
        each grid step along that direction within the supported
        interval
    rangeZ,rangeY,rangeX : list
        pairs of indices representing data range of supported interval
        in each dimension
    tgtinds : ndarray
        target indices into the parameter array for control points
        supporting this interval
    """
    kntsArr=self.kntsArr
    invCoefArr=self.invCoefArr
    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern
    t_start=time.time()
    t_last=t_start
    for cIndZ in range(len(invCoefArr[0])):
      coefList = invCoefArr[0][cIndZ]
      if coefList is None:
        continue
      firstZ, coefsZ = coefList
      nindZ=coefsZ.shape[0]
      for cIndY in range(len(invCoefArr[1])):
        coefList = invCoefArr[1][cIndY]
        if coefList is None:
          continue
        firstY, coefsY = coefList
        nindY=coefsY.shape[0]
        for cIndX in range(len(invCoefArr[2])):
          coefList = invCoefArr[2][cIndX]
          if coefList is None:
            continue
          firstX, coefsX = coefList
          nindX=coefsX.shape[0]

          indsX = indsXpattern + cIndX
          indsY = indsYpattern + cIndY * kntsArr[2][0]
          indsZ = indsZpattern + cIndZ * kntsArr[2][0] * kntsArr[1][0]
          tgtinds = indsZ + indsY + indsX
    
          rangeZ =[firstZ,firstZ+nindZ]
          rangeY =[firstY,firstY+nindY]
          rangeX =[firstX,firstX+nindX]
          if reportingLevel >=2 :
            t_new=time.time()
            t_step=t_new-t_last
            t_last=t_new
            print("{} {} {}: {:.2g}seconds".format(cIndZ,cIndY,cIndX,
                                             t_step))
          yield cIndZ,cIndY,cIndX, \
          coefsZ,coefsY,coefsX, \
          rangeZ,rangeY,rangeX, \
          tgtinds


  def fit(self,data=None, reportingLevel=0):
    """fit current data, or update current data and fit

    fit does not solve for parameters, it calculates the projection of
    data to parameter space `Atx`, and if necessary (on the first run)
    the parameter inner product matrix `AtA`. This means re-using fit
    for updated data is quicker (about by about three times) than
    using a fresh instance. However the `AtA` matrix depends on the
    shape of data, the mask and the knot placement, a new instance is
    required if any of these are to be changed.

    Running fit will invalidate the current parameters solution `P`.

    Parameters
    ----------
    data : ndarray : optional
        New data to fit, if no new data then will fit the previously
        supplied data.
    reportingLevel : int, optional
        If >=1 then report time taken for the fitting loop, if >=2
        then will also report time for each inner loop (each interval)

    Returns
    -------
    AtA : ndarray
        Parameter inner product matrix
    Atx : ndarray
        Projection of data to parameter space

    """
    mask = self.mask
    if (data is None):
      data = self.data
    if (not data.shape == self.data.shape):
      raise ValueError("Shape of data passed to fit must"+
                       "match that of data used to initialise")
    self.data=data
    self.P=None # Invalidate P
    totalPar = self.totalPar
    needAtA = self.AtA is None
    if needAtA:
      self.AtA = np.zeros((totalPar,totalPar))
    self.Atx = np.zeros(totalPar)
    AtA = self.AtA
    AtAflat = AtA.reshape((AtA.shape[0]*AtA.shape[1]))
    Atx = self.Atx
    kntsArr = self.kntsArr
    invCoefArr = self.invCoefArr

    t_start=time.time()
    t_last=t_start

    q1=self.q+1
    q13 = q1**3

    # optimal path, numpy 1.12 is slower for our case with just
    # True setting, numpy 1.16 is okay with both (and significantly
    # faster too)
    optimize="optimal"
    AtAOptPath=None
    AtxOptPath=None

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      localData = data[rangeZ[0]:rangeZ[1],
                       rangeY[0]:rangeY[1],
                       rangeX[0]:rangeX[1]]
      localMask = mask[rangeZ[0]:rangeZ[1],
                       rangeY[0]:rangeY[1],
                       rangeX[0]:rangeX[1]]
      # This is what we're actually doing, below, and may help
      # explain what might seem an odd order of indices
      #localAtens = np.einsum("zi,yj,xk->zyxijk",
      #                       coefsZ,coefsY,coefsX)
      #localA = localAtens.reshape((-1,q13))
      #localAtx = np.matmul(localA.transpose(),localData.reshape(-1))
      #localAtA = np.matmul(localA.transpose(),localA)

      AtxSum = "xc,yb,za,zyx,zyx->abc"
      if AtxOptPath is None:
        AtxOptPath = np.einsum_path(AtxSum,
          coefsX,coefsY,coefsZ,
          localData, localMask, optimize=optimize)[0]
      localAtx = np.einsum(
        AtxSum,
        coefsX,coefsY,coefsZ,
        localData, localMask, optimize=AtxOptPath).reshape((-1))

      Atx[tgtinds] += localAtx
      if needAtA:
        AtASum = "xc,yb,za,zi,yj,xk,zyx->abcijk"
        if AtAOptPath is None:
          AtAOptPath = np.einsum_path(
          AtASum,
          coefsX,coefsY,coefsZ,
          coefsZ,coefsY,coefsX,
          localMask, optimize=optimize )[0]
        localAtA = np.einsum(
          AtASum,
          coefsX,coefsY,coefsZ,
          coefsZ,coefsY,coefsX,
          localMask, optimize=AtAOptPath ).reshape((q13,q13))
        flatinds = tgtinds.reshape((q13,1)) + \
                   totalPar * tgtinds.reshape((1,q13))
        AtAflat[flatinds.reshape(q13**2)] += localAtA.reshape((q13**2))

    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)

    return AtA, Atx


    
  def solve(self,Lambda=None, mincLambda=None, voxelsLambda=None,
            reportingLevel=0):
    """determine paramters for current data fit

    solve will update the currently fitted parameters, if necessary
    running `fit()` first. It can be re-run with different Lambda
    smoothing parameters to allow producing different predictions
    without the need for re-fitting.

    Parameters
    ----------
    Lambda : float or None, optional
        The weighting to use for the bending energy. If None or not
        supplied then the value set at creation is used. If it was
        None at object creation then no smoothing is used (this may
        fail to solve if insufficient data coverage).
    mincLambda : bool, optional
        Defaults to initialised value. There are two quirks to the
        MINC N3 definition of Lambda that cause it to be a very
        different amount (by a factor of around 500-50000) of
        smoothing to what might be expected: first bending energy is
        computed purely on the scaled knot locations, meaning both
        volume and derivative scaling terms are missing, second the
        spline basis functions are not normalised, leading to a factor
        of 6^2 in AtA for each dimension (6^6=46656 overall). By
        default we produce the same smoothing for a given Lambda that
        MINC will, internally the Lambda used is
        Lambda_{MINC}*spacing/(6^6).

    voxelsLambda : bool, optional
        Defaults to initialised value. Whether to multiply the bending
        energy by the number of voxels in the volume (or mask). In N3
        v1.10 this did not happen; a fixed lambda=1.0 was used,
        although adjusted for subsampling (divided by subsampling
        factor cubed), in v1.12 the bending energy is multiplied by
        number of voxels used and separate subsampling adjustment is
        not needed. Appropriate lambda are therefore much lower; the
        default lambda for N3 v1.12 is 10^-7, roughly the reciprocal
        of the number of voxels in an ADNI-3 image and resulting in
        the same overall penalty. However masked smooths will differ,
        as the mask volume was not previously explicitly adjusted for.
        (But domain would have been reduced to the bounding box for
        the mask, offsetting this effect somewhat.)

    Returns
    -------
    P : ndarray
        Fitted control point parameters, as vector

    """
    if Lambda is None:
      Lambda = self.Lambda
    if mincLambda is None:
      mincLambda = self.mincLambda
    if voxelsLambda is None:
      voxelsLambda = self.voxelsLambda

    if self.AtA is None:
      self.fit(data)
    J = self.J
    AtA=self.AtA
    Atx=self.Atx

    if Lambda is None:
      Lambda={}
    elif not isinstance(Lambda, dict):
      Lambda = {self.costDerivative:Lambda}

    for deriv in Lambda:
      if not deriv in self.J:
        self.J[deriv] = \
          self.buildFullJ(deriv=deriv,reportingLevel=reportingLevel)

    AtAJ = AtA
    for deriv in Lambda:
      lambdaFac = Lambda[deriv]
      if mincLambda:
        # MINC calculates its bending energy integrals without
        # using knot distances, so the (d2f/dx2)^2 volume
        # integral that should have units 1/spacing doesn't.
        # Since our J is calculated with spacing, we need to
        # multiply back.
        # Additionally MINC also calculates its AtA matrix
        # with just the cube values in the Bspline coefficents,
        # not the 1/6 normally used (e.g. B1(u) = (1-u)^3/6)
        # this means its AtA (product of three tensors squared)
        # is a factor 6^3^2 higher than ours, to get the
        # equivalent internal Lambda to a requested Lambda_minc
        # we therefore need to multiply by spacing and divide
        # by 6^6.
        # Confirmed: tested this against MINC on a range of
        # image FOV, spacings, lambda, and resolutions.
        lambdaFac *= self.spacing**(2*deriv-3) / (6**3)**2
      if voxelsLambda:
        lambdaFac *= np.sum(self.mask > 0)
      AtAJ=AtAJ + J[deriv] * lambdaFac

    L = np.linalg.cholesky(AtAJ)
    p1=np.linalg.solve(L, Atx)
    P=np.linalg.solve(L.T.conj(),p1)
    self.P=P
    return self.P



  def predict(self,reportingLevel=0):
    """predict data (produce smoothed data)

    Predict the data from current parameters, if necessary both the
    fit and solve steps will be run.

    Parameters
    ----------
    reportingLevel : int, optional
        If >=1 then report time taken for the fitting loop, if >=2
        then will also report time for each inner loop (each interval)

    Returns
    -------
    pred : ndarray
        Predicted smoothed data, same shape as `data`

    """
    if self.P is None:
      self.solve(reportingLevel=reportingLevel,
                 Lambda=self.Lambda, mincLambda=self.mincLambda,
                 voxelsLambda=self.voxelsLambda)
    P=self.P
    q=self.q
    q1=q+1
    q13=q1**3
    pred = np.zeros(self.shape)

    t_start=time.time()
    t_last=t_start
    # Same process as earlier, but without AtA calculation.
    localApOptPath=None

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      localP = P[tgtinds]
      # Again, finding the local tensor, then vectorising,
      # without the overall einsum:
      # localAtens = np.einsum("zi,yj,xk->zyxijk",
      #                       coefsZ,coefsY,coefsX)
      # localA = localAtens.reshape((-1,q13))
      # localAp = np.matmul(localA,localP)
      # localAp = localAp.reshape((nindZ,nindY,nindX))
      localApSum="zi,yj,xk,ijk->zyx"
      if localApOptPath is None:
        localApOptPath = np.einsum_path(localApSum,
        coefsZ,coefsY,coefsX,
        localP.reshape((q1,q1,q1)),
        optimize=True)[0]
      localAp = np.einsum(
        localApSum,
        coefsZ,coefsY,coefsX,
        localP.reshape((q1,q1,q1)),
        optimize=localApOptPath)

      pred[rangeZ[0]:rangeZ[1],
           rangeY[0]:rangeY[1],
           rangeX[0]:rangeX[1]] = localAp
    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)
    return pred


    

  def buildFullJ(self,deriv,reportingLevel=0):
    """Build the full bending energy matrix.

    Expands the single interval bending energy matrix to cover the
    full supported domain.

    Parameters
    ----------
    reportingLevel : int, optional
        If >=1 then report time taken for the overall loop, if >=2
        then will also report time for each inner loop (each interval)

    Returns
    -------
    J : ndarray
        Bending energy matrix.

    """
    if deriv not in self.Jsub:
      self.Jsub[deriv] = self.buildJ(deriv)
    datashape = self.shape
    Jsub=self.Jsub[deriv]
    J=np.zeros(self.AtA.shape)
    invCoefArr = self.invCoefArr
    kntsArr=self.kntsArr
    Jflat=J.reshape(-1)
    indsXpattern = self.indsXpattern
    indsYpattern = self.indsYpattern
    indsZpattern = self.indsZpattern
    q = self.q
    q1 = q+1
    q13 = q1**3
    totalPar=self.totalPar
    t_start=time.time()
    t_last=t_start
    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      flatinds = tgtinds.reshape((q13,1)) + \
                 totalPar * tgtinds.reshape((1,q13))
     
      Jflat[flatinds.reshape(-1)] += Jsub.reshape(-1)
    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)
    return J


  def setupKnots(self):
    """construct the knots for each dimension"""
    shape = self.shape
    voxsizes = self.voxsizes
    spacing = self.spacing
    q = self.q
    domain = [[0,(nvox-1)*voxsize]
                   for (nvox, voxsize) in zip(shape, voxsizes)]

    kntsArr = [knots_over_domain(bound[0],bound[1],spacing,
                                 q=q, method=self.domainMethod)
               for bound in domain]
    self.kntsArr = kntsArr


  def setupKCP(self, knts=None):
    """setup knots and control points

    Sets up the control points for the domain and the support coefficients
    for knots along each direction which are used to form the full support
    tensor later.
    """
    if knts is None:
      self.setupKnots()
    else:
      self.kntsArr = knts
    shape = self.shape
    voxsizes = self.voxsizes
    kntsArr = self.kntsArr
    q = self.q
    q1 = q+1

    totalPar = 1
    for knts in self.kntsArr:
      totalPar *= knts[0]
    self.totalPar = totalPar
    # Form the tensor bases, 
    coefArr = [
      [ eval_nonzero_bspl(knts[1],voxsize*v,q=q) for v in range(nvox) ]
      for (knts, nvox, voxsize) in zip (kntsArr, shape, voxsizes)]
    self.coefArr = coefArr
    # Convert to the list of voxels by supported interval
    invCoefArr = [
      self.invertCoefList(coefList)
      for coefList in coefArr
    ]
    self.invCoefArr = invCoefArr

    # X is fastest changing: LAST index
    self.indsXpattern = np.tile(range(0,q1),q1*q1) 
    self.indsYpattern = np.tile(np.repeat(range(0,q1),q1),q1) * kntsArr[2][0]
    self.indsZpattern = np.repeat(range(0,q1),q1*q1) * kntsArr[2][0] * kntsArr[1][0]

    return


  def buildGammaMat(self, order, spacing=None, q=None):
    """Evaluate a component of the bending energy integral.

    Part of Shackleford et al. LNCS 7511 MICCAI 2012, integral of
    requested order derivative squared, corresponds to one of the
    \hat{\Gamma} of Eq. 17. Used to create one component of bending
    energy tensor through Hadamard product.

    Parameters
    ----------
    order : int
        Order "o" of derivative in
        \int (\partial^o f / \partial e_i^o)^2 de_i
    spacing : float, optional
        Spacing of knots. If not supplied then originally set spacing.
    q : int, optional
        Spline degree, only q=3 (cubic) currently supported.
        

    Returns
    -------
    gamma : ndarray
        Component of bending energy tensor.
    """
    if spacing is None:
      spacing = self.spacing
    if q is None:
      q=self.q
    if q!=3:
      raise ValueError('buildGammaMat only supports q=3')
    B = np.array([[1, -3, 3, -1],
                  [4, 0, -6, 3],
                  [1, 3, 3, -3],
                  [0, 0, 0, 1]])
    spacing=float(spacing)
    R = np.diag(np.power(spacing,[0,-1,-2,-3]))
    n = np.arange(1,8)
    psi = np.power(spacing,n)/n
    D = [
      np.diag([1,1,1,1]),
      np.diag([1,2,3],-1),
      np.diag([2,6],-2)
    ]
    Q = np.matmul(B,np.matmul(R,D[order]))
    # xsi_{a,b} is a 4x4 matrix corresponding to
    # xsi_{a,b}_{i,j} = (q_a outer q_b)_{i,j}
    # where q_a is row a of Q
    xsi = np.einsum("ai,bj->abij",Q,Q)
    offset = range((Q.shape[0]-1),-(Q.shape[1]),-1)
    # like:
    # 1000 0100 0010
    # 0000 1000 0100  ... etc.
    # 0000 0000 1000
    # 0000 0000 0000
    sigmabases = np.array(
      [np.fliplr(np.diag([1]*(4-abs(thisOff)),thisOff))
       for thisOff in offset])
    # Don't need sigma itself, but it's abij,uij->abu
    gamma = np.einsum("abij,uij,u->ab",xsi,sigmabases,psi)
    return gamma


  def buildBigGammaMat(self, orders, spacing=None, q=None):
    """Evaluate a term of the bending energy integral.

    Part of Shackleford et al. LNCS 7511 MICCAI 2012, Hadamard product
    of three integrals of given order, compute one of the full size
    tensor terms of the bending energy, corresponds to
    V^{\delta_x,\delta_y,\delta_z} of Eq. 18.

    Parameters
    ----------
    orders : list
        List of orders of derivatives, one for each of the 3 dimensions.
        For bending energy \delta_x+\delta_y+\delta_z=2, though this
        function will happily calculate other order combinations.
    spacing : float, optional
        Spacing of knots. If not supplied then originally set spacing.
    q : int, optional
        Spline degree, only q=3 (cubic) currently supported. If not
        supplied then originally set q.
        

    Returns
    -------
    bigGamma : ndarray
        Single term of bending energy tensor

    """
    if q is None:
      q=self.q
    if spacing is None:
      spacing=self.spacing
    gammaZ = self.buildGammaMat(order=orders[0])
    gammaY = self.buildGammaMat(order=orders[1])
    gammaX = self.buildGammaMat(order=orders[2])
    bigGamma = np.einsum("ai,bj,ck->abcijk",
                         gammaZ,
                         gammaY,
                         gammaX)
    return bigGamma.reshape(np.power(gammaZ.shape,3))


  def buildJ(self,deriv,spacing=None,q=None):
    """Evaluate a the full integral for the bending energy tensor on a
    supported interval.

    Implements Shackleford et al. Analytic Regularization of Uniform
    Cubic B-spline Deformation Fields LNCS 7511 MICCAI 2012 to
    calculate the matrix representing the bending energy integral as
    an inner product on control points.

    Parameters
    ----------
    spacing : float, optional
        Spacing of knots. If not supplied then originally set spacing.
    q : int, optional
        Spline degree, only q=3 (cubic) currently supported. If not
        supplied then originally set q.
    deriv : int, optional
        Derivative to use in energy term, range 0,...,q-1. Default=2
        is the thin plate bending energy, but 1 (to penalise gradient)
        or 0 (to penalise constants/coefficients) also possible.        
        

    Returns
    -------
    J : ndarray
        Tensor representing the bending energy integral for a supported
        volume as an inner product on the supporting coefficients.

    Notes
    ----------
    There is a discrepancy in Eq. 19 of Shackleford, the factor of 2
    on cross-derivatives used in bending energy is missing. This
    function includes it. (While arguably all cost functions are
    arbitrary, these correspond to cross terms of the frequency space,
    so are required for directional independence.)

    An alternative would be Wood 2016
    https://arxiv.org/abs/1605.02446v1 which sets up a similar thing
    using evaluation at a number of points of the splines to exactly
    fit the polynomial. This more easily adapts to q!=3, however it
    needs a bit of further work to allow the mixed d2/dxdy that
    Shackleford has and which we need for N3's thin-plate cost
    function, so sticking with this one. Or Shackleford's work could
    be extending to other q by generalising the B, D matrices in
    buildGammaMat to higher degree B-splines (as well as extending the
    integral terms vectors). https://www.plastimatch.org/ has already
    added a number of other energy terms using the same approach
    (though not necessarily for higher order splines).
    """
    # dist assumed isotropic so far, though relatively simple
    # to generalise if needed (mainly need to do type checking)
    if q is None:
      q=self.q
    if spacing is None:
      spacing=self.spacing
    if deriv is None:
      deriv = self.costDerivative

    if (deriv<0 or deriv>(q-1) or deriv%1 != 0):
      raise ValueError("deriv must be integer between 0 and q-1")
    
    from scipy.special import factorial
    J = None
    # fac: Confirmed with Gregory Sharp that Shackleford paper is
    # missing the 2* on the d2v/dxdy cross terms in eq. 18/19.
    # fac is the appropriate term from thin plate bending energy,
    # e.g. Wahba 1990, Spline Models for Observational Data,
    # eq. 12.1.5, or the Sled N3 paper eq. 26
    for dx in range(0,3):
      for dy in range(0,3):
        for dz in range(0,3):
          if (dx+dy+dz)==deriv:
            orders=(dx,dy,dz)
            # fac: see above
            fac = factorial(deriv)/np.prod(factorial([dx,dy,dz]))
            newJ = self.buildBigGammaMat(orders) * fac
            if J is None:
              J = newJ
            else:
              J += newJ
    return J

  @staticmethod
  def invertCoefList (coefList):
    """Convert a list of per-step cofficients into a by-interval list of
    coefficients for included steps.

    Parameters
    ----------
    coefList : list
        contains in order for each voxel location along the axis tuples of
        (index of first supporting knot, [q+1 control point coefficients])

    Returns
    -------
    Alocs : list

        List by indexed by supported interval (or first supporting
        coefficient) along axis, each element tuple: (first Voxel
        index, local support matrix A), where the local support matrix
        is nsteps * (q+1), the matrix of influences on each voxel step
        along the axis of each control point supporting this interval.

    """
    cIndList = [x[0] for x in coefList]
    q1 = coefList[0][1].shape[0]
    coefValList = np.array([x[1] for x in coefList]).reshape((-1,q1))
    firstC, voxinds, coefinds = np.unique(cIndList,
                                return_index=True, return_inverse=True)
    Alocs = [None] * (firstC.max()+1)
    for ind, (C,firstvox) in enumerate(zip(firstC, voxinds)):
      Alocs[C] = [firstvox, coefValList[coefinds==ind,:]]
    return Alocs


  @staticmethod
  def doubleKnots(kntsTuple, q):
    """Subdivide knot locations and provide updated coefficient number"""
    nCoef=kntsTuple[0]
    oldKnts = kntsTuple[1]
    halfKnts = (oldKnts[1:] + oldKnts[:-1]) /2
    newKnts = np.empty((2*oldKnts.size-1))
    newKnts[0::2] = oldKnts
    newKnts[1::2] = halfKnts
    newKnts = newKnts[q:-q]
    nCoef = 2 * nCoef - q
    return (nCoef, newKnts)


  def promote(self, reportingLevel=0):
    """Generate a new spline model with knot spacing halved

    Generates a new SplineSmooth3D instance, with intermediate knots
    inserted. If coefficients have already been solved then rescaled
    coefficients are calculated to produce an equivalent model.
        If current model has not been fitted and solved then the finer
    mesh only is built. This may be useful if promoting interpolators
    and fitters in step.

    Returns
    -------
    SplineSmooth3D

    """
    newSpline = self.__class__(
      data = self.data,
      voxsizes = self.voxsizes,
      spacing = self.spacing, # Not yet...
      mask = self.mask,
      q = self.q,
      domainMethod = self.domainMethod,
      Lambda = None, # No point calculating yet
      dofit = False
    )
    newSpline.promotion = self.promotion + 1
    newSpline.spacing = self.spacing / 2.0
    newSpline.Lambda= self.Lambda
    newSpline.mincLambda = self.mincLambda
    newSpline.voxelsLambda = self.voxelsLambda

    newKntsArr = [ self.doubleKnots(knts,self.q)
                          for knts in self.kntsArr ]
    newSpline.setupKCP(newKntsArr)

    # I think this even-odd convolution trick works
    # for all spline orders on uniform knots, due to
    # the symmetry of odd and even polynomial terms,
    # however extending it requires calculating the
    # contribution coefficients for the higher spline
    # orders which I cannot find set out analytically
    # anywhere, would prefer not to calculate recursively
    # and am not currently motivated to derive.

    if self.P is not None:
      Pshape = np.array([ kntsTuple[0] for kntsTuple in self.kntsArr])
      Pview = self.P.reshape(Pshape)
      Pexpand = np.zeros(2*Pshape+1)
      Pexpand[1::2,1::2,1::2] = Pview
      # q+2=5 is correct for q=3, think it's also correct for
      # q!=3 cases, corresponds to polynomials +1 for overlap
      # to next basis.
      coefConv = np.full(self.q+2,np.nan)
      coefConv[0::2] = np.array([1.0,6.0,1.0])/8
      coefConv[1::2] = np.array([1.0,1.0])/2
      coefConv3D = np.einsum("i,j,k->ijk",
                             coefConv,coefConv,coefConv)
      dropEnds = (self.q+1)//2
      # Attention! scipy convolve doesn't truly convolve as
      # the convolving function isn't reversed. However, we're
      # using a symmetric function anyway.
      # Unlike numpy (1D) convolve function it doesn't offer
      # output range control either, so we have to trim ends
      # ourselves (full, same, valid - last is the one we want).
      # Old versions of scipy (e.g. 0.14) defined (or at least
      # documented) origin differently, so be sure to look at
      # correct documentation, in 1.3.3 the middle element of
      # the filter is positioned on each input element with
      # origin=0. We use a NaN filled padding to make any
      # errors obvious.
      Pexpand = ndimage.convolve(Pexpand,coefConv3D,
                                 mode="constant",cval=np.nan)
      Pexpand = Pexpand[dropEnds:-dropEnds,
                        dropEnds:-dropEnds,
                        dropEnds:-dropEnds]
      newSpline.P = Pexpand.reshape(-1)
    return newSpline



class SplineSmooth3DUnregularized(SplineSmooth3D):
  """3D smoothing tensor B-spline class for fields defined as numpy
  arrays, fitting using Lee 1997 https://doi.org/10.1109/2945.620490
  scheme for multi-level splines

  *Experimental* Same methods as SplineSmooth3D, except that Atx and
  AtA are used differently, Atx is Lee's delta array, AtA is the omega
  array.

  """
  
  def __init__(self,data,voxsizes,spacing,mask=None,
               q=3,
               domainMethod="centre",
               Lambda=None, mincLambda=True, voxelsLambda=False,
               costDerivative=2,
               dofit=True):
    super(SplineSmooth3DUnregularized,self).__init__(
      data,voxsizes,spacing,mask=mask,
      q=q,
      domainMethod=domainMethod,
      Lambda=None, mincLambda=True, voxelsLambda=False, dofit=False)
    self.omegaWeights=None
    if(dofit):
      self.fit()
    return None
    
    

  def fit(self,data=None, reportingLevel=0):
    """fit current data, or update current data and fit

    fit does not solve for parameters, it calculates the projection of
    data to parameter space `Atx`, and if necessary the normalising
    array omega (as `AtA`). This means re-using fit for updated data
    is quicker than using a fresh instance. However omega (`AtA`)
    depends on the shape of data, the mask and the knot placement, a
    new instance is required if any of these are to be changed.

    Running fit will invalidate the current parameters solution `P`.

    Parameters
    ----------
    data : ndarray : optional
        New data to fit, if no new data then will fit the previously
        supplied data.
    reportingLevel : int, optional
        If >=1 then report time taken for the fitting loop, if >=2
        then will also report time for each inner loop (each interval)

    Returns
    -------
    AtA : ndarray
        Parameter weighting array
    Atx : ndarray
        Projection of data to parameter space

    """
    mask = self.mask
    if (data is None):
      data = self.data
    if (not data.shape == self.data.shape):
      raise ValueError("Shape of data passed to fit must"+
                       "match that of data used to initialise")
    self.data=data
    self.P=None # Invalidate P
    totalPar = self.totalPar
    needWeights = self.AtA is None or self.omegaWeights is None
    if needWeights:
      self.AtA = np.zeros((totalPar))
      self.omegaWeights = np.zeros(data.shape)
    omegaWeights = self.omegaWeights
    self.Atx = np.zeros(totalPar)
    AtA = self.AtA
    AtAflat = AtA
    Atx = self.Atx
    kntsArr = self.kntsArr
    invCoefArr = self.invCoefArr

    t_start=time.time()
    t_last=t_start

    q1=self.q+1
    q13 = q1**3

    # optimal path, numpy 1.12 is slower for our case with just
    # True setting, numpy 1.16 is okay with both (and significantly
    # faster too)
    optimize="optimal"
    phiSumOptPath=None
    voxWeightsOptPath=None
    cpWeightsOptPath=None

    if needWeights:
      for cIndZ,cIndY,cIndX, \
        coefsZ,coefsY,coefsX, \
        rangeZ,rangeY,rangeX, \
        tgtinds in self.coefIntervalIter(reportingLevel):
        localMask = mask[rangeZ[0]:rangeZ[1],
                         rangeY[0]:rangeY[1],
                         rangeX[0]:rangeX[1]]
        coefsZ2 = np.square(coefsZ)
        coefsY2 = np.square(coefsY)
        coefsX2 = np.square(coefsX)
        voxWeightSum = "xc,yb,za,zyx->zyx"

        if voxWeightsOptPath is None:
          voxWeightsOptPath = np.einsum_path(voxWeightSum,
            coefsX2,coefsY2,coefsZ2,
            localMask, optimize=optimize )[0]
        voxWeights = np.einsum(voxWeightSum,
                               coefsX2,coefsY2,coefsZ2,
                               localMask, optimize=voxWeightsOptPath)
        nonzero = voxWeights!=0
        voxWeights[nonzero] = 1/voxWeights[nonzero]
        omegaWeights[rangeZ[0]:rangeZ[1],
                     rangeY[0]:rangeY[1],
                     rangeX[0]:rangeX[1]] = voxWeights
        cpWeightSum = "xc,yb,za,zyx->abc"
        if cpWeightsOptPath is None:
          cpWeightsOptPath = np.einsum_path(cpWeightSum,
            coefsX2,coefsY2,coefsZ2,
            localMask, optimize=optimize)[0]
        cpWeights = np.einsum(cpWeightSum,
                               coefsX2,coefsY2,coefsZ2,
                               localMask, optimize=cpWeightsOptPath)
        AtAflat[tgtinds] += cpWeights.reshape(-1)

    for cIndZ,cIndY,cIndX, \
    coefsZ,coefsY,coefsX, \
    rangeZ,rangeY,rangeX, \
    tgtinds in self.coefIntervalIter(reportingLevel):
      coefsZ3 = np.power(coefsZ,3)
      coefsY3 = np.power(coefsY,3)
      coefsX3 = np.power(coefsX,3)
      localData = data[rangeZ[0]:rangeZ[1],
                       rangeY[0]:rangeY[1],
                       rangeX[0]:rangeX[1]]
      voxWeights = omegaWeights[rangeZ[0]:rangeZ[1],
                                rangeY[0]:rangeY[1],
                                rangeX[0]:rangeX[1]]
      phiSum = "xc,yb,za,zyx,zyx->abc"
      if phiSumOptPath is None:
        phiSumOptPath = np.einsum_path(phiSum,
          coefsX3,coefsY3,coefsZ3,
          voxWeights, localData, optimize=optimize)[0]
      phi = np.einsum(
        phiSum,
        coefsX3,coefsY3,coefsZ3,
        voxWeights, localData, optimize=phiSumOptPath)

      Atx[tgtinds] += phi.reshape(-1)

    if reportingLevel >= 1:
      t_now = time.time()
      print(t_now-t_start)

    return AtA, Atx


  def solve(self, reportingLevel=0,
            Lambda=None, mincLambda=None, voxelsLambda=None):
    """determine paramters for current data fit

    solve will update the currently fitted parameters, if necessary
    running `fit()` first. Lambda only accepted for compatibility with
    SplineSmooth3D.solve(), unused.

    Parameters
    ----------

    Returns
    -------
    P : ndarray
        Fitted control point parameters, as vector

    """

    if self.AtA is None:
      fit(data)
    AtA = self.AtA
    Atx = self.Atx
    P = np.zeros(Atx.shape)
    nonzero = AtA != 0
    P[nonzero] = Atx[nonzero] / AtA[nonzero]
    self.P=P
    return self.P

  def solveMultiLevel(self,data=None, reportingLevel=0, levels=1):
    smoother = self
    if (data is None):
      data = self.data
    resid = data
    Plast = None
    for level in range(levels):
      if level != 0:
        # Promote at start of subsequent levels
        resid = data - smoother.predict()
        smoother = smoother.promote()
        # Save the promoted state
        Plast = np.copy(smoother.P)
      smoother.fit(resid)
      smoother.solve()
      if Plast is not None:
        smoother.P += Plast
    return smoother

