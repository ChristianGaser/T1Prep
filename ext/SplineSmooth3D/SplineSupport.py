import numpy as np

### _bspl is not part of the public SciPy interface and may be withdrawn.
### It is present in SciPy v1.3.0 and v1.3.1, evaluate_all_bspl documentation:
### https://github.com/scipy/scipy/blob/v1.3.0/scipy/interpolate/_bspl.pyx#L163
### k, what scipy calls spline order, is actually degree,
### see their example k=3 for cubic splines
from scipy.interpolate._bspl import evaluate_all_bspl

def eval_nonzero_bspl(tfull,x,q=3,nu=0):
  tindstart = np.searchsorted(tfull,x,side="right")-1
  coeffstart = tindstart-q
  intervals = tfull.size - 2*q - 1
  nonZeroCoeffs = evaluate_all_bspl(tfull,q,x,tindstart,nu=nu)
  while coeffstart >= intervals:
      # Edge case, last internal knot is technically unsupported
      # since we are supporting from the left edge of the range,
      # but has its last spline coefficient zero, so can trim
      # it off and report coefficients, with a padding zero at
      # the start instead.
      coeffstart = coeffstart - 1
      nonZeroCoeffs = np.concatenate(([0],nonZeroCoeffs[0:q]))
  while coeffstart < 0:
      coeffstart = coeffstart + 1
      nonZeroCoeffs = np.concatenate((nonZeroCoeffs[1:],[0]))
  return coeffstart, nonZeroCoeffs



def knots_over_domain(a,b,spacing,q=3, method="centre"):
# Calculate the uniform knots for a given spline
# order and spacing.
# methods:
# centre: place a knot in the middle of domain
#   and then knots outwards at spacing to cover the whole
#   domain
# forcerange: supported intervals over the supplied domain only,
#   if this isn't an exact fit then round up the required
#   intervals and apply to the domain (resulting spacing
#   will be reduced
# runover: supported interval starts at a and ends on the
#   first knot on or past b
# 
  ab = np.array([a,b]) # Because ints...
  if method == "centre" or method == "center":
    # Centre places a control point in the middle of the
    # range, so will always have an even number of intervals
    mid = np.average(ab)
    midDist = np.max(np.abs(mid-ab))
    halfIntervals = np.ceil(midDist/spacing).astype("int")
    nIntervals = 2 * halfIntervals
    internalEnds = mid + np.array([-1,1])*halfIntervals*spacing
  elif method == "forcerange":
    nIntervals = np.ceil((b-a)/spacing).astype("int")
    internalEnds = ab
  elif method == "runover":
    nIntervals = np.ceil((b-a)/spacing).astype("int")
    internalEnds = np.array([a,a+spacing*nIntervals])
  elif method == "minc":
    eps=1.0e-14 
    nIntervals = np.ceil((b-a)/(spacing*(1+eps))).astype("int")
    nBasis = nIntervals+3
    nKnots = nBasis+4
    start = 0.5*(a + b - spacing * (nBasis+3))
    knots = np.zeros(nKnots)
    for j in range(0,nKnots):
      knots[j] = start + spacing * j
    return nBasis, knots
  else:
    raise ValueError("Not a supported knots method: '{}'".format(method))
  nInternalKnots = nIntervals + 1
  nCoeffs = nIntervals + q
  tinternal = np.linspace(internalEnds[0],
                          internalEnds[1],
                          nInternalKnots)
  tend = np.linspace(spacing,spacing*(q-1),q)
  tfull = np.concatenate((tinternal[0]-np.flip(tend,0),
                          tinternal,
                          tinternal[-1]+tend))
  return nCoeffs, tfull
