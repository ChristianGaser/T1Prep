import numpy as np

def rearrAx2(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[2]/2)
  # Earlier numpy.flip only one axis at a time
  return np.transpose(np.flip(np.flip(vol[:,:,sl],0),1))

def rearrAx1(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[1]/2)
  # Earlier numpy.flip only one axis at a time
  return np.transpose(np.flip(np.flip(vol[:,sl,:],0),1))

def rearrAx0(vol, sl=None) :
  if sl is None :
    sl = int(vol.shape[0]/2)
  # Earlier numpy.flip only one axis at a time
  return np.flip(np.flip(vol[sl,:,:],0),1)


def getrarrAxSel(ax=2):
  if ax == 0 :
    rearrSel = rearrAx0
  elif ax == 1 :
    rearrSel = rearrAx1
  elif ax == 2 :
    rearrSel = rearrAx2
  return rearrSel


def imgplot(plotobj, data, vmin=None, vmax=None, cmap='gray'):
  if not vmin:
    vmin = np.percentile(data,2.5)
  if not vmax:
    vmax = np.percentile(data,97.5)
  plotobj.matshow(data,cmap=cmap,vmin=vmin,vmax=vmax)
  return plotobj.axis('off')
