import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm,tqdm_notebook
    
class CheckInfo:
	def __init__(self):
		print('initializing checkinfo')
		self.test_pass = True

def run(archive,method,options):
	import sys

	this_module = sys.modules[__name__]
	method = getattr(this_module, method)
	
	return method(archive,options)
	
def grid_gradient(archive,options):	
	print('grid_gradient')

	rel_axes=list()
	for x_list in archive.axes_values:
		dx=x_list[-1]-x_list[0]
		x0 = x_list[0]
		z=list()
		for x in x_list:
			z.append((x-x0)/dx)
		rel_axes.append(z)
	print(rel_axes)

	deriv=list()
	rhist=list()
	ahist=list()
	khist=list()
	for k in tqdm(range(archive.ncoeff)):
		y=archive.coeff[...,k]
		yp=np.asarray(np.gradient(y,*rel_axes))
		n=len(archive.axes_shape)
		pack=[None]*n
		for i in range(n):
			max0 = np.amax( yp[i,...] )
			min0 = np.amin( yp[i,...] )
			eps0 = options.eps0
			max_y = np.amax(y)
			if max_y<=0:
				max_y = options.eps0
			ypr = yp[i,...]/max_y
			rmax0 = np.amax( ypr )
			rmin0 = np.amin( ypr )
			diff_list=list()
			for j in range(n):
				ydr = np.absolute(np.diff(ypr,axis=j))
				yd = np.absolute(np.diff(yp[i,...],axis=j))
				vr = np.amax(ydr)
				diff_list.append( vr )
				rhist.append( vr )
				va = np.amax(yd)
				ahist.append( va )
				khist.append( k )
			pack[i]={'min':min0,'max':max0,'rmin':rmin0,'rmax':rmax0, 'diff': diff_list}
		deriv.append(pack)
	ahist = np.asarray(ahist)
	rhist = np.asarray(rhist)
	khist = np.asarray(khist)
	
	info = CheckInfo()
	info.ahist = ahist
	info.rhist = rhist
	info.khist = khist
	info.rel_axes = rel_axes
	info.epsa = options.epsa
	info.eps0 = options.eps0
	info.epsr = options.epsr
	info.wa = np.logical_and( (ahist>options.epsa), (rhist>options.epsr)).sum()
	info.wr = (rhist>options.epsr).sum()
	info.m = len(ahist)
	info.q1 = 1.0 - info.wr/info.m 
	info.q2 = 1.0 - 0.9*info.wa/info.m - 0.1*info.wr/info.m

	return info

def plot_grid_gradient_hist(info):
	plt.hist2d(np.log10(info.rhist),np.log10(info.ahist),bins=np.linspace(-40,20,100),cmin=1,alpha=0.2)
	ind1 = (info.rhist>info.epsr) & (info.ahist>info.epsa)
	h=plt.hist2d(np.log10(info.rhist[ind1]),np.log10(info.ahist[ind1]),bins=np.linspace(-40,20,100),cmin=1,alpha=1.0)
	ind2 = (info.rhist>info.epsr)
	plt.hist2d(np.log10(info.rhist[ind2]),np.log10(info.ahist[ind2]),bins=np.linspace(-40,20,100),cmin=1,alpha=0.6)
	plt.colorbar(h[3])
	plt.xlabel(r'$\log \tilde{h}_{ijk}$')
	plt.ylabel(r'$\log h_{ijk}$')
	plt.grid()
	plt.show()

def nonnegative_interp(archive,options):
	print('nonnegative_interp')
	info = CheckInfo()
	return info
