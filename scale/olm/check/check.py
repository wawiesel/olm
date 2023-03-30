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

def calc_grid_gradient_kernel(rel_axes,yreshape,eps0):
	n=len(rel_axes)
	ncoeff = np.shape(yreshape)[0]
	rhist=np.zeros(n*n*ncoeff)
	ahist=np.zeros(n*n*ncoeff)
	khist=np.zeros(n*n*ncoeff)

	for k in tqdm(range(ncoeff)):
		y=yreshape[k,...]
		max_y = np.amax(y)
		if max_y<=0:
			max_y = eps0
		yp=np.asarray(np.gradient(y,*rel_axes))

		for i in range(n):
			ypi = yp[i,...]
			for j in range(n):
				iu = k*n*n + i*n + j

				ydr = np.absolute(np.diff(ypi,axis=j))/max_y
				rhist[iu]=np.amax(ydr)

				yda = np.absolute(np.diff(ypi,axis=j))
				ahist[iu]=np.amax(yda)

				khist[iu]=k

	return ahist,rhist,khist

def calc_grid_gradient(archive,options):
	print('calc_grid_gradient')

	rel_axes=list()
	for x_list in archive.axes_values:
		dx=x_list[-1]-x_list[0]
		x0 = x_list[0]
		z=list()
		for x in x_list:
			z.append((x-x0)/dx)
		rel_axes.append(z)

	yreshape=np.moveaxis(archive.coeff,[-1],[0])

	info = CheckInfo()
	info.rel_axes = rel_axes
	info.epsa = options.epsa
	info.eps0 = options.eps0
	info.epsr = options.epsr

	info.ahist,info.rhist,info.khist = calc_grid_gradient_kernel(rel_axes,yreshape,info.eps0)

	return info

def grid_gradient(archive,options):
	info = calc_grid_gradient(archive,options)

	info.wa = np.logical_and( (info.ahist>info.epsa), (info.rhist>info.epsr)).sum()
	info.wr = (info.rhist>info.epsr).sum()
	info.m = len(info.ahist)
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
