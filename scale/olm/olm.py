import sys
import h5py
import numpy as np
from tqdm import tqdm,tqdm_notebook
import check as check

def main(args):
    mode = args[0]
    print('main')
    try:
        this_module = sys.modules[__name__]
        mode_module = getattr(this_module,mode)
        archive,directives = parse(args[1:])
        p = 0
        for d in directives:
            info = mode_module.run(archive,d['method'],d['options'])
            print('q1:',info.q1,'q2:',info.q2)
            if not info.test_pass:
                p = 1
        return p

    except ValueError as ve:
        return str(ve)

def parse(args):
    print('parse')
    options = type('', (), {})()
    options.eps0=1e-20
    options.epsa=1e-1
    options.epsr=1e-1
    
    gg={'method': 'grid_gradient', 'options': options}
    return Archive(args[-1]),[gg]

def get_indices(axes_names,axes_values,point_data):
    y=[0]*len(point_data)
    for name in point_data:
        i=np.flatnonzero(axes_names==name)[0]
        iaxis = axes_values[i]
        value = point_data[name]
        diff = np.absolute(axes_values[i]-value)
        j=np.argmin(diff)
        #print(name,i,j)
        y[i] = j
    return tuple(y)

def extract_axes(h5):
    data=h5['incident']['neutron']
    dim_names=list()
    libs=list()
    ncoeff=0
    nvec=0
    for i in data.keys():
        if i!='TransitionStructure':
            libs.append( data[i] )
            ncoeff=np.shape( data[i]['matrix'])[1]
            nvec=np.shape( data[i]['loss_xs'])[1]
            labels = data[i]['tags']['continuous']
            for x in labels:
                dim_names.append(x)
    dim_names = list(np.unique(np.asarray(dim_names)))
    #print(dim_names)
    #print(libs)

    # create 1d dimensions array
    n=len(libs)
    dims=dict()
    for name in dim_names:
        dims[name]=np.zeros(n)
    times=list()
    D=6
    for i in range(n):
        for name in libs[i]['tags']['continuous']:
            value = libs[i]['tags']['continuous'][name]
            dims[name][i]=value[()].round(decimals=D)
        times.append( np.asarray(libs[i]['burnups']).round(decimals=D) )
    #print(dims)
    #print(times)

    # determine values in each dimension and add time
    axes_names=list(dim_names)
    ndims=len(dim_names) + 1
    axes_values=[0]*ndims
    i=0
    for name in dims:
        axes_values[i] = np.unique(dims[name].round(decimals=D))
        i+=1
    axes_names.append('times')
    axes_values[i]=times[0]
    
    #print('axes:',axes)
    #print('axes_names',axes_names)

    # determine the shape/size of each dimension
    axes_shape = list(axes_values)
    for i in range(ndims):
        axes_shape[i] = len(axes_values[i])

    # convert names and shapes to np array before leaving
    axes_names = np.asarray(axes_names)
    axes_shape = np.asarray(axes_shape)
    return axes_names,axes_values,axes_shape,ncoeff,nvec

class Archive:
    def __init__(self,file):
        print('loading archive file:',file)
        
        self.file_name = file
        self.h5 = h5py.File(file, 'r')
        self.axes_names,self.axes_values,self.axes_shape,self.ncoeff,self.nvec = extract_axes(self.h5)

        # populate coefficient data
        self.coeff=np.zeros((*self.axes_shape,self.ncoeff))
        data = self.h5['incident']['neutron']
        for i in tqdm(data.keys()):
            if i!='TransitionStructure':
                d=get_indices(self.axes_names, self.axes_values,data[i]['tags']['continuous'])
                dn=(*d,slice(None),slice(None))
                self.coeff[dn]=data[i]['matrix']

        # disallow single point dimensions for gradient calcs
        n=len(self.axes_shape)
        for i in range(n):
            if self.axes_shape[i]==1:
                self.axes_shape[i] = 2
                x0 = self.axes_values[i][0]
                if x0==0.0:
                    x1=0.05
                else:
                    x1=1.05*x0
                self.axes_values[i] = np.append(self.axes_values[i],x1)
                coeff = np.copy(self.coeff)
                self.coeff=np.repeat(self.coeff,2,axis=i)