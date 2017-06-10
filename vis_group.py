# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:16:31 2013

@author: chrisr
"""
import numpy as np
from scipy import io
import gcff

path = 'data/Synth/'
a=io.loadmat(path+'features.mat')
f=a['features'][0]
a=io.loadmat(path+'groundtruth') #'groundtruth_A' or #'groundtruth'

gt=a['GTgroups'][0]

##gt[i] is an array of arrays, each containing objects belonging to a certain class
## turn this into a labelling
## Some objects are missing in some frames
## Reindex these frames
oldgt=gt
gt=np.empty_like(oldgt)
for i in range(gt.shape[0]):
    g=oldgt[i].reshape(-1)
    #Transpose error from one dataset to another
    #size=f[i][:,0].size#(g.size<1) or max([(x.size<1) or x.max() for x in g])
    out=f[i][:,0].astype(np.int)
    if out.size>0:
        invert=np.ones(out.max()+1,dtype=np.int)*-1
        #invert[out]=np.arange(out.size-1,-1,-1)
        invert[out]=np.arange(out.size)
        array=np.arange(out.size)
        ## Labels don't mean anything but label l+1 occurs next 1 making visualisation difficult
        for j in range(g.shape[0]):        
            if g[j].size>0:
                g2=g[j].reshape(-1).astype(np.int)
                #Transpose error from one dataset to another
                temp=invert[g2]#array.size-invert[g2]-1
                array[temp]=temp[0]
        ##compress array
        _,a=np.unique(array,return_inverse=True)
        gt[i]=a

mask=np.empty(f.shape[0],dtype=np.bool)
for i in range(f.shape[0]):
    mask[i]=f[i].shape[0]>0

f=f[mask]
f=f[:gt.size]
gt=gt[mask[:gt.size]]


def vis(gt,f,est=False,title=False):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider #, Button, RadioButtons
    from matplotlib.patches import Ellipse

    fig = plt.figure()
    if title :
        fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = plt.subplot(111, aspect='equal')
    fig.subplots_adjust(left=0.25, bottom=0.25)
    ax.clear()

    axcolor = 'lightgoldenrodyellow'
    ax2 = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    
   
    def update(val):  
        val=int(val)
        ax.clear()
        loc=find_locs(f[val])
        if est is not False:
            p=np.random.permutation(est[val].max()+1)
            p2=np.random.permutation(gt[val].max()+1)
            calc_distance_vis(loc,f[val],p[est[val]],3500,ax)
        ax.scatter(f[val][:,1],f[val][:,2], c=p2[gt[val]],vmin=0, vmax=gt[val].max(),s=100)
    update(0)
    slider = Slider(ax2, 'Frame', 0, gt.shape[0] - 1,
                    valinit=0, valfmt='%i')

    slider.on_changed(update)

    plt.show()



#import segmentation

def calc_distance_vis(loc,f,labels,mdl,ax):
    from matplotlib.patches import Ellipse
    ax.scatter(f[:,1],f[:,2], c=labels,vmin=0, vmax=labels.max(),s=400)
    ax.plot(np.vstack((f[:,1],loc[:,0])),
            np.vstack((f[:,2],loc[:,1])),'g')
                
    u=np.unique(labels)
    dist=np.empty((loc.shape[0],u.shape[0]))
    dist2=np.zeros_like(dist)
    for i in range(u.shape[0]):
        means=loc[labels==i,:].mean(0)
        ells = Ellipse(means,np.sqrt(mdl), np.sqrt(mdl),0)
        ells.set_alpha(0.1)
        ax.add_artist(ells)
        
        dist[:,i]=((loc-means)**2).sum(1)
        mask=np.arange(loc.shape[0])[dist[:,i]<mdl]
        #means=means.T
        disp=f[:,1:3].copy()
        disp-=means
        for j in mask:
            for k in mask:
                distk=np.linalg.norm(disp[k])
                distj=np.linalg.norm(disp[j]) 
                if distk>distj:
                    inner=disp[k].dot(disp[j])
                    norm=distk*distj
                    if inner/norm>.5:
                        print (j,k,disp[j],disp[k])
                        print (distk,distj,inner,norm,distk/distj)
                        dist2[k,i]+=10**(distk/distj)
                        ax.plot(np.vstack((disp[k,0]+means[0],means[0])),
                                np.vstack((disp[k,1]+means[1],means[1])),'r')
                        ax.plot(np.vstack((disp[j,0]+means[0],means[0])),
                                np.vstack((disp[j,1]+means[1],means[1])),'b')    
        dist+=dist2
    return dist

def find_locs(f,stride=35):
    "Estimate focal centers for each person given features"
    locs=np.empty((f.shape[0],2))
    locs[:,0]=f[:,1]+np.cos(f[:,3])*stride
    locs[:,1]=f[:,2]+np.sin(f[:,3])*stride
    return locs


def make_est(f,stride=35,mdl=3500):
    """Solve entire sequence"""
    est=np.empty(f.shape[0],dtype=object)
    for i in range(f.shape[0]):
        est[i]=gcff.gc(f[i],stride,mdl)
    return est

est=make_est(f,stride=25,mdl=3000)

vis(gt,f,est,"Inner circles=GT, Outer=Est")
