'''
Perform Kolmogorov-Smirnov test on each voxel of wild bootstrapped data
returns pvalue map.

Created on 14 Jan 2013

@author: gfagiolo
'''
import os, sys
import multiprocessing 
from datetime import datetime

import scipy.stats as stats

import wildbootstrapAtom as wba
#this uses either nibabel or pynii to write nifti to disk

def poolKSTestFunc(frame):    
    """perform KS test and returns p-value for each voxel"""
    nvox = frame.DATA.shape[0]
    output = wba.np.zeros(nvox)
    for v in range(nvox):
        d = frame.DATA[v,:]
        output[v] = stats.kstest((d-d.mean())/d.std(), 'norm')[-1]
    return output

def ksTest(fname, nprocesses=None, use_nibabel=False):
    print 'Performing KS test and producing pval map for', fname    
    ddir = os.path.split(fname)[0]
    wbdata = nb.load(fname)
    affine = wbdata.get_affine()
    wbdata = wbdata.get_data()
    try:
        mask = nb.load(os.path.join(ddir, 'mask.nii.gz')).get_data()
    except Exception as e:
        print 'WARNING: no mask.nii.gz file found, using all voxels', e
        mask = wba.np.ones(wbdata.shape[:-1])
    t0 = datetime.now()
    pool = multiprocessing.Pool(nprocesses)
    mpI = wba.WildbootstrapImage(wbdata, mask)
    partitioned = mpI.partitionForPool()
    pool_result = pool.map(poolKSTestFunc, partitioned)
    dt = datetime.now()-t0
    se = dt.seconds+1.e-6*dt.microseconds
    print 'Computations done in %.2f secs (%.1f voxels/sec)'%(se, len(wba.np.flatnonzero(mask))/se)
    pool.close()
    pval =  mpI.recombineFromPool(pool_result, partitioned)
    if use_nibabel:
        import nibabel as nb
        nb.save(nb.Nifti1Image(pval, affine), fname.replace('.nii','.ks.pval.nii'))
    else:
        import pynii
        img = pynii.Nifti1Data()
        img.setAffine(affine)
        img.setData(pval)
        img.write(fname.replace('.nii','.ks.pval.nii'))
        
    print 'job done'
    
if __name__ == '__main__':
    if sys.argv > 1:
        for fname in sys.argv[1:]:
            ksTest(fname)
    else:
        print 'use:',sys.argv[0],' wildboostrapped_data'
        print 'optionally a mask.nii.gz file could be placed where the data is strored'