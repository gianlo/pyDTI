"""
A class to work on DTI acquisition saved in nifti and bvecs/bvals in FSL formats. 
It includes a multicore implementation of wildbootstrapping for FA

The MIT License

Copyright (c) 2013 Gianlorenzo Fagiolo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
from datetime import datetime
import multiprocessing

# import nibabel as nb
import pynii

__version__ = "0.90beta"

from helperFunctions import normalize_vectors, decompose_aff, \
    generate_coin_tosses, compute_log_signal, compute_FA, \
    eigvalues_vectors_from_tfit, autoMask
import wildbootstrapAtom as wba

#numpy is already imported once
np = wba.np
     
class FSLDTI(object):
    """FSL DTI volume class
    It's recommended that the data is preprocessed by FSL
    1) BET to generate brain MASK (i.e. signal ROI)
    2) eddy-current correction (i.e. to produce affine correction file 'data.ecclog')
    BVECTORS is a either a filename or an array containing the b-vecs
    BVALUES is a either a filename or an array containing the b-vals
    filename is the nifti filename containing the DTI data
    NUMBER_OF_BOOTSTRAPS is the number of bootstraps performed by the wildboostrap routine
"""
    FILENAME = None
    PATH = None
    VERBOSE = False
    DIMS = None
    BVECTORS = None
    NEW_BVECTORS = None
    BVECTORS_FILENAME = None
    BVALUES = None
    BVALUES_FILENAME = None
    NIFTI = None
    IMG_DATA = None
    MASK = None
    MASK_FILENAME = None
    NUMBER_OF_BOOTSTRAPS = None
    ROTATE_BVECS_MATRICES = None
    SIMPLETENSOR = None
    DIFFUSION_TENSOR_MAP = None
    FA_MAP = None
    FA_STD_MAP = None
    MD_MAP = None
    MD_STD_MAP = None
    CHANGE_NONPOSITIVEVALUES = False
    #temporary
    NOT_WARNED_YET = True
    
    def __init__(self, filename, bvecs=None, bvals=None, nboots=None, maskname=None, changenonpositivevalues=True, verbose=None):
        """This instances an object and reads various files"""
        self.FILENAME = filename
        #detect PATH of data file
        if filename:
            self.PATH = os.path.split(filename)[0]
            self.loadNiftiData()
        if not verbose is None:
            self.VERBOSE = verbose
        if bvecs:
            if isinstance(bvecs, np.ndarray):
                self.BVECTORS = bvecs
            elif isinstance(bvecs, str) or isinstance(bvecs, unicode):
                self.loadBVectors(bvecs)
#        else:
#            self.loadBVectors()
        if bvals:
            if isinstance(bvals, np.ndarray):
                self.BVALUES = bvals
            elif isinstance(bvals, str) or isinstance(bvals, unicode):
                self.loadBValues(bvals)
#        else:
#            self.loadBValues()
        if maskname is not None:
            self.loadMask(maskname)
        if not nboots is None:
            self.NUMBER_OF_BOOTSTRAPS = nboots
        
        self.CHANGE_NONPOSITIVEVALUES = changenonpositivevalues
            
    def getB0Image(self):
        return self.IMG_DATA[:, :, :, 0]
    
    def getNifti(self):
        return self.NIFTI
    
    def getNiftiFilename(self):
        return self.FILENAME
    
    def getImageData(self):
        return self.IMG_DATA
    
    def getNiftiAffine(self):
        return self.getNifti().get_affine()
    
    def getNiftiHeader(self):
        return self.getNifti().get_header()

    def getBValues(self):
        return self.BVALUES

    def getBValuesFilename(self):
        return self.BVALUES_FILENAME
    
    def getBVectors(self):
        return self.BVECTORS
            
    def getBVectorsFilename(self):
        return self.BVECTORS_FILENAME

    def getVolDims(self):
        return self.DIMS[:-1]
    
    def getSignalLength(self):
        return self.DIMS[-1]
                
    def loadNiftiData(self, filename=None):
        """Load dti volume data from nifti file"""
        if filename is None:
            filename = self.FILENAME
        else:
            #reset bvalues and bvectors!
            self.FILENAME = filename
            self.BVECTORS = None
            self.NEW_BVECTORS = None
            self.BVALUES = None
                                    
        if os.path.exists(filename) and \
            (filename.endswith('nii') or filename.endswith('nii.gz')):
            #self.NIFTI = nb.load(filename)
            self.NIFTI = pynii.Nifti1Data.load(filename)
            self.IMG_DATA = self.NIFTI.get_data()
            #self.NIFTI=nii.NiftiImage(filename)
            self.DIMS = self.IMG_DATA.shape
            #turn any non-positive value to 1 (since we're fitting log(data) )
            if self.CHANGE_NONPOSITIVEVALUES:
                self.changeNonPositiveValues()
            if self.VERBOSE:
                print 'cleansed data from non-positive values'

    def loadBValues(self, bvals_fname):
        """Loads b-values from BVALUES_FILENAME"""
        if os.path.exists(bvals_fname):
            self.BVALUES = np.loadtxt(bvals_fname)
            self.BVALUES_FILENAME = bvals_fname            

    def loadBVectors(self, bvecs_fname):
        """Loads b-vecs from BVALUES_FILENAME"""
        if os.path.exists(bvecs_fname):            
            self.BVECTORS = normalize_vectors(np.loadtxt(bvecs_fname))
            self.BVECTORS_FILENAME = bvecs_fname            

    def loadDataEddyCurrentCorrectionLog(self, data_ecc_log):
        """Load registration information generated by the FSL eddy-current correction"""
        ecclogname = os.path.join(self.PATH, data_ecc_log)
        if os.path.exists(ecclogname):
            fin = open(ecclogname)
            lines = fin.readlines()
            fin.close()
            self.ROTATE_BVECS_MATRICES = []
            aff_mats = []
            c = 0
            while c < len(lines):
                line = lines[c]
                c = c+1
                if line.startswith('Final'):
                    mat = np.zeros((4,4))
                    mat[3,:] = np.array([0,0,0,1])
                    #read affine matrix 4x4
                    line = lines[c][:-1]
                    c = c+1
                    mat[0,:] = np.array(line.split(), dtype = np.float64)
                    line = lines[c][:-1]
                    c = c+1
                    mat[1,:] = np.array(line.split(), dtype = np.float64)
                    line = lines[c][:-1]
                    c = c+1
                    mat[2,:] = np.array(line.split(), dtype = np.float64)
                    aff_mats.append(mat)
                    #just get the rotation part of the affine transformation
                    self.ROTATE_BVECS_MATRICES.append(
                        decompose_aff(mat, [0,0,0])[0])
            return aff_mats

    def correctBVectors(self):
        """Corrects the b-vecs """
        if self.BVECTORS is None:
            return None
        if self.ROTATE_BVECS_MATRICES is None:
            print 'Load data_ecc_log resulting from eddycurrent correction with the method loadDataEddyCurrentCorrectionLog'
            return None
        self.NEW_BVECTORS = np.zeros(self.BVECTORS.shape)
        if len(self.NEW_BVECTORS.T) != len(self.ROTATE_BVECS_MATRICES):
            print 'WARNING: eddy current correction file info differs from BVECTORS',\
                len(self.ROTATE_BVECS_MATRICES), len(self.NEW_BVECTORS.T)
            print 'using reverse order'
            #raise ValueError
            for n in range(len(self.NEW_BVECTORS.T)):
                self.NEW_BVECTORS[:, -n] = np.dot(self.ROTATE_BVECS_MATRICES[-n], self.BVECTORS[:,n])
        else:
            for n, mat in enumerate(self.ROTATE_BVECS_MATRICES):
                self.NEW_BVECTORS[:, n] = np.dot(mat, self.BVECTORS[:,n])
    
    def loadMask(self, maskname):
        """Loads the brain-MASK from a nifti file following FSL naming convention or from maskname"""
        if os.path.exists(maskname):
            mask = nb.load(maskname)
            if mask.shape == self.DIMS[:-1]:
                self.MASK = np.asarray(mask.get_data(), np.int8)
                self.MASK_FILENAME = maskname
        return None

    def computeCentralMask(self, perc_radius=0.95):
        self.MASK = autoMask(self.getB0Image(), perc_radius)
        
    def findCorruptedDirections(self, sample_size=10, perc_radius=0.9):
        def nc(do, sn, dn):
            if do.MASK is None:
                do.computeCentralMask()
            data = do.getImageData()
            selSet = np.intersect1d(np.flatnonzero(do.MASK[:,:,sn].squeeze()),
                        np.flatnonzero(do.MASK[:,:,sn+1].squeeze()))
            a = data[:,:,sn,dn].squeeze().flatten()[selSet]
            b = data[:,:,sn+1,dn].squeeze().flatten()[selSet]
            return np.sqrt(np.dot(a,b))/np.linalg.norm(a)/np.linalg.norm(b)
        if self.MASK is None:
            # a mask is needed, if not there, then estimate one
            #90% of volume should be alright
            self.computeCentralMask(perc_radius)
        #find mask's slices area along the through plane direction
        masksliceareas = np.sum(np.sum(self.MASK, 0), 0)
        #find locz: the location of the median(area) slice (through plane)
        med = np.median(masksliceareas)
        locz = np.flatnonzero(masksliceareas == med)[0]
        #save samples
        dims = (sample_size, 3, self.MASK.shape[2], self.getSignalLength())
        sampled_values = np.zeros(dims)
        #loop over directions        
        for ndir in range(self.getSignalLength()):
#            print ndir
            #take sample_size random points in the mask at locz
            spoints = np.random.permutation(np.flatnonzero(self.MASK[:, :, locz]))[:sample_size]
            #convert points into 2D mask indexes
            xpoints = np.unravel_index(spoints, self.MASK.shape[:2])
#            spoints = np.ravel_multi_index(spoints  + (sample_size*[locz], ), self.MASK.shape)
            for ix in range(sample_size):
                u = xpoints[0][ix]
                v = xpoints[1][ix]
                cdata = self.IMG_DATA[u, v, :, ndir]*self.MASK[u, v, :]
                locations = np.flatnonzero(self.MASK[u, v, :])
                #save locations
                sampled_values[ix, 0, :, ndir].flat[locations] = spoints[ix]
                sampled_values[ix, 1, :, ndir].flat[locations] = 1 
                #save values
                sampled_values[ix, 2, :, ndir].flat[locations] = cdata
#                me = cdata.mean()
#                st = cdata.std()
#                sample_values[ix] = 100.*st/me 
#                print "%d %.0f %.0f %.1f"%(ix, me, st, 100.*st/me)
             
#            print "%.1f %.1f"%(sample_cv.mean(), sample_cv.std())
        return sampled_values
                        
    def changeNonPositiveValues(self):
        'turn any non-positive value in IM_DATA to 1'
        selSet = np.flatnonzero(self.IMG_DATA<=0)
#        if self.MASK is not None:
#            selSet = np.setdiff1d(selSet, np.flatnonzero(self.MASK == 0))
        self.IMG_DATA.flat[selSet] = 1
        self.CHANGE_NONPOSITIVEVALUES = True

    def computeWildbootstrapFA(self, outname, nbootstraps):
        "generates a random wildbootstrap FA map"
        ####THIS IS NOT MAINTAINED#######
        dims = self.getVolDims() + (nbootstraps,)
        nbs = nbootstraps
        FA_map = np.zeros(dims)
        tosses = generate_coin_tosses(self.getSignalLength(), nbs)
        dwinfo = wba.DiffusionWeightingInfo(self.BVALUES, self.BVECTORS)
        t0 = datetime.now()
        print 'Starting wild boostrapping of FA (%d voxels)'%(len(np.flatnonzero(self.MASK)))
        for k in range(self.DIMS[2]):
            slice_voxels = len(np.flatnonzero(self.MASK[:, :, k]))
            if slice_voxels < 1:
                print 'Skipping slice', k+1, '(%d voxels)'%(slice_voxels)
                continue
            print 'Processing slice', k+1, '(%d voxels)'%(slice_voxels),
            for i in range(self.DIMS[0]):
                for j in range(self.DIMS[1]):
                    if not self.MASK is None and self.MASK[i, j, k] == 0:
                        continue
#                    self.modifyNonPositiveVoxel(i, j, k)
                    wbinit = wba.WildbootstrapInit(dwinfo, compute_log_signal(self.IMG_DATA[i, j, k,:]))
                    try:
                        bstrapsout = wba.wildboostrapVoxelData(wbinit, dwinfo, tosses)
                        sampleFAS = np.zeros(nbs)
                        for cbs in range(nbs):
                            sampleFAS[cbs] = compute_FA(eigvalues_vectors_from_tfit(bstrapsout[cbs, :]))
                    except np.linalg.LinAlgError as e:
                        print 'ERROR'
                        print e, 'at voxel', i, j ,k
                        raise e
                    except RuntimeWarning as rw:
                        print 'ERROR'
                        print rw, 'at voxel', i, j ,k
                        raise rw
                    FA_map[i, j, k, :] = sampleFAS
            t1 = datetime.now()
            elapsed_secs = (t1-t0).seconds+1.e-6*(t1-t0).microseconds
            print '%d secs elapsed, %.2f voxels/sec'%(elapsed_secs, float(slice_voxels)/elapsed_secs)                    
#         FA = nb.Nifti1Image(FA_map, self.NIFTI.get_affine())
#         nb.save(FA, outname)
        FA = pynii.Nifti1Data()
        FA.setData(FA_map)
        FA.setAffine(self.NIFTI.get_affine())
        FA.write(outname)
        
    def computeWildbootstrapFA_multicore(self, outname, nbootstraps, nprocess=None, weighted=True):
        partitioned = self.partitionWildboostrapForPool(nbootstraps, weighted=weighted)        
        pool = multiprocessing.Pool(nprocess)
        print 'Bootstrapping FA on multiple processes: Job started'
        t0 = datetime.now()
        z = pool.map(poolProcessData, partitioned)
        dt = datetime.now()-t0
        se = dt.seconds+1.e-6*dt.microseconds
        rate = len(np.flatnonzero(self.MASK))/se
        if nbootstraps > 0:
            print 'Computations done in %.2f secs (%.1f voxels/sec) (%.1f x 1e3 (Voxels x NDWI x NSAMPLES)/sec)'%(se, 
                        rate, rate*self.IMG_DATA.shape[-1]*nbootstraps*1.e-3)
        else:
            print 'Computations done in %.2f secs (%.1f voxels/sec) (%.1f x 1e3 (Voxels x NDWI)/sec)'%(se, 
                        rate, rate*self.IMG_DATA.shape[-1]*1.e-3)        
        pool.close()
#        del(pool)
        nii = pynii.Nifti1Data()
        nii.setData(self.recombineWildbootstrapFromPool(z, partitioned))
        nii.setAffine(self.NIFTI.get_affine())
        nii.write(outname)
#         nb.save(nb.Nifti1Image(self.recombineWildbootstrapFromPool(z, partitioned), self.NIFTI.get_affine()),
#                 outname)
        print 'Result saved in:',outname

    def getMask4D(self, nrepeats=None):
        if nrepeats is None:
            return np.repeat(self.MASK.reshape(self.MASK.shape + (1,)), self.IMG_DATA.shape[-1], 3)
        else:
            return np.repeat(self.MASK.reshape(self.MASK.shape + (1,)), nrepeats
                             , 3)
    def prepareWildBootstrapInit(self, numberofsamples):
        nbs = numberofsamples
        tosses = generate_coin_tosses(self.getSignalLength(), nbs)
        dwinfo = wba.DiffusionWeightingInfo(self.BVALUES, self.BVECTORS)
        return dwinfo, tosses

    def partitionWildboostrapForPool(self, numberofsamples, weighted=True):
        "returns data to use with multiprocessing.Pool.map"
        dwinfo, tosses = self.prepareWildBootstrapInit(numberofsamples)               
        ndw = self.IMG_DATA.shape[-1]
        mask4D = self.getMask4D()
        data_to_be_processed = list()
        for k in range(self.IMG_DATA.shape[2]):
            frame_active_voxels = len(np.flatnonzero(self.MASK[:, :, k]))
            if frame_active_voxels<1:
                continue
            frame = compute_log_signal(self.IMG_DATA[:, :, k, :].flat[np.flatnonzero(mask4D[:, :, k, :])])
            data_to_be_processed.append(
                                        (wba.FrameData(k, frame.reshape((frame_active_voxels, ndw))),
                                         dwinfo,
                                         tosses, 
                                         weighted
                                         )
                                        )
        return data_to_be_processed

    def recombineWildbootstrapFromPool(self, pool_result, partitioned):
        #get n of wildbootstrap samples by the length of the coin tosses list...
        nbs = len(partitioned[0][-2])
#        nbs = pool_result[0].shape[-1]
        if nbs > 0:
            output = np.zeros(self.MASK.shape + (nbs, ))
            mask4D = self.getMask4D(nbs)
            for n, el in enumerate(partitioned):
                k = el[0].get_frame_no()
                try:
                    output[:,:,k,:].flat[np.flatnonzero(mask4D[:,:,k,:])] = pool_result[n].flatten()
                except IndexError:
                    #maybe not all slices were processed, warn user and return
                    print 'Not all slices were processed, last slice processed %d (in python indexing)'%(k-1)
                    break
        else:
            output = np.zeros(self.MASK.shape)
            for n, el in enumerate(partitioned):
                k = el[0].get_frame_no()
                try:
                    output[:,:,k].flat[np.flatnonzero(self.MASK[:,:,k])] = pool_result[n].flatten()
                except IndexError:
                    #maybe not all slices were processed, warn user and return
                    print 'Not all slices were processed, last slice processed %d (in python indexing)'%(k-1)
                    break            
        return output

def poolProcessData(X):
    frame, dwinfo, tosses, weighted = X
    nvox = frame.DATA.shape[0]
    nbs = len(tosses)
    #if nbs = 0 just compute the FA map
    if nbs > 0:
        output = np.zeros((nvox, nbs))
        for v in range(nvox):
            wbinit = wba.WildbootstrapInit(dwinfo, frame.DATA[v, :], weighted)
            output[v, :] = map(lambda x:compute_FA(eigvalues_vectors_from_tfit(x)), 
                                   wba.wildboostrapVoxelData(wbinit, dwinfo, tosses))
    else:
        output = np.zeros(nvox)
        for v in range(nvox):
            wbinit = wba.WildbootstrapInit(dwinfo, frame.DATA[v, :], weighted)
            #get weighted lsq or ordinary leassquare fit
            tfit = wbinit.get_tfit_wls() if weighted else wbinit.get_tfit_ols()
            output[v] = compute_FA(eigvalues_vectors_from_tfit(tfit))
                
    return output

def bootstrapFA(fname, nbootstraps=100, weighted=True):
    ddir = os.path.split(fname)[0]
    print 'Bootstrapping ', fname
    dti = FSLDTI(fname)
    dti.loadBValues(os.path.join(ddir, 'bvals'))
    dti.loadBVectors(os.path.join(ddir, 'bvecs'))
    dti.loadMask(os.path.join(ddir, 'mask.nii.gz'))
    NCPUS = multiprocessing.cpu_count()
    if weighted:
        #weighted least square
        outname = fname.replace('.nii','.bs.wls.FA.%d.mp.nii'%nbootstraps)
    else:
        #ordinary least square
        outname = fname.replace('.nii','.bs.ols.FA.%d.mp.nii'%nbootstraps)
        
    if nbootstraps == 0:
        outname = outname.replace('.0.mp','.mp').replace('.bs.','.')
        
    if NCPUS > 1:
        dti.computeWildbootstrapFA_multicore(outname, nbootstraps, weighted=weighted)
    else:
        dti.computeWildbootstrapFA_multicore(outname, nbootstraps, 1, weighted=weighted)
#        dti.computeWildbootstrapFA(fname.replace('.nii','.bs.w.FA.%d.nii'%nbootstraps), nbootstraps)

def testme():
    print "Welcome to pyDTI"
    ddir = r'D:\var\TobyXe_enh_v_max_eprime\3274\201303261434_PHILIPSMR3_901_DTI32TE63SENSE'
    fname = os.path.join(ddir,'201303261434_PHILIPSMR3_901_DTI32TE63SENSE_001.dcm.nii.gz')
    dti = FSLDTI(fname)
    dti.loadBValues(fname + '.bvals')
    dti.loadBVectors(fname + '.bvecs')
    return dti
    
    
    
    
    
    
