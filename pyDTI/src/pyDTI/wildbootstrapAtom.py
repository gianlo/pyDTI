'''
Created on 10 Jan 2013

@author: gfagiolo
'''
from helperFunctions import generate_design_matrix, compute_leverage,\
    designmatrix_pseudoinverse, compute_hat_matrix, np, mult_diag,\
    generate_weights, weighted_least_square_fit_fast, ols_least_square_fit


class DiffusionWeightingInfo(object):
        
    def __init__(self, BVALUES, BVECTORS):
        self.set_bvalues(BVALUES)
        self.set_bvectors(BVECTORS)
        self.__updateProperties()
        
    def __updateProperties(self):
        self.set_dm(generate_design_matrix(self.get_bvalues() , self.get_bvectors()))
        self.set_levs(compute_leverage(self.__computeHatMatrix()))

    def __computeHatMatrix(self):
        dm_pinv = designmatrix_pseudoinverse(self.get_dm()) 
        self.set_dm_pinv(dm_pinv)       
        return compute_hat_matrix(self.get_dm(), dm_pinv)
        
    def get_bvalues(self):
        return self.__BVALUES

    def get_bvectors(self):
        return self.__BVECTORS

    def get_dm(self):
        return self.__DM
    
    def get_dm_pinv(self):
        return self.__DM_PINV

    def get_levs(self):
        return self.__LEVS

    def set_bvalues(self, value):
        self.__BVALUES = value

    def set_bvectors(self, value):
        self.__BVECTORS = value

    def set_dm(self, value):
        self.__DM = value
        
    def set_dm_pinv(self, value):
        self.__DM_PINV = value

    def set_levs(self, value):
        self.__LEVS = value
        
    BVALUES = property(get_bvalues, set_bvalues, None, "BVALUES's docstring")
    BVECTORS = property(get_bvectors, set_bvectors, None, "BVECTORS's docstring")
    DM = property(get_dm, set_dm, None, "Design Matrix")
    DM_PINV = property(get_dm_pinv, set_dm_pinv, None, "Design Matrix Pseudo-Inverse")
    LEVS = property(get_levs, set_levs, None, "Leverages as heteroscedasticity consistent covariance matrix estimator (HCCME)")


class WildbootstrapInit(object):
    '''
    '''

    def __init__(self, DWI_OBJ, DATA_INIT, IS_WEIGHTED=True):
        self.set_dwi_obj(DWI_OBJ)
        self.set_data_init(DATA_INIT) 
        self.set_weighted(IS_WEIGHTED)
        self.__initialise()

    def __initialise(self):
        x = self.get_dwi_obj().get_dm()
        y = self.get_data_init()
        #perform Ordinary LS fit
        ols_tfit = self.__compute_ols_tfit()
        self.set_tfit_ols(ols_tfit)
        if self.get_weighted():
            wls_weights = self.__compute_wls_weights(ols_tfit)
            self.set_wls_fac_1(mult_diag(wls_weights, x.T, False))
            self.set_wls_fac_2(np.dot(self.get_wls_fac_1(), x))
            wls_tfit = self.__compute_wls_tfit()
            self.set_tfit_wls(wls_tfit)
            self.set_data_fitted(np.dot(x, wls_tfit))
            self.set_lev_errors(np.multiply(self.get_dwi_obj().get_levs(), y-self.get_data_fitted()))
        else:
            self.set_data_fitted(np.dot(x, ols_tfit))
            self.set_lev_errors(np.multiply(self.get_dwi_obj().get_levs(), y-self.get_data_fitted()))

    def __compute_ols_tfit(self):
        return ols_least_square_fit(self.get_data_init(), None, self.get_dwi_obj().get_dm_pinv())
#        try:
#            tfit = np.linalg.lstsq(self.get_dwi_obj().get_dm(), self.get_data_init())[0]
#        except np.linalg.LinAlgError as e:
#            print 'ERROR:WildbootstrapInit lstsq', e
#            raise e, 'fit_DTI_signal: Warning using lstsq'
#        return tfit

    def __compute_wls_weights(self, ols_tfit):
        return generate_weights(np.exp(np.dot(self.get_dwi_obj().get_dm(), ols_tfit)))
    
    def __compute_wls_tfit(self):
        return weighted_least_square_fit_fast(self.get_data_init(), self.get_wls_fac_1(), self.get_wls_fac_2())
    
    def get_wls_fac_1(self):
        return self.__WLS_FAC1

    def get_wls_fac_2(self):
        return self.__WLS_FAC2

    def set_wls_fac_1(self, value):
        self.__WLS_FAC1 = value

    def set_wls_fac_2(self, value):
        self.__WLS_FAC2 = value

    def get_lev_errors(self):
        return self.__LEV_ERRORS

    def get_dwi_obj(self):
        return self.__DWI_OBJ

    def set_lev_errors(self, value):
        self.__LEV_ERRORS = value

    def set_dwi_obj(self, value):
        self.__DWI_OBJ = value

    def get_data_init(self):
        return self.__DATA_INIT

    def set_data_init(self, value):
        self.__DATA_INIT = value

    def get_data_fitted(self):
        return self.__DATA_FITTED

    def set_data_fitted(self, value):
        self.__DATA_FITTED = value

    def get_weighted(self):
        return self.__WEIGHTED

    def set_weighted(self, value):
        self.__WEIGHTED = value

    def get_tfit_ols(self):
        return self.__TFIT_OLS

    def get_tfit_wls(self):
        return self.__TFIT_WLS

    def set_tfit_ols(self, value):
        self.__TFIT_OLS = value

    def set_tfit_wls(self, value):
        self.__TFIT_WLS = value

    DATA_INIT = property(get_data_init, set_data_init, None, "DATA_INIT is voxel's log signal")
    DWI_OBJ = property(get_dwi_obj, set_dwi_obj, None, "DWI_OBJ contains diffusion weighted acquisition info (DiffustionWeightingInfo obj)")
    DATA_FITTED = property(get_data_fitted, set_data_fitted, None, "DATA_FITTED weighted LS fit of data")
    LEV_ERRORS = property(get_lev_errors, set_lev_errors, None, "LEV_ERRORS are the leveraged errors using HCCME in DiffustionWeightingInfo obj")    
    WLS_FAC1 = property(get_wls_fac_1, set_wls_fac_1, None, "WLS_FAC1 first factor to perform fast WLS fit")
    WLS_FAC2 = property(get_wls_fac_2, set_wls_fac_2, None, "WLS_FAC2 second factor to perform fast WLS fit")
    WEIGHTED = property(get_weighted, set_weighted, None, "WEIGHTED boolean True if performing weighted least square fit")
    TFIT_OLS = property(get_tfit_ols, set_tfit_ols, None, "TFIT_OLS ordinary least square tensor fit result")
    TFIT_WLS = property(get_tfit_wls, set_tfit_wls, None, "TFIT_WLS weighted least sqaure tensor fit result")


class FrameData(object):

    def __init__(self, FRAME_NO, DATA):
        self.__FRAME_NO = FRAME_NO
        self.__DATA = DATA

    def get_frame_no(self):
        return self.__FRAME_NO

    def set_frame_no(self, value):
        self.__FRAME_NO = value

    def get_data(self):
        return self.__DATA

    def set_data(self, value):
        self.__DATA = value

    DATA = property(get_data, set_data, None, "DATA's docstring")
    FRAME_NO = property(get_frame_no, set_frame_no, None, "FRAME_NO's docstring") 
        

class WildbootstrapImage(object):

    def __init__(self, DATA, MASK):
        self.__DATA = DATA
        self.__MASK = MASK

    def get_data(self):
        return self.__DATA

    def get_mask(self):
        return self.__MASK

    def set_data(self, value):
        self.__DATA = value

    def set_mask(self, value):
        self.__MASK = value

    def del_data(self):
        del self.__DATA

    def del_mask(self):
        del self.__MASK
    
    DATA = property(get_data, set_data, del_data, "Bootstrapped image, volume + bootstrap index")
    MASK = property(get_mask, set_mask, del_mask, "Image mask (an image map with 1 for active voxels 0 otherwise)")
    
    def getMask4D(self, nrepeats=None):
        if nrepeats is None:
            return np.repeat(self.MASK.reshape(self.MASK.shape + (1,)), self.DATA.shape[-1], 3)
        else:
            return np.repeat(self.MASK.reshape(self.MASK.shape + (1,)), nrepeats, 3)

    def partitionForPool(self):
        "returns data to use with multiprocessing.Pool.map"
        ndw = self.DATA.shape[-1]
        mask4D = self.getMask4D()
        data_to_be_processed = list()
        for k in range(self.DATA.shape[2]):
            frame_active_voxels = len(np.flatnonzero(self.MASK[:, :, k]))
            if frame_active_voxels<1:
                continue
            frame = self.DATA[:, :, k, :].flat[np.flatnonzero(mask4D[:, :, k, :])]
            data_to_be_processed.append(
                    FrameData(k, 
                              frame.reshape((frame_active_voxels, ndw))))
        return data_to_be_processed

    def recombineFromPool(self, pool_result, partitioned):
        """recombine (reduce) results of multiprocessing.Pool.map into an image"""
        nbs = pool_result[0].shape[-1]
        output = np.zeros(self.MASK.shape + (nbs, ))
        mask4D = self.getMask4D(nbs)
        for n, el in enumerate(partitioned):
            k = el.get_frame_no()
            try:
                output[:,:,k,:].flat[np.flatnonzero(mask4D[:,:,k,:])] = pool_result[n].flatten()
            except IndexError:
                #maybe not all slices were processed, warn user and return
                print 'Not all slices were processed, last slice processed %d (in python indexing)'%(k-1)
                break
        return output


def wildboostrapVoxelData(wbinit, dwinfo, cointosses):
    output = np.zeros((len(cointosses), 7))
    data_fitted = wbinit.get_data_fitted()
    lev_erros = wbinit.get_lev_errors()
    if wbinit.get_weighted():
        f1 = wbinit.get_wls_fac_1()
        f2 = wbinit.get_wls_fac_2()
        for ct in range(len(cointosses)):
            y = data_fitted + np.multiply(cointosses[ct], lev_erros)
            output[ct,:] = weighted_least_square_fit_fast(y, f1, f2)
    else:
        x_pinv = dwinfo.get_dm_pinv()
        for ct in range(len(cointosses)):
            y = data_fitted + np.multiply(cointosses[ct], lev_erros)
            output[ct,:] = ols_least_square_fit(y, None, x_pinv)
        
    return output

            