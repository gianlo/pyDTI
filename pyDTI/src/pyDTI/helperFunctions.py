'''
Created on 10 Jan 2013

@author: gfagiolo
'''

import numpy as np
import os

#uses also scipy

#===============================================================================
# CONSTANTS
#===============================================================================

FA_NORM = np.sqrt(1.5)    

#===============================================================================
# IMAGE PROCESSING
#===============================================================================

def autoMask(img, perc_radius=None, debug_bool=False):
    from scipy import ndimage
    da = np.uint16(img)
    COG = tuple(map(int, ndimage.center_of_mass(img)))
    markers = np.zeros(da.shape, dtype=np.int8)
    #set outside brain to 1
    markers.flat[0] = 1
    #set inside brain to 2
    markers.flat[np.ravel_multi_index(COG, markers.shape)] = 2
    mask = ndimage.watershed_ift(da, markers)-1
    if debug_bool:
        print len(np.flatnonzero(mask))
    if isinstance(perc_radius, float):
        #find distance to background
        dm = ndimage.distance_transform_bf(mask)
        #find radius of phantom (i.e. centre is most distant from background)
        radius = dm.flatten().max()
        #select only pixel outside range desired
        sel = np.flatnonzero(dm.flatten() <= (1.-perc_radius)*radius)
        #set those pixels to zero in the final mask
        mask.flat[sel] = 0
        if debug_bool:
            print len(np.flatnonzero(mask)), radius, (1. -perc_radius)*radius
    return np.uint8(mask)

#===============================================================================
# ###DTI functions###
#===============================================================================

def get_signal_intensity_from_fit(f):
    return np.exp(f[0])

def coin_toss(ns):
    return 2*(np.random.randint(2, size=ns)-0.5)

def generate_coin_tosses(sigdim, ntosses):
    tosses = list()
    stosses = set()
    for _ in range(ntosses):
        ctoss = coin_toss(sigdim)
        #check that it is a novel permutation
        while ''.join(map(str,ctoss)) in stosses:
            ctoss = coin_toss(sigdim)                    
        tosses.append(ctoss)
        stosses.add(''.join(map(str,ctoss)))
    return tosses

#fsl code from vector to tensor!
#inline SymmetricMatrix vec2tens(ColumnVector& Vec){
#  SymmetricMatrix tens(3);
#  tens(1,1)=Vec(1);
#  tens(2,1)=Vec(2);
#  tens(3,1)=Vec(3);
#  tens(2,2)=Vec(4);
#  tens(3,2)=Vec(5);
#  tens(3,3)=Vec(6);
#  return tens;
#}

def get_tensor_from_fit(f):
    return np.array([[f[1], f[2], f[3]],
                     [f[2], f[4], f[5]],
                     [f[3], f[5], f[6]]])

def generate_design_matrix(bs, v):
    """generate design matrix from BVALUES bs and BVECTORS v"""
    #design matrix has 1 in all first rows (to detect image b=0 SIGNAL)
    x = np.ones((bs.shape[0], 7))
    for i, b in enumerate(bs):
        if b == 0:
            x[i, 1:] = 0
        else:
            x[i, 1] = -b*v[0, i]**2
            x[i, 2] = -b*2*v[0, i]*v[1, i]
            x[i, 3] = -b*2*v[0, i]*v[2, i]
            x[i, 4] = -b*v[1, i]**2
            x[i, 5] = -b*2*v[1, i]*v[2, i]
            x[i, 6] = -b*v[2, i]**2
    return x

def compute_FA(e):    
    return FA_NORM*np.linalg.norm(e-np.mean(e))/np.linalg.norm(e)

def compute_MD(e):      
    return np.mean(e)

def compute_DTI_scalars(m):
    #assumes m is hermitian
    try:
        e  = np.linalg.eigh(m)[0] 
    except np.linalg.LinAlgError as lae:
        print 'ERROR'
        print lae, '\ntensor', m
        raise lae
    return np.array([compute_MD(e), compute_FA(e)])

def eigvalues_vectors_from_tfit(tfit):
    try:
        e  = np.linalg.eigh(get_tensor_from_fit(tfit))[0] 
    except np.linalg.LinAlgError as lae:
        print 'ERROR'
        print lae, '\ntensor', tfit
        raise lae
    return e

def designmatrix_pseudoinverse(x, weights_inverse=None):    
    #linlag.inv uses linalg.solve which in turn wraps lapack_lite.dgesv
    if weights_inverse is None:
        return np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
    else:
        return np.dot(np.dot(np.linalg.inv(np.dot(x.T, np.dot(weights_inverse, x))), x.T), weights_inverse)

def compute_hat_matrix(DESIGNMATRIX, DESIGNMATRIX_PSEUDOINVERSE):
    return np.diagonal(np.dot(DESIGNMATRIX, DESIGNMATRIX_PSEUDOINVERSE))

def compute_leverage(H_HAT):
    INV_LEVS = np.ones(H_HAT.shape)
    try:
        #the resampling is only done on the gradient directions, not on the logS component!
        INV_LEVS[1:] = 1./np.sqrt(1.-H_HAT[1:])                        
    except RuntimeWarning as rw2:
        print 'ERROR FIT: leveraged errors'
        print rw2, '\nHat matrix diagonal', H_HAT
        raise rw2
    return INV_LEVS

def compute_log_signal(SIGNAL):
    try:
        LOGSIGNAL = np.log(SIGNAL)
    except RuntimeWarning as rw:
        print 'setSignalERROR: non-positive value in signal'
        print rw, '\nsignal', SIGNAL[SIGNAL<=0]
        raise rw
    return LOGSIGNAL

def generate_weights(Sg, inverse=True, linear_form=True):
    #noted that in FSL code the weight is Sg(i)>0?Sg(i)**2:1
    if inverse:
        try:
            #noted that in FSL code the weight is Sg(i)>0?Sg(i)**2:1
            tmp = Sg
            tmp.flat[np.flatnonzero(tmp<1e-15)] = 1
            out = (1./tmp)**2
        except RuntimeWarning as rw:
#            print 'ERROR '
#            print rw, Sg
            raise rw
    else:
        out = Sg**2
    
    if linear_form:
        return out
    else:
        return np.diag(out)
    
def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

def weighted_least_square_fit(y, x, W):
    tmpfac = mult_diag(W, x.T, False)
    try:
        tfit = np.linalg.solve(np.dot(tmpfac, x), np.dot(tmpfac, y))
    except np.linalg.LinAlgError as e:
        tfit = np.linalg.lstsq(np.dot(tmpfac, x), np.dot(tmpfac, y))[0]
        print 'WARNING::weighted_least_square_fit: using lstsq', e
    return tfit

def weighted_least_square_fit_fast(y, f1, f2):
    try:
        tfit = np.linalg.solve(f2, np.dot(f1, y))
    except np.linalg.LinAlgError as e:
        tfit = np.linalg.lstsq(f2, np.dot(f1, y))[0]
        print 'WARNING::weighted_least_square_fit: using lstsq', e
    return tfit

def ols_least_square_fit(y, x, x_pinv=None):
    if x_pinv is not None:
        return np.dot(x_pinv, y)
    else:
        try:
            tfit = np.linalg.lstsq(x, y)[0]
        except np.linalg.LinAlgError as e:
            print 'ERROR:WildbootstrapInit lstsq', e
            raise e, 'fit_DTI_signal: Warning using lstsq'
        return tfit


def fit_DTI_signal(sig, x, weighted=False, x_pinv=None, is_log_sig=False, h_hat=None, inv_levs=None):
    """do a least square fit where sig is the SIGNAL and DESIGNMATRIX is the design matrix
    Returns: tfit, mu, errors, H_diag, lev_errors"""
    if is_log_sig:
        y = sig
    else:
        #log SIGNAL
        #this assumes that sig is strictly positive!!!!
        try:
            y = np.log(sig)
        except RuntimeWarning as rw:
            print 'ERROR FIT: non-positive value in signal'
            print rw, '\nsignal', sig
            raise rw
    #beta0=np.linalg.lstsq(DESIGNMATRIX,y)[0]
    #pseudo inverse z = (DESIGNMATRIX*xT)^(-1)*xT
    if x_pinv is None:
        z = designmatrix_pseudoinverse(x)
    else:
        z = x_pinv
    #compute hat matrix H diagonal, H = DESIGNMATRIX*z
    #in whitcher et al 2008 the H_diagonal is computed from the OLS regression
    if h_hat is None:
        H_diag = np.diagonal(np.dot(x, z))
    else:
        H_diag = h_hat
    #beta0 is the least squares fit result, beta0 = z*y
    beta0 = np.dot(z, y)
    if weighted:
        #generate synthetic SIGNAL
        try:
            Sg = np.exp(np.dot(x, beta0))
        except RuntimeWarning as rw:
            print 'ERROR FIT: generating synthetic signal'
            print rw, '\ndesign matrix',x, '\ntensor OLS fit' , beta0
            raise rw
            
        #compute weight matrix W
        try:
            W = generate_weights(Sg)
        except RuntimeWarning as rw:
            print 'ERROR FIT: generating weights'
            print rw, '\nsynthetic signal', Sg
            raise rw
        #do IS_WEIGHTED fit        
        #z = designmatrix_pseudoinverse(x, W) #np.dot(np.dot(np.linalg.inv(np.dot(DESIGNMATRIX.T,np.dot(W,DESIGNMATRIX))),DESIGNMATRIX.T),W)
        #beta is the final IS_WEIGHTED matrix
        #beta = np.dot(z, y)
        #linalg.lstsq wraps the function lapack_lite.dgelsd see some information on lapack dgelsd... That explains why the linalg calls the routine twice
#        LWORK (input) INTEGER
#        The dimension of the array WORK. LWORK must be at least 1. 
#        The exact minimum amount of workspace needed depends on M, N and NRHS. 
#        As long as LWORK is at least 12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2, 
#        if M is greater than or equal to N or 12*M + 2*M*SMLSIZ + 8*M*NLVL + M*NRHS + (SMLSIZ+1)**2, 
#        if M is less than N, the code will execute correctly. 
#        SMLSIZ is returned by ILAENV and is equal to the maximum size of the subproblems at 
#        the bottom of the computation tree (usually about 25), and NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 ) 
#        For good performance, LWORK should generally be larger. 
#        If LWORK = -1, then a workspace query is assumed; 
#        the routine only calculates the optimal size of the WORK array, returns this value as the first entry of 
#        the WORK array, and no error message related to LWORK is issued by XERBLA. 
#        IWORK (workspace) INTEGER array, dimension (MAX(1,LIWORK)) LIWORK >= 3 * MINMN * NLVL + 11 * MINMN, where MINMN = MIN( M,N ).      
        #tfit = np.linalg.lstsq(np.dot(x.T, np.dot(W,x)), np.dot(x.T, np.dot(W, y)))[0]
        #use faster diagonal matrix mul routine
        tmpfac = mult_diag(W, x.T, False)
#        tfit = np.linalg.lstsq(np.dot(tmpfac, x), np.dot(tmpfac, y))[0]
        try:
            tfit = np.linalg.solve(np.dot(tmpfac, x), np.dot(tmpfac, y))
        except np.linalg.LinAlgError as e:
            tfit = np.linalg.lstsq(np.dot(tmpfac, x), np.dot(tmpfac, y))[0]
            print 'fit_DTI_signal: Warning using lstsq', e
#            print e, '\ntensor fit', f_s[0]
    else:
        tfit = beta0
    #compute hat matrix H diagonal, H = DESIGNMATRIX*z
    #in whitcher et al 2008 the H_diagonal is computed from the OLS regression
    #H_diag = np.diagonal(np.dot(x, z))
    #compute MU
    mu = np.dot(x, tfit)
    #compute log residuals
    errors = y-mu
    if inv_levs is None:
        try:
            
            #the resampling is only done on the gradient directions, not on the logS component!
            inv_levs = np.ones(errors.shape)
            inv_levs[1:] = 1./np.sqrt(1.-H_diag[1:])        
            
        except RuntimeWarning as rw2:
            print 'ERROR FIT: leveraged errors'
            print rw2, '\nHat matrix diagonal', H_diag
            inv_levs = np.ones(errors.shape)
            raise rw2
    lev_errors = errors*inv_levs    
    return (tfit, mu, errors, H_diag, lev_errors)
        
def wild_bootstrap_DTI_scalars(x, mu, lev_err, nbs=1000, weighted=False, x_pinv=None, fixed_coins=None, h_hat=None, inv_levs=None):
    fas = np.zeros((2, nbs))
    for i in range(nbs):
        #generate a random error set
        if fixed_coins is None:
            e_s = np.multiply(coin_toss(len(lev_err)), lev_err)
        else:
            e_s = np.multiply(fixed_coins[i], lev_err)
        #generate new SIGNAL
        #use log of signal!
        e_s[0] = 0
        s_s = mu+e_s
#        LEGACY
#        try:
#            #the resampling is only done on the gradient directions, not on the logS component!
#            e_s[0] = 0
#            s_s = np.exp(mu+e_s)
#        except RuntimeWarning as rw:
#            print 'ERROR WILDBOOTSTRAP: generating synthetic signal'
#            print rw, '\nmu', mu, '\nleveraged errors with coin tosses', e_s
#            raise rw

        #compute least square DIFFUSION_TENSOR
        f_s = fit_DTI_signal(s_s, x, weighted=weighted, x_pinv=x_pinv, is_log_sig=True, h_hat=h_hat, inv_levs=inv_levs)
        #compute mean diffusivity and fractional anisotropy
        try: 
            fas[:, i] = compute_DTI_scalars(get_tensor_from_fit(f_s[0]))
        except np.linalg.LinAlgError as e:
            print 'ERROR WILDBOOTSTRAP: computing FA and MD'
            print e, '\ntensor fit', f_s[0]
            raise e
    return fas

def normalize_vectors(vecs):
    tmp = vecs.copy()
    for n in range(vecs.shape[1]):
        cnorm = np.linalg.norm(vecs[:, n])
        if np.abs(cnorm)>0:
            tmp[:, n] /= cnorm    
    return tmp



def loadBValues(bvals_fname):
    """Loads b-values from BVALUES_FILENAME"""
    if os.path.exists(bvals_fname):
        return np.loadtxt(bvals_fname)

def loadBVectors(bvecs_fname):
    """Loads b-vecs from BVALUES_FILENAME"""
    if os.path.exists(bvecs_fname):            
        return normalize_vectors(np.loadtxt(bvecs_fname))


##  int decompose_aff(ColumnVector& params, const Matrix& affmat, 
##            const ColumnVector& centre,
##            int (*rotmat2params)(ColumnVector& , const Matrix& ))
def decompose_aff(affmat, centre):
    """      // decomposes using the convention: mat = rotmat * skew * scale
      // order of parameters is 3 rotation + 3 translation + 3 scales + 3 skews
      // angles are in radians"""
##Tracer tr("decompose_aff");
##if (params. Nrows() < 12)
##params.ReSize(12);
##if (rotmat2params==0)  
##{ 
##cerr << "No rotmat2params function specified" << endl;  
##return -1; 
##}
#ColumnVector x(3), y(3), z(3);
#Matrix aff3(3,3);
#aff3 = affmat.SubMatrix(1,3,1,3);
    aff3 = affmat[:3,:3]
#x = affmat.SubMatrix(1,3,1,1);
#y = affmat.SubMatrix(1,3,2,2);
#z = affmat.SubMatrix(1,3,3,3);
    x = affmat[:3,0]
    y = affmat[:3,1]
    z = affmat[:3,2]
#float sx, sy, sz, a, b, c;
#sx = norm2(x);
    sx = np.linalg.norm(x)
#sy = std::sqrt( dot(y,y) - (Sqr(dot(x,y)) / Sqr(sx)) );
    sy = np.sqrt( np.dot(y,y) - (np.square(np.dot(x,y)) / np.square(sx)) )
#a = dot(x,y)/(sx*sy);
    a = np.dot(x,y)/(sx*sy)
#ColumnVector x0(3), y0(3);
#x0 = x/sx;
#y0 = y/sy - a*x0;
    x0 = x/sx
    y0 = y/sy - a*x0

#sz = std::sqrt(dot(z,z) - Sqr(dot(x0,z)) - Sqr(dot(y0,z)));
#b = dot(x0,z)/sz;
#c = dot(y0,z)/sz;
    sz = np.sqrt(np.dot(z,z) - np.square(np.dot(x0,z)) - np.square(np.dot(y0,z)))
    b = np.dot(x0,z)/sz
    c = np.dot(y0,z)/sz

#params(7) = sx;  params(8) = sy;  params(9) = sz;
#Matrix scales(3,3);
#float diagvals[] = {sx,sy,sz};
#diag(scales,diagvals);
    scales = np.diag([sx,sy,sz])
#Real skewvals[] = {1,a,b,0 , 0,1,c,0 , 0,0,1,0 , 0,0,0,1}; 
#Matrix skew(4,4);
#skew  << skewvals;
    skew = np.array([1,a,b,0 , 0,1,c,0 , 0,0,1,0 , 0,0,0,1]).reshape((4,4))
#params(10) = a;  params(11) = b;  params(12) = c;
#Matrix rotmat(3,3);
#rotmat = aff3 * scales.i() * (skew.SubMatrix(1,3,1,3)).i();
    rotmat = np.dot(np.dot(aff3 , np.linalg.inv(scales)),  np.linalg.inv(skew[:3,:3]))
#ColumnVector transl(3);
#transl = affmat.SubMatrix(1,3,1,3)*centre + affmat.SubMatrix(1,3,4,4)
#     - centre;
    transl = np.dot(aff3,centre) + affmat[:3,3]- centre
#for (int i=1; i<=3; i++)  { params(i+3) = transl(i); }
#ColumnVector rotparams(3);
#(*rotmat2params)(rotparams,rotmat);
#for (int i=1; i<=3; i++)  { params(i) = rotparams(i); }
#return 0;
#}
    return (rotmat,transl,[a,b,c],[sx,sy,sz])

##  int decompose_aff(ColumnVector& params, const Matrix& affmat, 
##            const ColumnVector& centre,
##            int (*rotmat2params)(ColumnVector& , const Matrix& ))
##    {
##      // decomposes using the convention: mat = rotmat * skew * scale
##      // order of parameters is 3 rotation + 3 translation + 3 scales + 3 skews
##      // angles are in radians
##      Tracer tr("decompose_aff");
##      if (params. Nrows() < 12)
##    params.ReSize(12);
##      if (rotmat2params==0)  
##    { 
##      cerr << "No rotmat2params function specified" << endl;  
##      return -1; 
##    }
##      ColumnVector x(3), y(3), z(3);
##      Matrix aff3(3,3);
##      aff3 = affmat.SubMatrix(1,3,1,3);
##      x = affmat.SubMatrix(1,3,1,1);
##      y = affmat.SubMatrix(1,3,2,2);
##      z = affmat.SubMatrix(1,3,3,3);
##      float sx, sy, sz, a, b, c;
##      sx = norm2(x);
##      sy = std::sqrt( dot(y,y) - (Sqr(dot(x,y)) / Sqr(sx)) );
##      a = dot(x,y)/(sx*sy);
##      ColumnVector x0(3), y0(3);
##      x0 = x/sx;
##      y0 = y/sy - a*x0;
##      sz = std::sqrt(dot(z,z) - Sqr(dot(x0,z)) - Sqr(dot(y0,z)));
##      b = dot(x0,z)/sz;
##      c = dot(y0,z)/sz;
##      params(7) = sx;  params(8) = sy;  params(9) = sz;
##      Matrix scales(3,3);
##      float diagvals[] = {sx,sy,sz};
##      diag(scales,diagvals);
##      Real skewvals[] = {1,a,b,0 , 0,1,c,0 , 0,0,1,0 , 0,0,0,1}; 
##      Matrix skew(4,4);
##      skew  << skewvals;
##      params(10) = a;  params(11) = b;  params(12) = c;
##      Matrix rotmat(3,3);
##      rotmat = aff3 * scales.i() * (skew.SubMatrix(1,3,1,3)).i();
##      ColumnVector transl(3);
##      transl = affmat.SubMatrix(1,3,1,3)*centre + affmat.SubMatrix(1,3,4,4)
##             - centre;
##      for (int i=1; i<=3; i++)  { params(i+3) = transl(i); }
##      ColumnVector rotparams(3);
##      (*rotmat2params)(rotparams,rotmat);
##      for (int i=1; i<=3; i++)  { params(i) = rotparams(i); }
##      return 0;
##    }

def probable_intensity_ranges(data_map, mask, delta):
    c, l = np.histogram(data_map.flatten()[np.flatnonzero(mask)], bins=100)
    dl = 0.5*(l[1]-l[0])
    p = np.float64(c)/np.sum(c)
    #desc order
    i = np.argsort(p)[::-1]
    cu = np.cumsum(p[i])
    #find histograms limits for most probable lot
    ci = i[np.flatnonzero(cu<delta)]
    intensities = []
    for e in ci:
        intensities.append(l[e]-dl,l[e]+dl)
    return intensities

if __name__ == '__main__':
    print 'Welcome to piDTI helperFunctions'