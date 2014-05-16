'''
Created on 10 Jan 2013

@author: gfagiolo
'''

import sys
from optparse import OptionParser

from pyDTI import bootstrapFA

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-o", dest='weighted', action="store_true",help="use least square fit (by default uses weighted least squares)")
    parser.add_option("-s", dest='nsamples', type="int", default=100, help="-s #bootstrap_samples (by default is set to 100)")
    parser.add_option("-v", dest='verbose', action="store_true", default=False, help="verbose")
    (options, args) = parser.parse_args()
#    nboots=100 if options.nsamples is None else int(options.nsamples) 
    nboots=options.nsamples 
    weighted=not options.weighted
    if options.verbose:
        print'Options: {nsamples: %d, weighted:'%nboots, weighted,'}'
    if len(args) > 0:
        for fname in args:
            bootstrapFA(fname, nboots, weighted)
    else:
        print 'use:',sys.argv[0],' [options] dti_preprocessed_data'
        print '-s #bootstrap_samples (by default is set to 100)'
        print '-o use least square fit (by default uses weighted least squares)'
        