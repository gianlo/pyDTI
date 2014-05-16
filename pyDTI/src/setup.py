'''
Install script

Created on 16 May 2014

@author: glf12
'''

from distutils.core import setup


setup(name='pyDTI',
      version='0.9beta',
      description='A python interface for processing DTI data with FSL format',
      author='Gianlorenzo Fagiolo',
      url='https://github.com/gianlo/pyDTI',
      license='License :: OSI Approved :: MIT License',
      requires=['numpy', 'pynii', 'scipy'],
      packages=['pyDTI'])
