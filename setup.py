from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='psf_ot_barycenter',
      version='0.1',
      description='OT barycenter analysis for PSF Application',
      long_description = readme(),
      url='http://github.com/benjaminleroy/psf_ot_barycenter',
      author='benjaminleroy',
      author_email='bpleroy@stat.cmu.edu',
      license='MIT',
      packages=['psf_ot_barycenter'],
      install_requires=[
          'astropy.io',
          'collections',
          'copy',
          'matplotlib',
          'numpy',
          'PIL',
          're',
          'scipy'
      ],
      test_suite='nose.collector',
      test_require=['nose'],
      zip_safe=False)
