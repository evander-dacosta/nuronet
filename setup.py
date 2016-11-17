import os

from setuptools import setup, find_packages

version = '0.0.1'
here = os.path.abspath((os.path.dirname(__file__)))
try:
    README = open(os.path.join(here, 'README.rst'))
    CHANGES = open(os.path.join(here, 'CHANGES.rst'))
except:
    README = CHANGES = ''
    
install_requires = [
    'matplotlib',
    'numpy',
    'scipy',
    'theano',
    'tabulate',
    'scikit-learn',
    'pandas'
    ]

tests_require = [
                 ]

docs_require = [
                ]


setup(name = 'nuronet2',
      version = version,
      description = "",
      long_description='\n\n'.join([README, CHANGES]),
      classifiers = [
                     "Programming Language :: Python::2.7"
                     ],
      keywords = '',
      author = 'Evander DaCosta',
      author_email = 'ebdc1g09@soton.ac.uk',
      url = 'https://github.com/k9triz/segnet',
      license = 'COMMERCIAL',
      packages = find_packages(),
      include_package_data = True,
      zip_safe = False,
      install_requires = install_requires,
      extras_require = {})