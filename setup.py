from setuptools import setup, find_packages
from automl.constants import *

setup(name='onepanel-automl',
      version=AUTOML_VERSION,
      description='Automated Machine Learning Framework',
      author='Onepanel, Inc.',
      author_email='info@onepanel.io',
      packages=find_packages(),
      dependency_links=[
          'https://github.com/hyperopt/hyperopt'
         ],
      install_requires=[
          'numpy==1.13.3',
          'pandas>=0.22.0',
          'scikit-learn>=0.19.1',
          'tqdm',
          'networkx>=2.0'
          ],
      license='License :: Other/Proprietary License',
      classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6'])
