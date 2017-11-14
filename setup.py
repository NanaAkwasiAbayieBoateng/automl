from distutils.core import setup

setup(name='auto-ml',
      version='0.1',
      description='Automated Machine Learning Framework',
      author='Kirill Dubovikov, Igor Fokin',
      author_email='dubovikov.kirill@gmail.com',
      packages=['automl'],
      install_requires=[
          'numpy>=1.13.3',
          'pandas>=0.20.3',
          'scikit-learn>=0.19.1',
          'tqdm',
          'hyperopt>=0.1',
<<<<<<< HEAD
          'networkx==1.11',
          'xgboost'
||||||| merged common ancestors
          'networkx==1.11'
=======
          'networkx==1.11',
          'xgboost>=0.6'
>>>>>>> ab3e23e0c9c1569697f5c22102f0dc2a18048ff2
          ],
      license='License :: Other/Proprietary License',
      classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6'],)
