from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from pathlib import Path
import shutil

class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent

        target_dir = build_dir if not self.inplace else root_dir

        self.copy_file(Path('automl') / '__init__.py', root_dir, target_dir)
        self.copy_file(Path('automl/data') / '__init__.py', root_dir, target_dir)
        self.copy_file(Path('automl/feature') / '__init__.py', root_dir, target_dir)
        self.copy_file(Path('automl/hyperparam') / '__init__.py', root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return

        shutil.copyfile(str(source_dir / path), str(destination_dir / path))

ext = [
    Extension('automl.*', ['automl/*.py']),
    Extension('automl.data.*', ['automl/data/*.py']),
    Extension('automl.feature.*', ['automl/feature/*.py']),
    Extension('automl.hyperparam.optimization',
              ['automl/hyperparam/optimization.py']),
    Extension('automl.hyperparam.templates',
              ['automl/hyperparam/templates.py']),
    Extension('automl.hyperparam.*', ['automl/hyperparam/*.py'])
]

setup(name='onepanel-automl',
      version='0.1.3',
      description='Automated Machine Learning Framework',
      author='Kirill Dubovikov, Igor Fokin',
      author_email='sales@onepanel.io',
      packages=[],
      cmdclass=dict(build_ext=MyBuildExt),
      ext_modules=cythonize(ext),
      include_package_data=True,
      install_requires=[
          'numpy>=1.13.3',
          'pandas>=0.20.3',
          'scikit-learn>=0.19.0',
          'tqdm',
          'hyperopt>=0.1',
          'networkx==1.11'
          ],
      license='License :: Other/Proprietary License',
      classifiers=['Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Operating System :: POSIX :: Linux',
                  'Programming Language :: Python :: 3.6'])
