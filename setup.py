from os import path
from codecs import open
from setuptools import setup, find_packages

# we're using a README.md, if you do not have this in your folder, simply
# replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Here you specify the meta-data of your package. The `name` argument is
# needed in some other steps.
setup(
    name='federated_cvdm_training_poc',
    version="1.0.0",
    description='test',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # TODO add a url to your github repository here (or remove this line if
    # you do not want to make your source code public)
    # url='https://github.com/....',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'vantage6-algorithm-tools==4.8.1',
        'pandas==2.2.3',
        'xlrd==2.0.1',
        'scikit-learn==1.6.1',
        'scipy==1.13.1',
        'torch==2.5.1',
        'configparser==7.1.0',
        'lifelines==0.30.0',
        'h5py==3.12.1'
    ]
)
