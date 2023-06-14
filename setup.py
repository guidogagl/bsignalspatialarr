from setuptools import setup, find_packages

setup(
    name='bsignalspatialarr',
    version='1.0',
    packages=find_packages(),
    author='Guido Gagliardi',
    description='A Python package for preprocessing brain signal features into images to improve the classification performances of CNNs.',
    install_requires=[
        'numpy>=1.19.2',
    ],
)