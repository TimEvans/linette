from setuptools import setup, find_packages

# Get version from tncontract/version.py
exec(open("linette/version.py").read())

setup(
    name='linette',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='GNU General Public License v3.0',
    author='Tim Evans',
    author_email='',
    description='Linette - Basic library for handling linear networks of Gaussian Random Tensors'
)
