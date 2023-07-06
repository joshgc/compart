from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'numpy>=1.13',
    'matplotlib==2.2.3',
    'scipy==1.10.0',
    'sk-video==1.1.10',
    'Pillow==5.3.0',
]

setup(
    name='smooth_fft_tf',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='SmoothLife on TF')
