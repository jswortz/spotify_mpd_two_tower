"""
    Setup for apache beam pipeline.
"""
import setuptools


NAME = 'create_tfrecords'
VERSION = '1.0'
REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.41.0',
    'tensorflow==2.8.0'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
