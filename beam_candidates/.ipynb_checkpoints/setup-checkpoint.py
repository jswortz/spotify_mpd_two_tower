"""
    Setup for apache beam pipeline.
"""
import setuptools


NAME = 'create_tfrecords_candidates'

VERSION = '1.1'
REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.41.0',
    'tensorflow==2.8.0',
    'gcsfs==2022.8.2'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
