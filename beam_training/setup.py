"""
    Setup for apache beam pipeline.
"""
import setuptools


NAME = 'create_tfrecords_training'
VERSION = '1.0'
REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.43.0',
    'tensorflow==2.8.0',
    'gcsfs'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
