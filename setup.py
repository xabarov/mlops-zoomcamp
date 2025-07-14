"""
Setup script for the project.
"""

from setuptools import find_packages, setup

setup(
    name='mlops-zoomcamp',
    version='1.00',
    description='My homework for MLOPs course',
    author='Xabarov Roman',
    author_email='xabarov1985@gmail.com',
    url='https://github.com/xabarov/mlops-zoomcamp',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: >= 3.9',
        'Topic :: Text Processing :: Linguistic',
    ],
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "xgboost",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "pyyaml",
        "fastapi",
    ],
    include_package_data=True,
    zip_safe=False,
)
