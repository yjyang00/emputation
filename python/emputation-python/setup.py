import re
from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f if l.strip()]

with open('emputation/__init__.py') as f:
    version = re.search(r'^__version__ = "(.+)"$', f.read(), re.M).group(1)


setup(
    name='emputation',
    version=version,
    description='Emputation: identification-guided neural imputation',
    author='Yanjiao Yang, Yikun Zhang, Xinwei Shen, Yen-Chi Chen',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
