import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="rootable",
    version="0.1.0",
    author="Johannes Bilk",
    author_email="johannes.bilk@physik.uni-giessen.de",
    description="A simple packages for extracting PXD data from root files",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ub.uni-giessen.de/gc2052/fromroot",
    packages=setuptools.find_packages(),
    license='MIT',
    python_requires='>=3.10',
    install_requires=[
        "numpy>=1.21.0",
        "uproot>=4.0.11"
    ],
    extras_require={
        'pandas': ['pandas>=1.0.0']
    },
    keywords=['python', 'pxd', 'root'],
    classifiers= [
        "Development Status :: 0.1.0",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
