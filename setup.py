import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GLM_JIVE",
    version="0.0.1",
    author="Soufiane Mourragui <soufiane.mourragui@gmail.com>, ",
    author_email="soufiane.mourragui@gmail.com",
    description="PERCOLATE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/NKI-CCB/TRANSACT",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'torch'],
    python_requires='>=3.8',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 1 - Planning",
    ),
)