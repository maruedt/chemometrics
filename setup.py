import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chemometrics",
    version="0.1",
    author="Matthias Rüdt",
    author_email="mruedt@icloud.com",
    description="package for chemometric data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maruedt/chemometrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    python_requires='>=3.6',
)
