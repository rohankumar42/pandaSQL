import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'pandas==1.0.5',
    'psutil>=5.7.2'
]

setuptools.setup(
    name='pandaSQL',
    version='0.0.1',
    author="Rohan Kumar",
    author_email="rk@rohankumar.io",
    description="Pandas-style Data Analysis with SQLite3 Backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohankumar42/pandaSQL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
    install_requires=requires,
    python_requires='>=3.6',
)
