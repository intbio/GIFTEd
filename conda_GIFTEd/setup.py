from setuptools import setup, find_packages

setup(
    name = 'gifted',
    description = 'Gromacs Interactive Forcefield & Topology Editor',
    url = 'https://github.com/intbio/GIFTEd',
    author = 'E. Andreeva, G.Armeev',
    author_email = 'andreeva.elizaveta.2015@post.bio.msu.ru',
    license = 'Public Domain',
    packages = find_packages(exclude=['examples', 'docs', 'tests*']),
    install_requires = [
        'rdkit == 2018.03.2.0',
        'numpy == 1.14.3',
        'matplotlib == 2.2.2',
    ],
    python_requires='==3.6.*',
    zip_safe = False)
