from setuptools import setup

v_temp = {}
with open("bw2calc/version.py") as fp:
    exec(fp.read(), v_temp)
version = ".".join((str(x) for x in v_temp["version"]))


setup(
    name='bw2calc',
    version=version,
    packages=["bw2calc"],
    author="Chris Mutel",
    author_email="cmutel@gmail.com",
    license="NewBSD 3-clause; LICENSE.txt",
    url="https://bitbucket.org/cmutel/brightway2-calc",
    install_requires=[
        "bw_processing",
        "numpy",
        "pandas",
        "scipy",
        "stats_arrays",
    ],
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
