from distutils.core import setup

setup(
    name="sklearn_special_ensembles",
    version="1.1.0",
    license="MIT",
    description="A library of specialized ensembles for sklearn-type base models.",
    author="Ethan Wilk",
    author_email="ejwilk@caltech.edu",
    url="https://github.com/ewilk0/sklearn_special_ensembles/tree/v1.0.1",
    download_url="https://github.com/ewilk0/sklearn_special_ensembles/archive/refs/tags/v1.0.1.tar.gz",
    packages=[
        "sklearn_special_ensembles", 
        "sklearn_special_ensembles.models", 
        "sklearn_special_ensembles.tests"
    ],
    keywords=["sklearn", "ensemble", "modeling", "data analysis", "machine learning"],
    install_requires=[
        "pandas",
        "numpy"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3'
    ]
)