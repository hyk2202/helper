from setuptools import setup, find_packages

setup(
    name="helper_hyk",
    version="0.0.1",
    description="hyk_test_pip",
    author="Kim Han-Youl",
    author_email="hyk2202@gmail.com",
    license="MIT",
    packages=find_packages(exclude=[]),
    keywords=["data",  "helper", ],
    python_requires=">=3.11",
    zip_safe=False,
    url="https://github.com/hyk2202/helper",
    install_requires=[
        "tabulate",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scipy",
        "pingouin",
        "scikit-learn",
        "imblearn",
        "pdoc3",
        "pmdarima",
        "prophet",
        'bs4',
        'nltk',
        'pmdarima'
    ],
)
