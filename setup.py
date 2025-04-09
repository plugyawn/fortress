from setuptools import setup, find_packages

setup(
    name="fortress",
    version="0.1.0",
    description="Advanced cryptocurrency backtesting and trading framework",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "gymnasium",
        "pandas-ta",
        "python-binance",
        "torch",
        "transformers",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        'console_scripts': [
            'fortress-cli=fortress.cli:main',
        ],
    },
) 