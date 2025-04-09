from setuptools import setup, find_packages

setup(
    name="crypto_backtester",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "gymnasium",
        "pandas-ta",
        "python-binance",
        "requests",
    ],
    description="A flexible cryptocurrency backtesting framework",
    author="CryptoBacktester Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'crypto-backtest=crypto_backtester.cli:main',
        ],
    },
) 