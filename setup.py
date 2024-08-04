from setuptools import setup, find_packages

setup(
    name='smart-ml',
    version='0.1.0',
    author='Ishi',
    author_email='ishishiv3012@gmail.com',
    description='A Python library for machine learning replicating scikit-learn functionality.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ishi3012/smart-ml',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.23.0',
        'pandas>=1.5.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
    ],
    extras_require={
        'dev': ['pytest>=7.2.0', 'sphinx>=6.0.0'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)