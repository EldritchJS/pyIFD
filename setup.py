from setuptools import find_packages, setup

setup(
    name='pyIFD',
    extras_require=dict(tests=['pytest']),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
