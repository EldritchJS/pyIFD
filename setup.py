from setuptools import find_packages, setup

setup(
    name='pyIFD',
    extras_require=dict(tests=['pytest']),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    setup_requires=[
        'cython','numpy'],
    install_requires=[
        'cython',
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-image',
        'pillow',
        'opencv-python',
        'jpegio @ git+https://github.com/eldritchjs/jpegio',
        'PyWavelets',]
)
