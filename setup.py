from setuptools import find_packages, setup

setup(
    name='pyIFD',
    version='0.0.2',
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
        'PyWavelets',
        'jpegio']
)
