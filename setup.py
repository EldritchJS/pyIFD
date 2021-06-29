import codecs
import os

from setuptools import find_packages, setup

install_requires = [
    "setuptools",
]

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="pyIFD",
    version=0.1, #get_version("art/__init__.py"),
    description="Toolbox for image forgery detection.",
    #long_description=long_description,
    long_description_content_type="text/markdown",
    author="EldritchJS",
    author_email="jschlessman@gmail.com",
    maintainer="EldritchJS",
    maintainer_email="jschlessman@gmail.com",
    url="https://github.com/EldritchJS/pyIFD",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
