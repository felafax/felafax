from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="felafax",
    version="1.0",
    packages=find_packages(where='llama3_jax'),
    package_dir={'': 'llama3_jax'},
    install_requires=requirements,
)
