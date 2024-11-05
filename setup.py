from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
        name='moojoco',
        version='1.1.3',
        description='A unified framework for implementing and interfacing with MuJoCo and MuJoCo-XLA simulation '
                    'environments.',
        long_description=readme,
        url='https://github.com/Co-Evolve/moojoco',
        license=license,
        packages=find_packages(exclude=('tests', 'docs')),
        install_requires=required
        )
