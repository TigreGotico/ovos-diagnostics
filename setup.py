import os
import os.path

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))

def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]

setup(
    name='ovos-diagnostics',
    version='0.0.1',
    packages=['ovos_diagnostics'],
    url='',
    license='',
    author='JarbasAI',
    author_email='jarbasai@mailfence.com',
    description='',
    install_requires=required('requirements.txt'),
    entry_points={
        'console_scripts': ['ovos-diagnostics=ovos_diagnostics:cli']
    }
)
