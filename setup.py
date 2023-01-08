from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.md
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'), encoding='utf-16') as f:
    requirements = f.read().splitlines()

setup(
    name='migperf',
    version='0.0.1',
    description='Multi-Instance-GPU profiling tool',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Xing Di',
    author_email='xing.cyrildi@gmail.com',
    url='https://github.com/MLSysOps/MIGProfiler',
    download_url='https://github.com/MLSysOps/MIGProfiler/archive/master.zip',
    keywords=['benchmark', 'deep learning', 'MLOps', 'neural networks'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        "Operating System :: OS Independent",
    ],
)
