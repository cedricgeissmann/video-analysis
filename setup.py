from setuptools import setup, find_packages

setup(
    name='video-analysis',
    version='0.0.1',
    author='Cedric Geissmann',
    author_email='cedric.geissmann@gmail.com',
    install_requires=('opencv-python==4.5.3.56', 'mediapipe==0.8.6.1'),
    package_dir={"": 'src'},
    packages=find_packages()
)
