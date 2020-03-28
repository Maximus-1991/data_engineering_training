from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='image_recognition_maarten',
    version='0.1.0',
    author='maarten',
    author_email='maarten.s1991@gmail.com',
    description='Package to classify cifar-10 images',
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Maximus-1991/data_engineering_training.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
