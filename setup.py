from setuptools import setup, find_packages

setup(
    name="gemma3n-video",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers==4.41.0",
    ],
)