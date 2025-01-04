from setuptools import setup, find_packages

setup(
    name="data_quality_assessment",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "datasets>=2.0.0",
        "tqdm>=4.0.0",
        "pyyaml>=6.0.0",
    ],
)
