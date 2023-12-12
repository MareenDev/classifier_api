from setuptools import setup, find_packages

def fetch_requirements(filename):
    requirements = []
    with open(filename) as f:
        for ln in f.read().split("\n"):
            ln = ln.strip()
            if '--index-url' in ln:
                ln = ln.split('--index-url')[0].strip()
            requirements.append(ln)
        return requirements

setup(
    name="classifier API",
    version=0.1,
    author="Anderson, Elisa Mareen",
    description="API for Fashion MNIST classifier",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Flask, API, FashionMNIST",
    packages=find_packages(),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.10.0",
)
