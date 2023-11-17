import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="popdynamics", # Replace with your own username
    version="0.0.1",
    description="Tools for simulating population dynamics of branching processes vs SDE model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        "numpy",
    ],
)
