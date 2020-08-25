import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LCADMM", # Replace with your own username
    version="0.0.1",
    author="Yuxiao Chen",
    author_email="chenyx@caltech.edu",
    description="Linear constrained ADMM",
    install_requires=["CVXOPT",
                      "CVXPY"
                      "numpy >= 1.9",
                      "scipy >= 0.15"],
    url="https://github.com/chenyx09/LCADMM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
