import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    package_dir = {"": "pkgs"},
    packages = setuptools.find_packages(where="pkgs"),
    long_description = long_description
)
