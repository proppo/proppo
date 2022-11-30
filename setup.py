from setuptools import setup, find_packages


setup(name="proppo",
      version="0.1",
      description="Custom propagations for backpropagation libraries such as PyTorch.",
      url="https://github.com/proppo/proppo",
      install_requires=["torch", "tqdm", "numpy"],
      packages=["proppo", "proppo.modules"])
