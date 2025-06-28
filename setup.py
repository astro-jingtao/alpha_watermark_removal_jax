from setuptools import setup, find_packages

setup(name="awrjax",
      version="0.0.1",
      author="Tao Jing",
      author_email="jingt20@mails.tsinghua.edu.cn",
      description="",
      packages=find_packages(),
      python_requires='>=3.9',
      install_requires=['jax', 'numpy', 'opencv-python'])
