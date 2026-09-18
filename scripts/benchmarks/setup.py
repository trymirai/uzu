from setuptools import Extension, setup

setup(ext_modules=[Extension("mach._mach", sources=["src/mach/_mach.c"])])
