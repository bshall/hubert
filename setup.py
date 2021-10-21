from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="hubert",
    version="0.1",
    author="Benjamin van Niekerk",
    author_email="benjamin.l.van.niekerk@gmail.com",
    url="https://github.com/bshall/hubert",
    description="The HuBERT-Soft and HuBERT-Discrete models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/bshall/hubert",
        "Samples": "https://bshall.github.io/soft-vc/",
    },
    keywords="Speech Synthesis, Voice Conversion, PyTorch",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["hubert"],
    python_requires=">=3.8",
    install_requires=[
        "tensorboard==2.7.0",
        "torch==1.9.1",
        "torchaudio==0.9.1",
        "tqdm==4.62.3",
    ],
)
