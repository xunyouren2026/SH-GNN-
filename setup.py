from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sh-gnn",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Spherical Harmonic Graph Neural Network with strict SO(3) equivariance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourhub/sh-gnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch_geometric>=2.3.0",
        "torch_scatter",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "onnx>=1.13.0",
        "onnxruntime>=1.14.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pydantic>=2.0.0",
        "tensorboard>=2.11.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "tensorrt": ["tensorrt>=8.5"],
        "dev": ["pytest", "black", "isort", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "sh-gnn=sh_gnn.cli:main",
        ],
    },
)

# 允许直接运行 python setup.py（但不推荐）
if __name__ == "__main__":
    print("请使用: pip install -e . 安装本包")
    input("按回车键退出...")