from setuptools import setup, find_packages

setup(
    name="supply-chain-rl",
    version="1.0.0",
    description="Reinforcement Learning for Supply Chain Management",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=1.6.0",
        "torch>=1.12.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "networkx>=2.6.0",
        "PyYAML>=6.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
