from setuptools import setup, find_packages

setup(
    name="cr_detection",
    version="0.1.0",
    description="Clash Royale game state detection for reinforcement learning",
    author="Will",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "pyautogui>=0.9.54",
        "pyobjc-framework-Quartz>=9.0",
    ],
)
