# 🧠 Minari Dataset Setup for MuJoCo InvertedDoublePendulum

This repository sets up and downloads datasets from [Minari](https://minari.farama.org) for the **MuJoCo InvertedDoublePendulum** environment.

## 📦 Environment Setup


### 0. Virtual Environment Setup
```bash
python -m venv venv
source ./venv/bin/activate
```

### 1. Clone the Minari repository and Initial Setup
```bash
git clone https://github.com/Farama-Foundation/Minari.git --single-branch
cd Minari
pip install -e ".[all]"
```

### 2. Download Datasets
```bash
minari download mujoco/inverteddoublependulum/expert-v0
minari download mujoco/inverteddoublependulum/medium-v0
```

### 3. Check Downloaded Datasets
```bash
minari list local
```


### 4. Install additional dependencies
```bash
pip install torch Pillow
```

### 5. Code Execution
```bash
cd ../
python main.py
```

## 📚 Reference  
- [Minari Documentation – Basic Usage](https://minari.farama.org/content/basic_usage/)  
- [Farama Foundation GitHub](https://github.com/Farama-Foundation)  
