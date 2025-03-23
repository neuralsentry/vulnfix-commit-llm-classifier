![NeuralSentry Logo](.github/img/neuralsentry-full-light.png)

# vulnfix-commit-llm-classifier

> **Note**
> 
> Find out more about the project here:
> 
> https://sopossible.sp.edu.sg/student-life/annual-events/sp-industry-innovation/spii2024/digital-transformation

`vulnfix-commit-llm-classifier` is a tool that clones remote Git repositories, extracts commit information, and classifies commits using NeuralSentry's [vulnerability-fix commit (VFC) classification model](https://huggingface.co/neuralsentry/vulnfixClassification-StarEncoder-DCM-Balanced).
![Demo Gif](.github/img/demo.gif)

## Installation

```bash
# Python 3.11+ and pip is required!
# Optional (venv)
python3 -m venv venv
source venv/bin/activate

# Install Pytorch
# See: https://pytorch.org/get-started/locally/
# For systems with NVIDIA GPUs:
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# For systems without NVIDIA GPUs:
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Built-in Help

```bash
python main.py --help
```

## Examples

### Windows

```ps1
python main.py `
  -i data/examples.txt `
  --output data/output.csv `
  --bugfix-threshold 0.95 `
  --batch-size 32 `
  --after "2023-01-01"
```

### Linux

```bash
python main.py \
  -i data/examples.txt \
  --output data/output.csv \
  --bugfix-threshold 0.95 \
  --batch-size 32 \
  --after "2023-01-01"
```
