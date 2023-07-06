![neuralsentry logo](.github/img/neuralsentry-full-dark.png)
# commit-bugfix-classifier

## Installation
*Requires installation of Python and pip as prerequisites*

**Windows**


```ps1
# Optional (venv)
python -m venv venv
venv\Scripts\activate

# Install Pytorch
# See: https://pytorch.org/get-started/locally/
# For systems with NVIDIA GPUs:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For systems without NVIDIA GPUs:
pip3 install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

**Linux**

```bash
# Optional (venv)
python -m venv venv
source venv/bin/activate

# Install Pytorch
# See: https://pytorch.org/get-started/locally/
# For systems with NVIDIA GPUs:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For systems without NVIDIA GPUs:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```
## Usage
**Windows**
```ps1
python main.py `
  -i data/examples.txt `
  --output data/output.csv `
  --bugfix-threshold 0.95 `
  --batch-size 32 `
  --after "2023-01-01"
```

**Linux**
```bash
python main.py \
  -i data/examples.txt \
  --output data/output.csv \
  --bugfix-threshold 0.95 \
  --batch-size 32 \
  --after "2023-01-01"
```
