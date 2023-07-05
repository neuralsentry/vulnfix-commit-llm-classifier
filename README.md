# commit-bugfix-classifier

## Usage

**Windows**

```ps1
# Optional (venv)
python -m venv venv
venv\Scripts\activate

# Install Pytorch
# See: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117


# Install dependencies
pip install -r requirements.txt

# Run
python main.py -i data/examples.txt --output data/output.csv --bugfix-threshold 0.95 --batch-size 32 --after "2023-01-01"
```

**Linux**

```bash
# Optional (venv)
python -m venv venv
source venv/bin/activate

# Install Pytorch
# See: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt

# Run
python main.py -i data/examples.txt --output data/output.csv --bugfix-threshold 0.95 --batch-size 32 --after "2023-01-01"
```
