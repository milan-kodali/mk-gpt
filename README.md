## mk-gpt
learning about llms

## Set up venv
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## GPU Tips
- Monitor status: `watch nvidia-smi`
- For [A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) B=48 seems to maximize available memory given T=1024, when using torch.autocast to BF16
- Runpod workflow
  - Generate SSH keys
  - Add keys to Runpod console
  - Create a Pod (eg 2x A100) with SSH terminal access (no Jupyter notebook)
  - Add to local SSH config (below)
  - Connect & clone repo to /workspace/ (preserved across instance stop/start)

```
Host runpod
    HostName xxx.xxx.xxx.x
    User root
    Port xxxxx
    IdentityFile ~/.ssh/xxxxxxxx
```
