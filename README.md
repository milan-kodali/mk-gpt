## mk-gpt
learning about llms

## Set up venv
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## GPU Tips
- Check status: `nvidia-smi` or `watch -n 1 nvidia-smi`
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
