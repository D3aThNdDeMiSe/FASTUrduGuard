# Environment setup

Tested with Python 3.10 / 3.11 on Windows 11 + WSL Ubuntu 22.04. Python 3.13
also works for the *non-CUDA* parts; many DL libraries don't yet ship 3.13
wheels — if you must use 3.13, expect bitsandbytes/peft/accelerate to lag.

## CPU-only quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

This is enough to run data prep, mBERT/DistilBERT/XLNet/RoBERTa training on
CPU, the aggregator, the LIME demo, and CI.

## RTX 5060 / 5060 Ti (Blackwell, sm_120)

Both cards report compute capability **12.0**. As of writing, this requires:

* CUDA Toolkit 12.8 or newer
* PyTorch 2.5+ with the `cu128` wheels (or nightly)
* `bitsandbytes` does **not** yet have official sm_120 builds. The training
  driver auto-detects this and falls back to FP16 LoRA, which still fits a
  base-size XLM-R fine-tune on 8 GB at `max_seq_len=256`.

Install the GPU stack (Windows, with miniconda):

```powershell
conda install -c conda-forge cuda-version=12.8 -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install -e .
# verify:
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If you see `CUDA error: no kernel image is available for execution on the device`,
your torch wheel does not yet include sm_120. Either upgrade torch nightly or
fall back to CPU mode for that model (set `placement: cpu` in `configs/models.yaml`).

## Two-node coordination

Both nodes use the **same Git repo**. Add SSH keys / a fine-grained PAT on each
node so `git pull` / `git push` works non-interactively. Recommended: one
private repository per project where both team members are collaborators.

```powershell
git config --global user.name  "<you>"
git config --global user.email "<you>@..."
```

Set the rank on each node before invoking any training script:

```powershell
$env:FUG_RANK="0"   # on the RTX 5060 box
# or
$env:FUG_RANK="1"   # on the RTX 5060 Ti box
```

A run takes its rank from `FUG_RANK`. Everything else is identical between
the two nodes.
