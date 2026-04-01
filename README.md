# LeWorldModel (Kaggle Edition)

### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes\*](https://x.com/lucasmaes_), [Quentin Le Lidec\*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

> **Note:** This is a customized fork of the original LeWM repository designed specifically for **Kaggle**. If you don't have access to a local GPU, you can use the instructions below to train this state-of-the-art World Model entirely in the cloud for free with the kaggle T4.

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With \~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

\<p align="center"\>
\<b\>[ \<a href="[https://arxiv.org/pdf/2603.19312v1](https://arxiv.org/pdf/2603.19312v1)"\>Paper\</a\> | \<a href="[https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY\_e5nh3XT4r4e?usp=sharing](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing)"\>Checkpoints\</a\> | \<a href="[https://huggingface.co/collections/quentinll/lewm](https://huggingface.co/collections/quentinll/lewm)"\>Data\</a\> | \<a href="[https://le-wm.github.io/](https://le-wm.github.io/)"\>Website\</a\> ]\</b\>
\</p\>

-----

## 🚀 Running on Kaggle (No Local GPU Required)

You can run this entire training pipeline using a free Kaggle GPU notebook. Open a new Kaggle Notebook, ensure the **Accelerator** is set to  GPU (T4x2), and run the following blocks in sequential cells.

### Step 1: Environment Setup & Installation

Clone this repository, install the required packages, and set up the necessary directory structures.

```python
# Clone the repository and move into it
!git clone https://github.com/RodrickKamsiyonna/le-wm_kaggle.git
%cd le-wm_kaggle

# Install dependencies
!pip install hydra-core stable_worldmodel stable_pretraining lightning huggingface_hub

# Set up environment variables and directories
import os
os.makedirs("/kaggle/working/stablewm", exist_ok=True)
os.makedirs("/kaggle/data", exist_ok=True)

os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"
os.environ["HYDRA_FULL_ERROR"] = "1" 

print("STABLEWM_HOME set to:", os.environ["STABLEWM_HOME"])
```

### Step 2: Download & Extract the Dataset

This downloads the `tworoom` dataset directly from HuggingFace and extracts it to the correct cache directory.

```python
from huggingface_hub import hf_hub_download
import os

# Download the compressed dataset
path = hf_hub_download(
    repo_id="quentinll/lewm-tworooms",
    filename="tworoom.tar.zst",
    repo_type="dataset",
    local_dir="/kaggle/working/data"
)
print("Downloaded to:", path)

# Move the file to our dedicated data folder
!mv /kaggle/working/data/tworoom.tar.zst /kaggle/data/tworoom.tar.zst

# Install zstd for extraction
!apt-get update && apt-get install -y zstd

# Extract the .h5 files into STABLEWM_HOME
!tar --zstd -xvf /kaggle/data/tworoom.tar.zst -C /kaggle/working/stablewm/

# Verify the extraction was successful
!find /kaggle/working/stablewm/ -name "*.h5"
```

### Step 3: Configure Weights & Biases (WandB)

Training relies heavily on WandB for logging. You must securely provide your API key using Kaggle Secrets and update the config file to point to your specific WandB entity (team/username).

```python
import wandb
from kaggle_secrets import UserSecretsClient

# Retrieve your WandB API key from Kaggle Secrets
# Make sure you have added a secret named 'wandb_key' in the Kaggle side panel!
user_secrets = UserSecretsClient()
wandb_api_key = user_secrets.get_secret("wandb_key")
wandb.login(key=wandb_api_key)

# Update the Hydra config to use your specific WandB entity
filepath = "/kaggle/working/le-wm_kaggle/config/train/lewm.yaml"

with open(filepath, "r") as f:
    content = f.read()

# Replace with your actual WandB team/username
content = content.replace("entity: lewm", "entity: rodrickkamsi2-afe-babalola-university")

with open(filepath, "w") as f:
    f.write(content)

print("Config updated. Verifying WandB entity:")
for line in content.split("\n"):
    if "entity" in line:
        print(" ->", line.strip())
```

### Step 4: Start Training\!

Launch the training script. Checkpoints will be automatically saved to your `/kaggle/working/stablewm` directory.

```bash
!python train.py data=tworoom
```

-----

## Local Installation (For standard setups)

If you are running this locally on your own hardware, the setup is slightly different:

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training.

**Installation:**

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data Management

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/` locally, or `/kaggle/working/stablewm` on Kaggle). Dataset names are specified without the `.h5` extension in the configs.

## Planning & Evaluation

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

\<div align="center"\>

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm\_noprop | ✓ | ✓ | ✓ | ✓ |

\</div\>

## Loading a checkpoint via API

Each tar archive contains two files per checkpoint:

  * `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use.
  * `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance.

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:

  * `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix.
  * `cache_dir` — optional override for the checkpoint root.

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions

Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues) on the original repository\! For questions or collaborations, please contact `lucas.maes@mila.quebec`

If you find this code useful, please reference it in your paper:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
