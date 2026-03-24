---
title: "Setup 🤗"
description: "Get your Google Colab instance/local environment ready for the Learn Hugging Face course."
toc: true
toc-depth: 3
code-copy: true    
---

# Getting setup for the Hugging Face ecosystem

The following steps are to help you get started with the Hugging Face ecosystem.

Best to follow the "Start here" steps and then go through the other setup steps as necessary.

## Start here (universal steps)

1. Create a free Hugging Face account at <https://huggingface.co/join>.
2. Create a Hugging Face access token with read and write access at <https://huggingface.co/settings/tokens>.
    * You can create a read/write token using the fine-grained settings and selecting all the appropriate options.
    * Read more on Hugging Face access tokens at <https://huggingface.co/docs/hub/en/security-tokens>.

<figure style="text-align: center;">
    <!-- figtemplate -->
    <img src="https://huggingface.co/datasets/mrdbourke/learn-hf-images/resolve/main/setup/00-creating-huggingface-token.png"
     alt="Two screenshots of the Hugging Face website showing the process of creating an access token. The first screenshot displays the 'Access Tokens' page with an existing token and a highlighted '+ Create new token' button. The second screenshot shows the 'Create new Access Token' page, where the user can specify the token name and permissions for repositories, webhooks, inference, collections, discussions, posts, and billing." 
     style="width: 100%; max-width: 900px; height: auto;"/>
     <figcaption>To read from and write to your Hugging Face Hub account, you'll need to set up an access token. You can have one token for reading and one for writing. However, I personally use a single token for reading and writing. </figcaption>
</figure>

**Note:** Do not share your token with others. Always keep it private and avoid saving it in raw text format.

## Getting setup on Google Colab

**Note:** If you're unfamiliar with Google Colab, I'd recommend going through Sam Witteveen's video [Colab 101](https://youtu.be/Ii6gs9zADEA?si=qbuRpFKTptd30tXz) and then [Advanced Colab](https://youtu.be/ieLpZ4wnb4A?si=AK1VecZck-BHoee6) to learn more.

1. Follow the steps in Start here.
2. Add your Hugging Face read/write token as a Secret in Google Colab.
    * Naming this Secret `HF_TOKEN` will mean that Hugging Face libraries automatically recognize your token for future use.

<figure style="text-align: center;">
    <!-- figtemplate -->
    <img src="https://huggingface.co/datasets/mrdbourke/learn-hf-images/resolve/main/setup/01-adding-huggingface-token-to-google-colab.png"
     alt="Two screenshots illustrating how to add a Hugging Face token to Google Colab. The first screenshot shows the Hugging Face 'Access Token' page with an existing token and a highlighted token name. The second screenshot shows the Google Colab 'Secrets' tab with a list of stored secrets, including the Hugging Face token, with options to add new secrets or manage existing ones" 
     style="width: 100%; max-width: 900px; height: auto;"/>
     <figcaption>For accessing models and datasets from the Hugging Face Hub (both read and write) inside Google Colab, you'll need to add your Hugging Face token as a Secret in Google Colab. Once you give your Google Colab notebook access to the token, it can be used by Hugging Face libraries to interact with the Hugging Face Hub.</figcaption>
</figure>

Alternatively, if you need to force relogin for a notebook session, you can run:

```python
import huggingface_hub # requires !pip install huggingface_hub

# Login to Hugging Face
huggingface_hub.login()
```

And enter your token in the box that appears (**note:** this token will only be active for the current notebook session and will delete when your Google Colab instance terminates).

## Getting started locally

1. Follow the steps in Start here.
2. Follow your specific hardware steps below.

| Hardware | Package Manager | Backend | Setup Guide |
|----------|----------------|---------|-------------|
| NVIDIA GPU | Conda | CUDA | [NVIDIA GPU + Conda](#nvidia-gpu--conda-local-setup) |
| NVIDIA GPU | uv (pip) | CUDA | [NVIDIA GPU + uv](#nvidia-gpu--uv-pip-local-setup) |
| macOS (Apple Silicon) | Conda | MPS | [macOS + Conda](#macos--conda-local-setup) |
| macOS (Apple Silicon) | uv (pip) | MPS | [macOS + uv](#macos--uv-pip-local-setup) |

## Global Hugging Face library requirements

Depending on your environment/local hardware, there are a handful of foundation libraries we'll need to install from the Hugging Face ecosystem:

* [`transformers`](https://huggingface.co/docs/transformers/en/installation) - comes pre-installed on Google Colab but if you're running on your local machine, you can install it via `pip install transformers`.
* [`datasets`](https://huggingface.co/docs/datasets/installation) - a library for accessing and manipulating datasets on and off the Hugging Face Hub, you can install it via `pip install datasets`.
* [`evaluate`](https://huggingface.co/docs/evaluate/installation) - a library for evaluating machine learning model performance with various metrics, you can install it via `pip install evaluate`.
* [`accelerate`](https://huggingface.co/docs/accelerate/basic_tutorials/install) - a library for training machine learning models faster, you can install it via `pip install accelerate`.
* [`gradio`](https://www.gradio.app/guides/quickstart#installation) - a library for creating interactive demos of machine learning models, you can install it via `pip install gradio`.

## NVIDIA GPU + Conda local setup

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) to get the conda package manager.

### Clone the course repository

```bash
git clone https://github.com/mrdbourke/learn-huggingface
```

Change into the target directory:

```bash
cd learn-huggingface
```

### Create and activate conda environment

Create environment:

```bash
conda create -n learn-hf python=3.12 -y
```

**Note:** This setup has been test with Python 3.12. If you'd like, you can use a different/later version. 

Activate it:

```bash
conda activate learn-hf
```

### Install PyTorch 

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Install Hugging Face CLI and Login

Install Hugging Face CLI:

```
python -m pip install -U "huggingface_hub[cli]"
```

Login with your Hugging Face account to authenticate your local machine:

```bash
hf auth login
```

### Install dependencies

Install dependenies we'll need for the projects:

```bash
python -m pip install transformers datasets evaluate accelerate gradio trl matplotlib jupyter
```

**Note:** If you run into any dependency issues during running the projects, you can always install them via `pip install [DEPENDENCY_NAME]`. 

### Check that the imports work

```bash
python -c "
import torch, transformers, datasets, accelerate, gradio, trl, matplotlib, huggingface_hub

assert torch.cuda.is_available(), 'CUDA GPU not available'

print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
print('cuda_device', torch.cuda.get_device_name(0))

x = torch.tensor([1.0, 2.0]).to('cuda')
print('cuda_tensor_device', x.device)

print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('gradio', gradio.__version__)
print('trl', trl.__version__)
print('matplotlib', matplotlib.__version__)
print('huggingface_hub', huggingface_hub.__version__)

print('Conda env ready! Good to code!')
"
```

If these work, we're good to go!!

### Get started

**Option A: Jupyter Lab**

```bash
jupyter lab
```

**Option B: VS Code**

Open the project in VS Code:

```bash
code .
```

**Note:** This requires [VS Code](https://code.visualstudio.com/) installed locally with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) so you can run `.ipynb` notebooks directly in VS Code.

Alternatively, you can also start writing Python scripts to follow along and learn.

## NVIDIA GPU + uv (pip) local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to get a fast Python package manager.

### Clone the course repository

```bash
git clone https://github.com/mrdbourke/learn-huggingface
```

Change into the target directory:

```bash
cd learn-huggingface
```

### Create and activate virtual environment

Create environment:

```bash
uv venv learn-hf --python 3.12
```

**Note:** This setup has been tested with Python 3.12. If you'd like, you can use a different/later version.

Activate it:

```bash
source learn-hf/bin/activate
```

### Install PyTorch

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Install Hugging Face CLI and Login

Install Hugging Face CLI:
```bash
uv pip install -U "huggingface_hub[cli]"
```

Login with your Hugging Face account to authenticate your local machine:
```bash
huggingface-cli login
```

### Install dependencies

Install dependencies we'll need for the projects:

```bash
uv pip install transformers datasets evaluate accelerate gradio trl matplotlib jupyter
```

**Note:** If you run into any dependency issues during running the projects, you can always install them via `uv pip install [DEPENDENCY_NAME]`.

### Check that the imports work

```bash
python -c "
import torch, transformers, datasets, accelerate, gradio, trl, matplotlib, huggingface_hub

assert torch.cuda.is_available(), 'CUDA GPU not available'

print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
print('cuda_device', torch.cuda.get_device_name(0))

x = torch.tensor([1.0, 2.0]).to('cuda')
print('cuda_tensor_device', x.device)

print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('gradio', gradio.__version__)
print('trl', trl.__version__)
print('matplotlib', matplotlib.__version__)
print('huggingface_hub', huggingface_hub.__version__)

print('uv env ready! Good to code!')
"
```

If these work, we're good to go!!

### Get started

**Option A: Jupyter Lab**

```bash
jupyter lab
```

**Option B: VS Code**

Open the project in VS Code:

```bash
code .
```

**Note:** This requires [VS Code](https://code.visualstudio.com/) installed locally with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) so you can run `.ipynb` notebooks directly in VS Code.

Alternatively, you can also start writing Python scripts to follow along and learn.

## macOS + Conda local setup

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) to get the conda package manager.

**Note:** macOS uses the MPS (Metal Performance Shaders) backend for GPU acceleration on Apple Silicon. Training on MPS is generally much slower than on NVIDIA GPUs with CUDA, however, inference works quite well. MPS is great for learning and experimentation but if you need faster training, consider using a cloud GPU (e.g. Google Colab) or an NVIDIA GPU machine.

### Clone the course repository

Clone the `learn-huggingface` repo:

```bash
git clone https://github.com/mrdbourke/learn-huggingface
```

Change into the target directory:

```bash
cd learn-huggingface
```

### Create and activate conda environment

Create environment:

```bash
conda create -n learn-hf python=3.12 -y
```

**Note:** This setup has been tested with Python 3.12. If you'd like, you can use a different/later version.

Activate it:

```bash
conda activate learn-hf
```

### Install PyTorch

```bash
python -m pip install torch torchvision
```

**Note:** On macOS, the default PyTorch install includes MPS (Metal Performance Shaders) support. No special index URL is needed.

### Install Hugging Face CLI and Login

Install Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
```

Login with your Hugging Face account to authenticate your local machine:

```bash
hf auth login
```

### Install dependencies

Install dependencies we'll need for the projects:

```bash
python -m pip install transformers datasets evaluate accelerate gradio trl matplotlib jupyter
```

**Note:** If you run into any dependency issues during running the projects, you can always install them via `pip install [DEPENDENCY_NAME]`.

### Check that the imports work

```bash
python -c "
import torch, transformers, datasets, accelerate, gradio, trl, matplotlib, huggingface_hub

assert torch.backends.mps.is_available(), 'MPS not available'

print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available())
print('mps_built', torch.backends.mps.is_built())

x = torch.tensor([1.0, 2.0]).to('mps')
print('mps_tensor_device', x.device)

print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('gradio', gradio.__version__)
print('trl', trl.__version__)
print('matplotlib', matplotlib.__version__)
print('huggingface_hub', huggingface_hub.__version__)

print('macOS Conda env ready! Good to code!')
"
```

If these work, we're good to go!!

### Get started

**Option A: Jupyter Lab**

```bash
jupyter lab
```

**Option B: VS Code**

Open the project in VS Code:

```bash
code .
```

**Note:** This requires [VS Code](https://code.visualstudio.com/) installed locally with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) so you can run `.ipynb` notebooks directly in VS Code.

Alternatively, you can also start writing Python scripts to follow along and learn.

## macOS + uv (pip) local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to get a fast Python package manager.

**Note:** macOS uses the MPS (Metal Performance Shaders) backend for GPU acceleration on Apple Silicon. Training on MPS is generally much slower than on NVIDIA GPUs with CUDA, however, inference works quite well. MPS is great for learning and experimentation but if you need faster training, consider using a cloud GPU (e.g. Google Colab) or an NVIDIA GPU machine.

### Clone the course repository

Clone the `learn-huggingface` repo:

```bash
git clone https://github.com/mrdbourke/learn-huggingface
```

Change into the target directory:

```bash
cd learn-huggingface
```

### Create and activate virtual environment

Create environment:

```bash
uv venv learn-hf --python 3.12
```

**Note:** This setup has been tested with Python 3.12. If you'd like, you can use a different/later version.

Activate it:

```bash
source learn-hf/bin/activate
```

### Install PyTorch

```bash
uv pip install torch torchvision
```

**Note:** On macOS, the default PyTorch install includes MPS (Metal Performance Shaders) support. No special index URL is needed.

### Install Hugging Face CLI and Login

Install Hugging Face CLI:

```bash
uv pip install -U "huggingface_hub[cli]"
```

Login with your Hugging Face account to authenticate your local machine:

```bash
huggingface-cli login
```

### Install dependencies

Install dependencies we'll need for the projects:

```bash
uv pip install transformers datasets evaluate accelerate gradio trl matplotlib jupyter
```

**Note:** If you run into any dependency issues during running the projects, you can always install them via `uv pip install [DEPENDENCY_NAME]`.

### Check that the imports work

```bash
python -c "
import torch, transformers, datasets, accelerate, gradio, trl, matplotlib, huggingface_hub

assert torch.backends.mps.is_available(), 'MPS not available'

print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available())
print('mps_built', torch.backends.mps.is_built())

x = torch.tensor([1.0, 2.0]).to('mps')
print('mps_tensor_device', x.device)

print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('gradio', gradio.__version__)
print('trl', trl.__version__)
print('matplotlib', matplotlib.__version__)
print('huggingface_hub', huggingface_hub.__version__)

print('macOS uv env ready! Good to code!')
"
```

If these work, we're good to go!!

### Get started

**Option A: Jupyter Lab**

```bash
jupyter lab
```

**Option B: VS Code**

Open the project in VS Code:

```bash
code .
```

**Note:** This requires [VS Code](https://code.visualstudio.com/) installed locally with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) so you can run `.ipynb` notebooks directly in VS Code.

Alternatively, you can also start writing Python scripts to follow along and learn.