# Getting setup for the Hugging Face ecosystem

## TK - Start here (universal steps)

1. Create a free Hugging Face account at <https://huggingface.co/join>.
2. Create a Hugging Face access token with read and write access at <https://huggingface.co/settings/tokens>.
    * You can create a read/write token using the fine-grained settings and selecting the appropriate options.
    * Read more on Hugging Face access tokens at <https://huggingface.co/docs/hub/en/security-tokens>. 

## TK - Getting setup on Google Colab

1. Follow the steps in Start here.
2. Add your Hugging Face read/write token as a Secret in Google Colab.

TK image - show image for loading a secret in Google Colab

If you need to force relogin for a notebook session, you can run:

```python
import huggingface_hub

# Login to Hugging Face
huggingface_hub.login()
```

And enter your token in the box that appears (**note:** this token will only be active for the current notebook session and will delete when your Google Colab instance terminates).

## TK - Getting started locally

1. Follow the steps in Start here.
2. Install the Hugging Face CLI with `pip install -U "huggingface_hub[cli]"`.
3. Follow the setup steps mentioned in <https://huggingface.co/docs/huggingface_hub/en/guides/cli>. 

## Installing Hugging Face libraries

We'll need to install the following libraries from the Hugging Face ecosystem:

* [`transformers`](https://huggingface.co/docs/transformers/en/installation) - comes pre-installed on Google Colab but if you're running on your local machine, you can install it via `pip install transformers`.
* [`datasets`](https://huggingface.co/docs/datasets/installation) - a library for accessing and manipulating datasets on and off the Hugging Face Hub, you can install it via `pip install datasets`.
* [`evaluate`](https://huggingface.co/docs/evaluate/installation) - a library for evaluating machine learning model performance with various metrics, you can install it via `pip install evaluate`.
* [`accelerate`](https://huggingface.co/docs/accelerate/basic_tutorials/install) - a library for training machine learning models faster, you can install it via `pip install accelerate`.
* [`gradio`](https://www.gradio.app/guides/quickstart#installation) - a library for creating interactive demos of machine learning models, you can install it via `pip install gradio`.

