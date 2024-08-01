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

