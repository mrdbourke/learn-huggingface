# Glossary

* Transformer = A deep learning model that adopts the attention mechanism to draw global dependencies between input and output
* Tokenization = turning a series of data (text or image) into a series of tokens, where a token is a numerical representation of the input data, for example, in the case of text, tokenization could mean turning the words in a sentence into numbers (e.g. "hello world" -> [101, 102])
* Tokens = a token is a letter, word or word-piece (word) that a model uses to represent input data, for example, in the case of text, a token could be a word (e.g. "hello") or a word-piece (e.g. "hell" and "o"), see: https://platform.openai.com/tokenizer for an example
* `transformers` = A Python library by Hugging Face that provides a wide range of pre-trained transformer models, fine-tuning tools, and utilities to use them 
* `datasets` = A Python library by Hugging Face that provides a wide range of datasets for NLP and CV tasks
* `tokenizers` = A Python library by Hugging Face that provides a wide range of tokenizers for NLP tasks
* `evaluate` = A Python library by Hugging Face with premade evaluation functions for various tasks
* `torch` = PyTorch, an open-source machine learning library
* Hugging Face Hub = Place to store datasets, models, and other resources of your own + find existing datasets, models & scripts others have shared
* Hugging Face Spaces = A platform to share and run machine learning apps/demos, usually built with Gradio or Streamlit
* HF = Hugging Face
* NLP = Natural Language Processing
* CV = Computer Vision
* TPU = Tensor Processing Unit
* GPU = Graphics Processing Unit
* Prediction probability = the probability of a model's prediction for a given input, is a score between 0 and 1 with 1 being the highest, for example, a model may have a prediction probability of 0.95, this would mean it's quite confident with its prediction but it doesn't mean it's correct. A good way to inspect potential issues with a dataset is to show examples in the test set which have a high prediction probability but are wrong (e.g. pred prob = 0.98 but the prediction was incorrect).
* Hugging Face Pipeline (`pipeline`)  = A high-level API for using model for various tasks (e.g. `text-classification`, `audio-classification`, `image-classification`, `object-detection` and more), see the docs: https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/pipelines#transformers.pipeline 