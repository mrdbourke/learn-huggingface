# Glossary

* Transformer = A deep learning model that adopts the attention mechanism to draw global dependencies between input and output
* `transformers` = A Python library by Hugging Face that provides a wide range of pre-trained transformer models, fine-tuning tools, and utilities to use them 
* `datasets` = A Python library by Hugging Face that provides a wide range of datasets for NLP and CV tasks
* `tokenizers` = A Python library by Hugging Face that provides a wide range of tokenizers for NLP tasks
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