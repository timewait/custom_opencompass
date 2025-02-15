# Overview
This codebase is a modified version of the OpenCompass framework, specifically tailored to include a new class, HuggingFaceNoiseModel, which introduces the capability to inject Gaussian noise into the logits during the generation process of a Hugging Face model. This modification is built on top of the existing HuggingFaceBaseModel class, which is part of the OpenCompass library.

# Key Modifications
Introduction of HuggingFaceNoiseModel:

The HuggingFaceNoiseModel class extends the HuggingFaceBaseModel class to support the injection of Gaussian noise into the logits during text generation.

This is achieved by adding a GaussianNoiseLogitsProcessor to the LogitsProcessorList in the generation process.

Gaussian Noise Injection:

The GaussianNoiseLogitsProcessor class is responsible for adding Gaussian noise to the logits (scores) during the generation process.

The noise is sampled from a normal distribution with a configurable mean and standard deviation (std).

Integration with Generation Configuration:

The HuggingFaceNoiseModel class checks for the presence of a noise_std parameter in the generation_kwargs. If found, it initializes the GaussianNoiseLogitsProcessor with the specified standard deviation and adds it to the LogitsProcessorList.

The noise_std parameter is then removed from generation_kwargs to avoid conflicts during the generation process.

OpencCompass Config:

THe main script is eval_noise_llama3.py, which contains the model, datasets, and std config. You can reproduce our experiments following Usage step by step.


# Usage
To use the HuggingFaceNoiseModel class, follow these steps:

## 🛠️ Installation

Below are the steps for quick installation and datasets preparation.

### 💻 Environment Setup

We highly recommend using conda to manage your python environment.

- #### Create your virtual environment

  ```bash
  conda create --name opencompass python=3.10 -y
  conda activate opencompass
  ```

- #### Install OpenCompass via pip

  ```bash
    pip install -U opencompass

    ## Full installation (with support for more datasets)
    # pip install "opencompass[full]"

    ## Environment with model acceleration frameworks
    ## Manage different acceleration frameworks using virtual environments
    ## since they usually have dependency conflicts with each other.
    # pip install "opencompass[lmdeploy]"
    # pip install "opencompass[vllm]"

    ## API evaluation (i.e. Openai, Qwen)
    # pip install "opencompass[api]"
  ```

- #### Install OpenCompass from source

  If you want to use opencompass's latest features, or develop new features, you can also build it from source

  ```bash
    git clone https://github.com/open-compass/opencompass opencompass
    cd opencompass
    pip install -e .
    # pip install -e ".[full]"
    # pip install -e ".[vllm]"
  ```
- ### Run command
  ```bash
    python -u run.py configs/eval_noise_llama3.py
  ```
