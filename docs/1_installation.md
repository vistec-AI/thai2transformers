
## Installation 

<br>

### 1) Manual installation

<br>

1) PyTorch

    In this repository, we use PyTorch as a framework to train langauage model. The version of PyTorch that we used is 1.5.0 with CUDA 10.2.

    ```
    pip install torch==1.5.0
    ```

2) SentencePiece

    In order to manually build SentencePiece model from raw text files, it is required to install SentencePiece from source. (ref: https://github.com/google/sentencepiece#c-from-source)

    ```
    apt-get update
    apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece
    mkdir build
    cd build
    cmake ..
    make -j $(nproc)
    make install
    ldconfig -v
    ```
    On OSX/macOS, replace the last command with sudo update_dyld_shared_cache


    To use trained SentencePiece model, you can only install sentencepice via pip.

    ```
    pip install sentencepiece==0.1.94
    ```


3) Huggingface's `transformers` 


    Currently, we use the library from huggingface.co namely [transformers](https://github.com/huggingface/transformers) to pretrain our Thai language models.

    `transformers` can be installed via pip. (the version of transformers we used is 3.4.0)

    ```
    pip install transformers==3.4.0
    ```


    For faster training on GPUs with PyTorch (0.4 or newer), install Nvidia's apex library (https://github.com/NVIDIA/apex). apex can be installed with CUDA and C++ extensions (for performance and full functionality).

    ```
    git clone https://github.com/NVIDIA/apex.git
    cd apex

    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
