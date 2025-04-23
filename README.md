This is our source codes with test scripts the paper "A model and package for German ColBERT" and the package colbert-kit.

# Installation

To install the package, you have three options depending on your FAISS and CUDA setup. The key difference between the options lies in whether you want to use FAISS on CPU or GPU, and which CUDA version your system supports.

If you are not using CUDA 11.x or 12.x, or if you do not need GPU acceleration for indexing and searching, install the CPU version:

<pre> pip install colbert-kit[cpu] </pre>

If you are using CUDA 11.x and want to enable FAISS GPU support:

<pre> pip install colbert-kit[gpu-cu11] </pre>

If you are using CUDA 12.x, install the version built for CUDA 12:

<pre> pip install colbert-kit[gpu-cu12] </pre>

**Note:** HNSW indexing is currently not supported on GPU in FAISS. GPU acceleration works well for Flat and IVF-based indexes.

# Documentation

The documentation folder contains detailed explanations about the background of the package, along with sample code snippets to help you understand and use the model effectively.

# Test scripts

All test scripts are located in the test_scripts folder. These scripts demonstrate how to use the package in practice.

To run a test script, follow these steps:

<pre> cd test_scripts </pre>

<pre> python test/script/you/want/to/run.py </pre>

# Source codes of the package

The source code for the package is located in the colbert_kit_src folder. This directory contains all core components and detailed parameter definitions for ColBERT-based retrieval.

# Model

You can download and place our ColBERT model inside the model folder. This model will be used by the test scripts for evaluation and retrieval.

# Test data

The data_test folder contains test data. Currently, we include the miracl-de-dev dataset, which consists of:

queries.csv

docs.csv

qrels.csv

# Folder to save index

FAISS indices can be saved in the indices_test folder. This allows reusing pre-built indexes for faster testing and retrieval.