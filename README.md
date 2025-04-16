This is our source codes with test scripts the paper "TQColBERT: A model and package for German ColBERT" and the our package tqcolbert.

To install the package, you have three options depending on your FAISS and CUDA setup. The key difference between the options lies in whether you want to use FAISS on CPU or GPU, and which CUDA version your system supports.

If you are not using CUDA 11.x or 12.x, or if you do not need GPU acceleration for indexing and searching, install the CPU version:

<pre> ```pip install tqcolbert[cpu] ``` </pre>

If you are using CUDA 11.x and want to enable FAISS GPU support:

<pre> ```pip install tqcolbert[gpu-cu11] ``` </pre>

If you are using CUDA 12.x, install the version built for CUDA 12:

<pre> ```pip install tqcolbert[gpu-cu12] ``` </pre>

**Note:** HNSW indexing is currently not supported on GPU in FAISS. GPU acceleration works well for Flat and IVF-based indexes.