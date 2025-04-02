# How to get best performance on DSR1 in TRT-LLM

NVIDIA has announced world-record DeepSeek-R1 inference performance at NVIDIA GTC 2025. A single NVIDIA DGX system with eight NVIDIA Blackwell GPUs can achieve over 250 tokens per second per user or a maximum throughput of over 30,000 tokens per second on the massive, state-of-the-art 671 billion parameter DeepSeek-R1 model. [NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)

In this blog, we share the configrations and procedures about how to reproduce the number on both B200 and H200 with Pytorch workflow.

## B200 min-latency
Our benchmark results are based on **Batch = 1, ISL = 1K, OSL = 2K, num_requests = 10 from real dataset**

### Prerequisites

``` bash
# Prerequisites
apt-get update && apt-get -y install git git-lfs
git lfs install

# Improve GPU performance
sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4

# Replace with your actual path
YOUR_WORK_PATH=<YOUR_WORK_PATH>
YOUR_MODEL_PATH=<YOUR_MODEL_PATH>
YOUR_DATA_PATH=<YOUR_DATA_PATH>

# Clone the TensorRT-LLM repository
cd $YOUR_WORK_PATH
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# Clone the DeepSeek-R1-FP4 model
cd $YOUR_MODEL_PATH
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/nvidia/DeepSeek-R1-FP4
git lfs pull  # Download the full model weight will take a long time
```
**Note**: Replace `<*_PATH>` to your actual path. 

### Build Docker 
Create a docker and run:

``` bash
cd TensorRT-LLM
make -C docker jenkins_run LOCAL_USER=1 DOCKER_RUN_ARGS="-v $YOUR_MODEL_PATH:$YOUR_MODEL_PATH:ro -v $YOUR_DATA_PATH:$YOUR_DATA_PATH:ro -v $YOUR_WORK_PATH:$YOUR_WORK_PATH"
```
Here we set `LOCAL_USER=1` argument to set up the local user account inside the container.

### Compile and Install
Here we compile the source inside the container:

``` bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures "100-real"  --python_bindings --clean
```

Install and set environment variables:

```bash
pip install --user build/tensorrt_llm*.whl
export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=`pwd`
```


### Benchmark
To do the benchmark, run the following command:

```bash
export TRTLLM_ENABLE_PDL=1

DS_R1_NVFP4_ALLMOE_MODEL_PATH=$YOUR_MODEL_PATH/DeepSeek-R1-FP4
trtllm-bench --model deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_NVFP4_ALLMOE_MODEL_PATH \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --num_requests 10 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 4 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

Explanation:
- `trtllm-bench`: A CLI packags benchmarking utility that aims to make it easier for users to reproduce our officially published. [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).
- `--dataset`: Prompt dataset used to benchmark. our official benchmark dataset has ISL = 1K, OSL = 2K
- `--backend`: Inference backend. Here we use Pytorch backed. 
- `--tp 8`: Tensor parallel size is 8.
- `--ep 4`: Expert parallel size is 4.
- `--extra_llm_api_options`: Used to specify some extra config. The content of the file is as follows:

    ``` yaml
    pytorch_backend_config:
        enable_overlap_scheduler: true
        use_cuda_graph: true
    speculative_config:
        decoding_type: MTP
        num_nextn_predict_layers: 3
    ```


### Expected Result Format
The perf might be different from different datasets and machines

``` 
===========================================================                     
= PERFORMANCE OVERVIEW                                                               
===========================================================                                                                                                               
Request Throughput (req/sec):                     0.1222                       
Total Output Throughput (tokens/sec):             250.0612                     
Per User Output Throughput (tokens/sec/user):     71.5615                
Per GPU Output Throughput (tokens/sec/gpu):       31.2576                
Total Latency (ms):                               81839.9807                       
Average request latency (ms):                     45318.9080
```

## B200 max-throughput

## H200 min-latency

## H200 max-throughput


