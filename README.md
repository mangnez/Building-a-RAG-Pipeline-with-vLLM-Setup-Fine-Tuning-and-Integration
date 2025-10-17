This repository acts as  detailed guide from setting up vLLM from scratch along with installing linux environment- Ubuntu to fine-tuning the LLM for... and finally integrating it into a RAG pipeline for developing an intelligent asisstant.

#1 INTRODUCTION 
- why use vLLM? in what ways is it better than other LLMS?
  
Developed at UC Berkley, Vllm was specifically designed to address speed and memory challenges that come with maintaining large language models.
according to IBM studies, 
Serving LLMs in production—especially on VMs or Kubernetes—requires massive computation per word, unlike traditional workloads. This leads to high costs, slow response times, and memory inefficiency. Traditional frameworks often hoard GPU memory, forcing over-provisioning. As user load increases, batch processing bottlenecks cause latency spikes. Scaling to large organizations exceeds single-GPU limits, making efficient deployment challenging.
UC Berkeley introduced vLLM, an open-source project designed to solve key challenges in serving large language models—like memory fragmentation, inefficient batch execution, and scalability. The core innovation is paged attention, which breaks the KV cache into smaller, manageable memory chunks—similar to virtual memory—allowing efficient access and reduced GPU strain. Combined with continuous batching, vLLM dramatically improves throughput (up to 24× faster than Hugging Face Transformers and TGI), lowers latency, and optimizes GPU usage for scalable, high-performance inference.

1. Continuous Batching
vLLM uses a novel scheduling algorithm that allows multiple requests to be batched dynamically.
This leads to higher throughput and lower latency, especially under concurrent loads.

2. Paged Attention
Unlike traditional attention mechanisms, vLLM uses paged attention to manage memory more efficiently.
This allows it to serve longer context windows without running out of GPU memory.

3. OpenAI-Compatible API
You can drop vLLM into any system that expects OpenAI-style endpoints (/v1/completions, /v1/chat/completions).
Makes it easy to swap out expensive APIs with local inference.

4. Tool Calling Support
vLLM supports tool calling, enabling it to trigger external functions or APIs based on user prompts.
This is crucial for building intelligent agents and RAG pipelines.

5. LoRA Adapter Support
You can fine-tune models using LoRA and load adapters dynamically.
This makes file-based tuning lightweight and modular.

6. Multi-GPU & Multi-Model Support
vLLM can run across multiple GPUs and serve multiple models simultaneously.
Ideal for enterprise-scale deployments.

1) in order to set up vLLM, we need a LINUX environment
2) install Linux on Windows 11 with WSL
LINUX SETUP BASICS-
download ubuntu on microsoft store, once the process is complete open 
<img width="916" height="235" alt="image" src="https://github.com/user-attachments/assets/82970172-908d-4fbd-bfda-1ca3bdd45327" />
<img width="1452" height="409" alt="image" src="https://github.com/user-attachments/assets/8abaf0c9-2822-44fb-a61e-6b2f6c8c14cd" />
now ubuntu is running on your local computer
by typing-
<img width="1706" height="290" alt="image" src="https://github.com/user-attachments/assets/4af80e97-f584-4c0f-8b5e-7d73d8379fa2" />
to access any folders or files we can go to root directory
<img width="1610" height="394" alt="image" src="https://github.com/user-attachments/assets/0918d7e0-0cb6-4f1d-aae1-1145a682e2d4" />
this way we can access the drives - C or D and the folders or files present in them.

3) System Prep
   Make sure your system is ready:
<img width="1668" height="645" alt="image" src="https://github.com/user-attachments/assets/80d8b504-b01a-451e-9ec0-8085c10f3003" />

<img width="1705" height="848" alt="image" src="https://github.com/user-attachments/assets/b97eedf2-3bb4-4cba-9c6f-de7e8d06adcf" />

Install vLLM package on your IDE (Pycharm)
<img width="1243" height="365" alt="image" src="https://github.com/user-attachments/assets/9fdbe1bb-8af5-4530-a318-4c01a516b753" />


Install Build Dependencies

Make sure you have the essentials:

sudo apt update
sudo apt install -y build-essential python3-dev git wget curl
sudo apt install -y libopenblas-dev libomp-dev
Also include CUDA installation for additional GPU usage 
<img width="1523" height="230" alt="image" src="https://github.com/user-attachments/assets/ff3ad173-517c-42a5-9116-a9a5c10886f4" />

~$ nvidia-smi
<img width="1058" height="394" alt="image" src="https://github.com/user-attachments/assets/f76d5c5a-94e5-43f2-a4c3-afa5ed540fbb" />

<img width="573" height="28" alt="image" src="https://github.com/user-attachments/assets/de2bebdb-d63e-45b5-a83c-ee139b7837af" />

$ pip install torch torchvision torchaudio - - index-url https: //download
. pytorch. org/whl/cu121
huggingface hub pip install vllm

## Authenticated with Hugging Face
Create a hugging face account and under your credentials create an access token with read permissions, these are sufficient to install the model we will be using.

huggingface-cli login

Successfully authenticated and downloaded model files

## Downloaded GGUF Model via Python

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="models"
)

✅ Exit code 0 confirmed successful download

## Attempted to run vLLM with GGUF model


python3 -m vllm.entrypoints.openai.api_server \
  --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --device cpu \
  --host 0.0.0.0 \
  --port 8080

This command fails because we are trying to run a GGUF model with vLLM 
vLLM does not support this format, it doesn't support CPU- only interface.

RuntimeError: Failed to infer device type

libcuda.so.1: cannot open shared object file
-------------------------------------------------------------------------
## Use a Supported Model Format
Use a model in PyTorch or safetensors format, like:

mkdir models
cd models

--model mistralai/Mistral-7B-Instruct-v0.2

This model is hosted on Hugging Face and works with vLLM.

## Ensure You Have a CUDA-Capable GPU

After installing nvidia drivers we can check
nvidia-smi

then,

pip install vllm


python3 -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 \
  --port 8080


Now we can test this endpoint-

curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is gravity?",
    "max_tokens": 100
  }'

Now let us try to build a simple chatbot using a RAG pipeline and the LLM we just configured.









