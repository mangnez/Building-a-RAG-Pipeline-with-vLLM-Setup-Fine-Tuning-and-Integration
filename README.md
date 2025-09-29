this repository acts as  detailed guide from setting up vLLM from scratch along with installing linux environment- Ubuntu to fine-tuning the LLM for... and finally integrating it into a RAG pipeline for developing an intelligent asisstant.

#1 question - why use vLLM? in what ways ways is it better than other LLMS?
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

3) 
