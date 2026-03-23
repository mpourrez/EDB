# EdgeStressBench: Benchmarking Dependability of Edge Applications and Devices

## What is DeepEdgeBench?

[EdgeStressBench](https://github.com/mpourrez/EDB) is a novel benchmark tool to evaluate the latency and resource utilization of edge applications under various resource stress on edge devices. 

EdgeStressBench contains 14 different edge applications performing micro- and application-level benchmarking on edge devices. Micro-benchmark workloads are to measure the performance of a specific resource type on the devices, e.g., CPU, memory, network bandwidth, and disk I/O. Application-level benchmark workloads are developed based on real-world edge computing use-cases. In particular, various machine learning and AI applications, e.g., [image classification](https://en.wikipedia.org/wiki/Computer_vision#Recognition), and [object detection](https://en.wikipedia.org/wiki/Object_detection), are developed for DeepEdgeBench.

## Getting Started
To get started, install the proper dependencies via Anaconda.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate benchmark

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate benchmark-gpu
```
