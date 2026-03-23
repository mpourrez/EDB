# EdgeStressBench (EDB)

**EdgeStressBench** is a framework for reproducible evaluation of edge systems under **controlled resource stress**. It provides an end-to-end experimental pipeline that integrates workload orchestration, parameterized stress injection, structured tracing, and post-processing for comparative analysis across heterogeneous edge devices.

This repository accompanies the paper:

> **EdgeStressBench: A Reproducible Benchmarking Framework for Edge Systems Under Resource Stress**

## Overview

Edge computing studies often rely on ad hoc combinations of applications, stress tools, logging scripts, and analysis code, making results difficult to reproduce and compare fairly. EdgeStressBench addresses this gap by providing a unified and extensible framework for running controlled experiments on edge devices under configurable resource-stress conditions.

EdgeStressBench supports:

- heterogeneous edge platforms (e.g., Raspberry Pi 4B, Jetson Nano)
- representative edge workloads
- configurable CPU, memory, and I/O stress injection
- structured application-level and system-level tracing
- repeatable experiment campaigns across devices and conditions
- optional extensibility to replication-aware evaluation

## Key Features

- **Unified orchestration** for running workloads under baseline and stressed conditions
- **Parameterized stress injection** using configurable resource-stress scenarios
- **Structured logging** of latency, run metadata, and resource traces
- **Cross-device experimentation** across heterogeneous edge hardware
- **Post-processing and plotting scripts** for analysis and paper figures
- **Extensible design** for broader dependability studies, including replication-aware execution modes

## Repository Structure

Example structure below; update this section to match your exact repo layout.

```text
EDB/
├── orchestrator/         # experiment driver and coordination logic
├── workloads/            # workload implementations and workload-specific configs
├── stressors/            # stress injection utilities / wrappers
├── tracing/              # resource monitoring and trace collection
├── analysis/             # parsing, aggregation, and plotting scripts
├── configs/              # experiment configuration files
├── results/              # optional output directory for generated results
├── scripts/              # helper scripts for setup and execution
├── docs/                 # artifact notes, paper figures, or documentation
└── README.md
