# Quantum Swap Modelling for Adaptive Traffic Signalling in Intelligent
Transportation
# Quantum vs Classical Clustering for Traffic Signal Control

This project compares classical and quantum clustering-based methods for adaptive traffic signal control. We simulate a 4-way intersection using `pygame`, analyzing vehicle wait times and throughput, while dynamically adapting green light durations.

---

## 📂 Repository Structure
📁 project-root/
├── 📁 images/
│ ├── 📁 down/
│ │ ├── car.png
│ │ ├── bus.png
│ │ ├── ...
│ ├── 📁 left/
│ ├── 📁 right/
│ ├── 📁 up/
│ ├── 📁 signals/
│ │ ├── red.png
│ │ ├── green.png
│ │ ├── yellow.png
│ └── intersection.png
├── normal_clustering_submit.py
├── quantum_clustering_submit.py
├── normal_clustering_results.py
├── quantum_clustering_results.py
└── README.md



---

## 🚦 Project Overview

We simulate real-time traffic control using:

- **Classical KMeans clustering** (in `normal_clustering_submit.py`)
- **Quantum clustering** using swap test or similar strategies (in `quantum_clustering_submit.py`)

These algorithms adjust traffic light durations based on real-time vehicle densities and positions.

---

## 📜 Code Descriptions

| File Name | Description |
|----------|-------------|
| `normal_clustering_submit.py` | Traffic signal simulation with classical KMeans clustering-based green time adjustment |
| `quantum_clustering_submit.py` | Traffic signal simulation using quantum-inspired clustering (e.g. cosine similarity via swap test) |
| `normal_clustering_results.py` | Plots and analyzes results (e.g. wait time, throughput) from classical clustering |
| `quantum_clustering_results.py` | Plots and analyzes results from quantum clustering strategy |

---

## 🧪 Features

- Live vehicle generation
- Dynamic signal adjustment based on real-time clustering
- Vehicle tracking, average wait time logging
- Throughput and wait time analysis
- Basic quantum clustering integration via vector similarities

---

## 🧑‍💻 How to Run

> Make sure all dependencies are installed and the images are correctly placed.

### 1. Install dependencies

```bash
pip install pygame scikit-learn matplotlib


### 2. Run Classical Clustering Simulation
```bash
python normal_clustering_submit.py
### 3. Run Quantum Clustering Simulation
```bash

python quantum_clustering_submit.py
### 4. Visualize Results
```bash

python normal_clustering_results.py
python quantum_clustering_results.py

## Sample Output Metrics
Throughput: Number of vehicles that crossed the signal

Average Wait Time: Per vehicle in seconds

Output appears every 10 seconds in the terminal

Visualization graphs in results scripts
