
# Quantum Swap Modelling for Adaptive Traffic Signalling in Intelligent Transportation
#  Quantum vs Classical Clustering for Traffic Signal Control

This project compares classical and quantum clustering-based methods for adaptive traffic signal control. We simulate a 4-way intersection using `pygame`, analyzing vehicle wait times and throughput, while dynamically adapting green light durations.

<img width="1569" height="878" alt="image" src="https://github.com/user-attachments/assets/0c1aa296-423f-4b82-8b64-e399dd0859f5" />

<img width="1275" height="469" alt="Screenshot 2025-07-31 122722" src="https://github.com/user-attachments/assets/6565948c-0374-49d9-89c9-72a9311676e0" />


---

## 📂 Repository Structure

```
📁 project-root/
├── 📁 images/
│   ├── 📁 down/
│   │   ├── car.png
│   │   ├── bus.png
│   │   ├── ...
│   ├── 📁 left/
│   ├── 📁 right/
│   ├── 📁 up/
│   ├── 📁 signals/
│   │   ├── red.png
│   │   ├── green.png
│   │   ├── yellow.png
│   └── intersection.png
├── normal_clustering_submit.py
├── quantum_clustering_submit.py
├── normal_clustering_results.py
├── quantum_clustering_results.py
└── README.md
```

---

##  Project Overview

We simulate real-time traffic control using:

- **Classical KMeans clustering** (in `normal_clustering_submit.py`)
- **Quantum clustering** using swap test or similar strategies (in `quantum_clustering_submit.py`)

These algorithms adjust traffic light durations based on real-time vehicle densities and positions.

---

##  Code Descriptions

| File Name | Description |
|----------|-------------|
| `normal_clustering_submit.py` | Traffic signal simulation with classical KMeans clustering-based green time adjustment |
| `quantum_clustering_submit.py` | Traffic signal simulation using quantum-inspired clustering (e.g. cosine similarity via swap test) |
| `normal_clustering_results.py` | Plots and analyzes results (e.g. wait time, throughput) from classical clustering |
| `quantum_clustering_results.py` | Plots and analyzes results from quantum clustering strategy |

---

##  Features

- Live vehicle generation
- Dynamic signal adjustment based on real-time clustering
- Vehicle tracking, average wait time logging
- Throughput and wait time analysis
- Basic quantum clustering integration via vector similarities

---

##  How to Run

> Make sure all dependencies are installed and the images are correctly placed.

### 1. Install dependencies

```bash
pip install pygame scikit-learn matplotlib
```

For quantum clustering:

```bash
pip install qiskit
```

### 2. Run Classical Clustering Simulation

```bash
python normal_clustering_submit.py
```

### 3. Run Quantum Clustering Simulation

```bash
python quantum_clustering_submit.py
```

### 4. Visualize Results

```bash
python normal_clustering_results.py
python quantum_clustering_results.py
```

---

##  Sample Output Metrics

- **Throughput**: Number of vehicles that crossed the signal
- **Average Wait Time**: Per vehicle in seconds
- Output appears every 10 seconds in the terminal
- Visualization graphs in results scripts

---

##  Quantum Details

The quantum clustering implementation uses:

- **Cosine similarity** between vehicle positions
- **Swap test (optional in backend)** to compute overlaps
- **KMeans-like clustering** based on quantum metrics

> This is a hybrid approach to test the viability of quantum representations in dynamic systems.

---

##  TODO

-  Replace simulated quantum similarity with actual Qiskit swap test circuit
-  Integrate QML models for signal prediction
-  Add benchmark graphs comparing classical vs quantum performance

---

##  Image Assets

All images used for vehicles and intersection simulation are in the `images/` folder, with the following subfolders:

- `right/`, `left/`, `up/`, `down/` – directional vehicle sprites
- `signals/` – red, green, yellow light icons
- `intersection.png` – main background image

Ensure correct file names and dimensions for smooth simulation.

---

##  License

This project is open-source and intended for educational and research purposes.

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
```
