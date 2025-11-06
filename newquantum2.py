# quantum_traffic_swap.py
# Complete Pygame traffic sim with quantum swap-test green time allocation.

import random, time, threading, sys, math
import pygame
import numpy as np

# Qiskit imports
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False
    # we'll fall back to classical dot-product if qiskit unavailable

# ----------------------------
# Config
# ----------------------------
USE_QUANTUM = True and HAVE_QISKIT  # set False to use deterministic classical inner-product proxy
SHOTS = 512                         # qasm shots for swap test (if using Qiskit)
BASE_GREEN = 5                      # min green seconds
MAX_GREEN = 30                      # max green seconds

# Simulation timing
FPS = 30

# Default signal durations
defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 2

# Signal and simulation metadata
signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Vehicle speeds
speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}

# Spawn positions
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

# Vehicles containers
vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0},
            'down':  {0:[], 1:[], 2:[], 'crossed':0},
            'left':  {0:[], 1:[], 2:[], 'crossed':0},
            'up':    {0:[], 1:[], 2:[], 'crossed':0}}

vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 570, 'down': 310, 'left': 820, 'up': 555}

stoppingGap = 10
movingGap = 10

# Pygame init
pygame.init()
simulation = pygame.sprite.Group()

# Threading lock for safe access
vehicles_lock = threading.Lock()

# Metrics
throughput_count = 0
total_wait_time = 0.0
exited_vehicles = 0

# ----------------------------
# Classes
# ----------------------------
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.spawn_time = time.time()
        self.exit_time = None
        # append to global list
        with vehicles_lock:
            vehicles[direction][lane].append(self)
            self.index = len(vehicles[direction][lane]) - 1

        # load image if present, else use rectangle
        try:
            path = "images/" + direction + "/" + vehicleClass + ".png"
            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (int(img.get_width()*0.5), int(img.get_height()*0.5)))
            self.image = img.convert_alpha()
        except Exception:
            surf = pygame.Surface((30,15))
            surf.fill((0,200,0))
            self.image = surf

        # compute stop position relative to previous vehicle
        with vehicles_lock:
            if self.index > 0 and vehicles[direction][lane][self.index-1].crossed == 0:
                prev = vehicles[direction][lane][self.index-1]
                if direction == 'right':
                    self.stop = prev.stop - prev.image.get_rect().width - stoppingGap
                elif direction == 'left':
                    self.stop = prev.stop + prev.image.get_rect().width + stoppingGap
                elif direction == 'down':
                    self.stop = prev.stop - prev.image.get_rect().height - stoppingGap
                elif direction == 'up':
                    self.stop = prev.stop + prev.image.get_rect().height + stoppingGap
            else:
                self.stop = defaultStop[direction]
        # update spawn offsets
        if direction == 'right':
            x[direction][lane] -= self.image.get_rect().width + stoppingGap
        elif direction == 'left':
            x[direction][lane] += self.image.get_rect().width + stoppingGap
        elif direction == 'down':
            y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == 'up':
            y[direction][lane] += self.image.get_rect().height + stoppingGap
        simulation.add(self)

    def move(self):
        """Move the vehicle according to rules and signal state; update exit metrics."""
        global throughput_count, total_wait_time, exited_vehicles
        d = self.direction
        w, h = self.image.get_rect().width, self.image.get_rect().height

        if d == 'right':
            if self.crossed == 0 and self.x + w > stopLines[d]:
                self.crossed = 1
            if (self.x + w <= self.stop or self.crossed or (currentGreen == 0 and currentYellow == 0)) and \
               (self.index == 0 or self.x + w < vehicles[d][self.lane][self.index-1].x - movingGap):
                self.x += self.speed

            if not hasattr(self, 'exited') and self.x > 1400:
                self.exited = True
                self.exit_time = time.time()
        elif d == 'down':
            if self.crossed == 0 and self.y + h > stopLines[d]:
                self.crossed = 1
            if (self.y + h <= self.stop or self.crossed or (currentGreen == 1 and currentYellow == 0)) and \
               (self.index == 0 or self.y + h < vehicles[d][self.lane][self.index-1].y - movingGap):
                self.y += self.speed

            if not hasattr(self, 'exited') and self.y > 900:
                self.exited = True
                self.exit_time = time.time()
        elif d == 'left':
            if self.crossed == 0 and self.x < stopLines[d]:
                self.crossed = 1
            if (self.x >= self.stop or self.crossed or (currentGreen == 2 and currentYellow == 0)) and \
               (self.index == 0 or self.x > vehicles[d][self.lane][self.index-1].x + vehicles[d][self.lane][self.index-1].image.get_rect().width + movingGap):
                self.x -= self.speed

            if not hasattr(self, 'exited') and self.x < -200:
                self.exited = True
                self.exit_time = time.time()
        elif d == 'up':
            if self.crossed == 0 and self.y < stopLines[d]:
                self.crossed = 1
            if (self.y >= self.stop or self.crossed or (currentGreen == 3 and currentYellow == 0)) and \
               (self.index == 0 or self.y > vehicles[d][self.lane][self.index-1].y + vehicles[d][self.lane][self.index-1].image.get_rect().height + movingGap):
                self.y -= self.speed

            if not hasattr(self, 'exited') and self.y < -200:
                self.exited = True
                self.exit_time = time.time()

        # update metrics when vehicle exits
        if hasattr(self, 'exited') and self.exited and self.exit_time is not None:
            if not hasattr(self, '_counted') or not self._counted:
                self._counted = True
                with vehicles_lock:
                    throughput_count += 1
                    wait = max(0.0, self.exit_time - self.spawn_time)
                    total_wait_time += wait
                    exited_vehicles += 1

# ----------------------------
# Feature vectors: per-direction counts (cars,buses,trucks,bikes)
# ----------------------------
def direction_feature_vectors():
    """Return 4 vectors (for right,down,left,up), each = [#cars,#buses,#trucks,#bikes] across lanes."""
    F = []
    with vehicles_lock:
        for dir_name in ['right','down','left','up']:
            counts = [0,0,0,0]
            for lane in range(3):
                for v in vehicles[dir_name][lane]:
                    # map vehicleClass to type index
                    t = None
                    for k, name in vehicleTypes.items():
                        if name == v.vehicleClass:
                            t = k; break
                    if t is None: continue
                    counts[t] += 1
            F.append(np.array(counts, dtype=float))
    return F  # list of 4 arrays shape (4,)

# ----------------------------
# Quantum swap test similarity
# ----------------------------
def _pad_to_power_of_two(vec):
    L = len(vec)
    n = int(np.ceil(np.log2(L)))
    size = 2**n
    if size == L:
        return vec, n
    padded = np.zeros(size, dtype=float)
    padded[:L] = vec
    return padded, n

def swap_test_similarity_qiskit(vec1, vec2, shots=SHOTS):
    """Return squared overlap |<psi1|psi2>|^2 estimated by swap test using AerSimulator (qasm)."""
    # Normalize
    eps = 1e-12
    if np.linalg.norm(vec1) < eps or np.linalg.norm(vec2) < eps:
        return 0.0
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    # pad to power of two
    v1p, n = _pad_to_power_of_two(v1)
    v2p, n2 = _pad_to_power_of_two(v2)
    assert n == n2
    # build circuit: ancilla + n qubits for v1 + n qubits for v2
    num_qubits = 1 + 2*n
    qc = QuantumCircuit(num_qubits, 1)
    anc = 0
    reg1 = list(range(1, 1+n))
    reg2 = list(range(1+n, 1+2*n))
    # Hadamard ancilla
    qc.h(anc)
    # initialize states (Initialize accepts list of amplitudes)
    qc.initialize(v1p.tolist(), reg1)
    qc.initialize(v2p.tolist(), reg2)
    # controlled-swaps for all qubits
    for i in range(n):
        qc.cswap(anc, reg1[i], reg2[i])
    qc.h(anc)
    qc.measure(anc, 0)
    # run on AerSimulator qasm
    backend = AerSimulator()
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    prob0 = counts.get('0', 0) / max(1, shots)
    fidelity_est = 2 * prob0 - 1
    # numerical safety: fidelity in [0,1]
    fidelity_est = max(0.0, min(1.0, fidelity_est))
    return fidelity_est

def swap_test_similarity_classical(vec1, vec2):
    """Exact classical squared overlap for normalized vectors: (dot)^2"""
    eps = 1e-12
    if np.linalg.norm(vec1) < eps or np.linalg.norm(vec2) < eps:
        return 0.0
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    dot = np.dot(v1, v2)
    return float(dot*dot)

def swap_test_similarity(vec1, vec2):
    """Wrapper: use quantum simulator if available and requested, else classical exact."""
    if USE_QUANTUM and HAVE_QISKIT:
        try:
            return swap_test_similarity_qiskit(vec1, vec2)
        except Exception as e:
            # fallback
            return swap_test_similarity_classical(vec1, vec2)
    else:
        return swap_test_similarity_classical(vec1, vec2)

# ----------------------------
# Build similarity matrix S between directions
# ----------------------------
def build_similarity_matrix(direction_vectors):
    """direction_vectors: list of 4 arrays (len 4). Returns 4x4 symmetric matrix S of fidelities."""
    S = np.zeros((4,4), dtype=float)
    for i in range(4):
        for j in range(i,4):
            if i == j:
                S[i,j] = 1.0
            else:
                f = swap_test_similarity(direction_vectors[i], direction_vectors[j])
                S[i,j] = f
                S[j,i] = f
    return S

# ----------------------------
# Convert S + loads -> green times (principled)
# ----------------------------
def allocate_green_times_from_similarity(S, loads):
    """
    S: 4x4 similarity matrix (higher => more similar)
    loads: array of 4 non-negative numbers (counts per direction)
    Algorithm:
      - compute dissimilarity per direction: d_i = 1 - mean_j S[i,j]
      - raw_score_i = d_i * loads[i]
      - if sum(raw_score)==0 -> fallback to proportional by loads
      - normalized = raw_score / sum(raw_score)
      - G_i = BASE_GREEN + round( (MAX_GREEN - BASE_GREEN) * normalized_i )
    This gives principled allocation: high load and dissimilar → larger share.
    """
    loads = np.array(loads, dtype=float)
    d = 1.0 - np.mean(S, axis=1)  # dissimilarity in [0,1]
    raw = d * loads
    if raw.sum() <= 0.0:
        # fallback: proportional to loads (or equal if no loads)
        if loads.sum() <= 0.0:
            return {i: BASE_GREEN for i in range(4)}
        norm = loads / (loads.sum())
        greens = {}
        for i in range(4):
            greens[i] = int(BASE_GREEN + round((MAX_GREEN - BASE_GREEN) * norm[i]))
        return greens
    norm = raw / raw.sum()
    greens = {}
    for i in range(4):
        greens[i] = int(BASE_GREEN + round((MAX_GREEN - BASE_GREEN) * norm[i]))
        # safety clamp
        greens[i] = max(BASE_GREEN, min(MAX_GREEN, greens[i]))
    return greens

# ----------------------------
# Quantum-based green time update (controller will call this periodically)
# ----------------------------
def updateGreenTimesFromQuantum():
    global defaultGreen
    # 1) compute direction vectors (counts of types)
    dir_vectors = direction_feature_vectors()
    # 2) Optionally normalize each vector? Keep raw counts to reflect load; similarity handles normalization
    # 3) Build similarity matrix
    S = build_similarity_matrix(dir_vectors)
    # 4) compute loads (total vehicles per direction)
    loads = [vec.sum() for vec in dir_vectors]
    # 5) compute green allocation
    newTimes = allocate_green_times_from_similarity(S, loads)
    defaultGreen = newTimes
    # debug print optionally
    print("Quantum S matrix:\n", np.round(S,3))
    print("Loads:", loads, "Assigned greens:", defaultGreen)

# ----------------------------
# Signal timing and controller loop (no recursion)
# ----------------------------
def initialize_signals():
    global signals
    signals = []
    signals.append(TrafficSignal(0, defaultYellow, defaultGreen[0]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[1]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[2]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[3]))

def controller_loop(poll_interval=5):
    """Controller runs in its own thread and updates green times periodically, then steps through phases."""
    global currentGreen, nextGreen, currentYellow, signals, defaultGreen
    initialize_signals()
    while True:
        # Recompute green times using quantum similarity
        updateGreenTimesFromQuantum()
        # Set current green time and proceed with phase
        signals[currentGreen].green = defaultGreen[currentGreen]
        signals[nextGreen].red = defaultYellow

        # GREEN phase (decrement once per second)
        while signals[currentGreen].green > 0:
            updateValues()
            time.sleep(1)

        # YELLOW phase
        currentYellow = 1
        for i in range(3):
            for v in vehicles[directionNumbers[currentGreen]][i]:
                v.stop = defaultStop[directionNumbers[currentGreen]]
        while signals[currentGreen].yellow > 0:
            updateValues()
            time.sleep(1)

        # RESET and rotate
        currentYellow = 0
        signals[currentGreen].green = defaultGreen[currentGreen]
        signals[currentGreen].yellow = defaultYellow
        signals[currentGreen].red = defaultRed

        currentGreen = nextGreen
        nextGreen = (currentGreen + 1) % noOfSignals
        # loop continues

def updateValues():
    global signals, currentGreen, currentYellow
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

# ----------------------------
# Vehicle generator
# ----------------------------
def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(0,2)  # include lane 0..2
        temp = random.randint(0,99)
        dist = [40,70,90,100]
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        else:
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(0.5)

# ----------------------------
# Utilities & metrics
# ----------------------------
def getLiveVehicleCounts():
    with vehicles_lock:
        return {direction: sum(len(vehicles[direction][lane]) for lane in range(3)) for direction in directionNumbers.values()}

def getThroughput():
    global throughput_count
    with vehicles_lock:
        return throughput_count

def getAverageWaitTime():
    with vehicles_lock:
        return (total_wait_time / exited_vehicles) if exited_vehicles > 0 else 0.0

# ----------------------------
# Main Pygame loop (must be main thread)
# ----------------------------
def main():
    # start controller and generator threads
    threading.Thread(target=controller_loop, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()

    screen = pygame.display.set_mode((1400,800))
    pygame.display.set_caption("SIM - Quantum Swap Traffic")
    try:
        background = pygame.image.load('images/intersection.png')
    except Exception:
        background = pygame.Surface((1400,800)); background.fill((50,50,50))
    try:
        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
    except Exception:
        redSignal = pygame.Surface((30,30)); redSignal.fill((255,0,0))
        yellowSignal = pygame.Surface((30,30)); yellowSignal.fill((200,200,0))
        greenSignal = pygame.Surface((30,30)); greenSignal.fill((0,200,0))

    font = pygame.font.Font(None, 30)
    infoFont = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    metric_timer = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        screen.blit(background, (0,0))

        # draw signals
        for i in range(noOfSignals):
            sig = signals[i]
            img = yellowSignal if i == currentGreen and currentYellow else greenSignal if i == currentGreen else redSignal
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")

        # signal timers
        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])

        # draw vehicles snapshot
        with vehicles_lock:
            all_vehicles = list(simulation.sprites())
        for v in all_vehicles:
            screen.blit(v.image, (v.x, v.y))
            v.move()

        # live counts
        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for dir_idx, direction in directionNumbers.items():
            text = infoFont.render(f"{direction.upper()} vehicles: {vehicleCounts[direction]}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.update()
        clock.tick(FPS)

        # metrics every 10 seconds
        if time.time() - metric_timer >= 10.0:
            metric_timer = time.time()
            print(f"[METRICS @ {time.strftime('%H:%M:%S')}] Throughput: {getThroughput()}, Avg Wait Time: {getAverageWaitTime():.2f}s")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("Qiskit available:", HAVE_QISKIT, "USE_QUANTUM:", USE_QUANTUM)
    main()
