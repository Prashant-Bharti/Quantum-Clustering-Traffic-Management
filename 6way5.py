# 6way.py
# 6-way (star) traffic sim with quantum swap-test green time allocation,
# generalized from original 4-way quantum_traffic_swap.py.
#
# Folder structure:
#   images/
#     right/
#     left/
#     up/
#     down/
#     upleft/
#     upright/
#     downleft/
#     downright/
#       car.png, bus.png, truck.png, bike.png
#     signals/red.png, yellow.png, green.png
#     intersection.png

import os
import random
import time
import threading
import sys
import math

import pygame
import numpy as np

# Qiskit optional
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

# ----------------------------
# Config
# ----------------------------
USE_QUANTUM = True and HAVE_QISKIT
SHOTS = 512

# SHORTER green range now
BASE_GREEN = 3       # min green seconds
MAX_GREEN  = 12      # max green seconds

FPS = 30

noOfSignals = 6
defaultGreen = {i: BASE_GREEN for i in range(noOfSignals)}
defaultRed = 150
defaultYellow = 2

signals = []
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}

pygame.init()
simulation = pygame.sprite.Group()
vehicles_lock = threading.Lock()

# Metrics
throughput_count = 0
total_wait_time = 0.0
exited_vehicles = 0

# ----------------------------
# 6-way geometry
# ----------------------------
SCREEN_W, SCREEN_H = 1400, 800
CENTER = np.array([SCREEN_W // 2, SCREEN_H // 2], dtype=float)

# Directions indices 0..5 mapped to folders and labels
# Angles: 0°, 60°, 120°, 180°, 240°, 300°
directionFolders = {
    0: 'right',      # 0°
    1: 'upright',    # 60°
    2: 'upleft',     # 120°
    3: 'left',       # 180°
    4: 'downleft',   # 240°
    5: 'downright',  # 300°
}
directionLabels = {k: v.upper() for k, v in directionFolders.items()}

# distances along each ray
R_SPAWN       = 900   # spawn distance from center
R_STOP_LINE   = 120   # actual stop line distance from center
R_QUEUE_STOP  = 160   # queue stop distance from center (before stop line)
R_EXIT        = 1000  # beyond center where we despawn
LANE_WIDTH    = 16
STOPPING_GAP  = 10
MOVING_GAP    = 10

# crossing zone for lateral offset for green direction
R_CROSS       = 200
CROSS_SHIFT   = 2 * LANE_WIDTH

# Vehicles containers: 3 lanes per approach
vehicles = {i: {0: [], 1: [], 2: [], 'crossed': 0} for i in range(noOfSignals)}

geom = {}
signalCoods = []
signalTimerCoods = []


def build_geometry():
    """Precompute ray geometry: queue-stop, stop-line, exit, and signal positions."""
    global geom, signalCoods, signalTimerCoods
    geom = {}
    signalCoods = []
    signalTimerCoods = []

    for k in range(noOfSignals):
        theta = math.radians(60.0 * k)
        out = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        u = out                     # direction of travel
        n = np.array([-u[1], u[0]]) # perpendicular for lanes

        lane_offsets = {0: -LANE_WIDTH, 1: 0.0, 2: +LANE_WIDTH}
        starts = {}
        for lane in range(3):
            starts[lane] = CENTER - u * R_SPAWN + n * lane_offsets[lane]

        # Scalars along ray: s=0 at spawn, s increases toward center (at s=R_SPAWN)
        stop_line_s  = R_SPAWN - R_STOP_LINE
        queue_stop_s = R_SPAWN - R_QUEUE_STOP   # queue is BEFORE stop line (smaller s)
        end_s        = R_SPAWN + R_EXIT

        geom[k] = dict(
            theta=theta,
            u=u,
            n=n,
            lane_offsets=lane_offsets,
            starts=starts,
            stop_line_s=stop_line_s,
            queue_stop_s=queue_stop_s,
            end_s=end_s,
        )

        # signal & timer near stop line
        sig_pos   = CENTER - u * (R_STOP_LINE + 30)
        timer_pos = CENTER - u * (R_STOP_LINE + 55)
        signalCoods.append((int(sig_pos[0]), int(sig_pos[1])))
        signalTimerCoods.append((int(timer_pos[0]), int(timer_pos[1])))


build_geometry()

# ----------------------------
# Assets
# ----------------------------
def load_signal_image(name):
    path = os.path.join('images', 'signals', f'{name}.png')
    try:
        return pygame.image.load(path)
    except Exception:
        surf = pygame.Surface((30, 30))
        if name == 'red':
            surf.fill((255, 0, 0))
        elif name == 'yellow':
            surf.fill((200, 200, 0))
        else:
            surf.fill((0, 200, 0))
        return surf


sig_red    = load_signal_image('red')
sig_yellow = load_signal_image('yellow')
sig_green  = load_signal_image('green')


def load_background():
    path = os.path.join('images', 'intersection.png')
    try:
        return pygame.image.load(path)
    except Exception:
        bg = pygame.Surface((SCREEN_W, SCREEN_H))
        bg.fill((50, 50, 50))
        guide = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for k in range(noOfSignals):
            u = geom[k]['u']
            p1 = (int(CENTER[0] - u[0] * R_SPAWN), int(CENTER[1] - u[1] * R_SPAWN))
            p2 = (int(CENTER[0] + u[0] * R_SPAWN), int(CENTER[1] + u[1] * R_SPAWN))
            pygame.draw.line(guide, (80, 80, 80, 180), p1, p2, 3)
        bg.blit(guide, (0, 0))
        return bg


def load_vehicle_sprite(vehicleClass, dir_index):
    """
    Load sprite from its direction folder, fall back to others, else rectangle.
    """
    preferred = [directionFolders.get(dir_index)]
    preferred.extend(['upleft', 'upright', 'downleft', 'downright', 'up', 'down', 'left', 'right'])
    for folder in preferred:
        if folder is None:
            continue
        p = os.path.join('images', folder, f'{vehicleClass}.png')
        if os.path.exists(p):
            try:
                img = pygame.image.load(p)
                img = pygame.transform.scale(img, (int(img.get_width() * 0.5), int(img.get_height() * 0.5)))
                return img.convert_alpha()
            except Exception:
                pass
    surf = pygame.Surface((30, 15))
    surf.fill((0, 200, 0))
    return surf


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
    """
    Generalized from original Vehicle:
      - All directions spawn and move toward queue_stop_s.
      - Only currentGreen may cross the stop_line_s.
      - Once crossed, vehicles clear the intersection regardless of light.
    """
    def __init__(self, lane, vehicleClass, dir_index):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.dir_index = dir_index
        g = geom[dir_index]
        self.u = g['u']
        self.n = g['n']

        self.image = load_vehicle_sprite(vehicleClass, dir_index)
        rect = self.image.get_rect()
        self.body_len = max(rect.width, rect.height)

        self.s = 0.0
        self.crossed = 0
        self.spawn_time = time.time()
        self.exit_time = None

        self.base_start = g['starts'][lane].copy()
        self.pos = self.base_start + self.u * self.s
        self.x, self.y = float(self.pos[0]), float(self.pos[1])

        with vehicles_lock:
            vehicles[dir_index][lane].append(self)
            self.index = len(vehicles[dir_index][lane]) - 1

            # stop_s relative to previous vehicle or default queue stop
            if self.index > 0 and vehicles[dir_index][lane][self.index - 1].crossed == 0:
                prev = vehicles[dir_index][lane][self.index - 1]
                self.stop_s = prev.stop_s - (prev.body_len + STOPPING_GAP)
            else:
                self.stop_s = g['queue_stop_s']  # queue stop, not the stop line

        simulation.add(self)

    def _world_from_scalar(self, s, lateral_extra=0.0):
        return self.base_start + self.u * s + self.n * lateral_extra

    def move(self):
        global throughput_count, total_wait_time, exited_vehicles
        dir_i = self.dir_index
        g = geom[dir_i]

        front_s = self.s + self.body_len

        # Mark crossed when front passes stop line (NOT queue stop)
        if self.crossed == 0 and front_s > g['stop_line_s']:
            self.crossed = 1

        # Movement rule (like original):
        # - All cars may move while front_s <= stop_s (approach queue stop).
        # - Only currentGreen vehicles may move beyond their queue stop and then cross.
        # - Once crossed, vehicles keep moving to clear the box.
        with vehicles_lock:
            can_move = False
            green_here = (currentGreen == dir_i and currentYellow == 0)

            if self.index == 0:
                # first in lane
                if (front_s <= self.stop_s) or self.crossed or green_here:
                    can_move = True
            else:
                prev = vehicles[dir_i][self.lane][self.index - 1]
                # maintain gap relative to previous vehicle's position
                gap_ok = (self.s + self.body_len) < (prev.s - MOVING_GAP)
                if ((front_s <= self.stop_s) or self.crossed or green_here) and gap_ok:
                    can_move = True

        if can_move:
            self.s += self.speed

        # side offset in intersection for green approach (pass stopped opposite)
        lateral_extra = 0.0
        s_center = R_SPAWN
        if (currentGreen == dir_i and currentYellow == 0) and (abs(self.s - s_center) <= R_CROSS):
            if self.lane == 0:
                lateral_extra = -CROSS_SHIFT
            elif self.lane == 2:
                lateral_extra = +CROSS_SHIFT
            else:
                lateral_extra = CROSS_SHIFT * 0.5

        self.pos = self._world_from_scalar(self.s, lateral_extra=lateral_extra)
        self.x, self.y = float(self.pos[0]), float(self.pos[1])

        # exit
        if not hasattr(self, 'exited') and self.s > g['end_s'] + 50:
            self.exited = True
            self.exit_time = time.time()

        if hasattr(self, 'exited') and self.exited and self.exit_time is not None:
            if not hasattr(self, '_counted') or not self._counted:
                self._counted = True
                with vehicles_lock:
                    throughput_count += 1
                    wait = max(0.0, self.exit_time - self.spawn_time)
                    total_wait_time += wait
                    exited_vehicles += 1


# ----------------------------
# Feature vectors (same logic, N=6)
# ----------------------------
def direction_feature_vectors():
    """Return 6 vectors (one per approach) = [#cars,#buses,#trucks,#bikes]."""
    F = []
    with vehicles_lock:
        for dir_idx in range(noOfSignals):
            counts = [0, 0, 0, 0]
            for lane in range(3):
                for v in vehicles[dir_idx][lane]:
                    for k, name in vehicleTypes.items():
                        if name == v.vehicleClass:
                            counts[k] += 1
                            break
            F.append(np.array(counts, dtype=float))
    return F


# ----------------------------
# Quantum swap-test similarity (exactly like original)
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
    eps = 1e-12
    if np.linalg.norm(vec1) < eps or np.linalg.norm(vec2) < eps:
        return 0.0
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    v1p, n1 = _pad_to_power_of_two(v1)
    v2p, n2 = _pad_to_power_of_two(v2)
    assert n1 == n2
    n = n1

    num_qubits = 1 + 2*n
    qc = QuantumCircuit(num_qubits, 1)
    anc = 0
    reg1 = list(range(1, 1+n))
    reg2 = list(range(1+n, 1+2*n))

    qc.h(anc)
    qc.initialize(v1p.tolist(), reg1)
    qc.initialize(v2p.tolist(), reg2)
    for i in range(n):
        qc.cswap(anc, reg1[i], reg2[i])
    qc.h(anc)
    qc.measure(anc, 0)

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    prob0 = counts.get('0', 0) / max(1, shots)
    fidelity_est = 2*prob0 - 1
    fidelity_est = max(0.0, min(1.0, fidelity_est))
    return fidelity_est


def swap_test_similarity_classical(vec1, vec2):
    eps = 1e-12
    if np.linalg.norm(vec1) < eps or np.linalg.norm(vec2) < eps:
        return 0.0
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    dot = np.dot(v1, v2)
    return float(dot*dot)


def swap_test_similarity(vec1, vec2):
    if USE_QUANTUM and HAVE_QISKIT:
        try:
            return swap_test_similarity_qiskit(vec1, vec2)
        except Exception:
            return swap_test_similarity_classical(vec1, vec2)
    else:
        return swap_test_similarity_classical(vec1, vec2)


def build_similarity_matrix(direction_vectors):
    N = len(direction_vectors)
    S = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i, N):
            if i == j:
                S[i, j] = 1.0
            else:
                f = swap_test_similarity(direction_vectors[i], direction_vectors[j])
                S[i, j] = f
                S[j, i] = f
    return S


def allocate_green_times_from_similarity(S, loads):
    """
    Same structure as original, but with:
      - dynamic alpha depending on total load
      - shorter overall range [BASE_GREEN, MAX_GREEN]
    """
    loads = np.array(loads, dtype=float)
    total_load = loads.sum()
    if total_load <= 0.0:
        return {i: BASE_GREEN for i in range(len(loads))}

    d = 1.0 - np.mean(S, axis=1)
    raw = d * loads

    if raw.sum() <= 0.0:
        norm = loads / total_load
    else:
        norm = raw / raw.sum()

    # adaptive alpha: for low load, dynamic part small, so all greens ~ BASE_GREEN
    alpha_max = MAX_GREEN - BASE_GREEN
    L_ref = 30.0   # vehicles at which we use full alpha
    alpha = alpha_max * min(1.0, total_load / L_ref)

    greens = {}
    for i in range(len(loads)):
        g = BASE_GREEN + alpha * norm[i]
        g_int = int(round(g))
        g_int = max(BASE_GREEN, min(MAX_GREEN, g_int))
        greens[i] = g_int

    return greens


def updateGreenTimesFromQuantum():
    global defaultGreen
    dir_vectors = direction_feature_vectors()
    S = build_similarity_matrix(dir_vectors)
    loads = [vec.sum() for vec in dir_vectors]
    newTimes = allocate_green_times_from_similarity(S, loads)
    defaultGreen = newTimes
    print("Quantum S matrix:\n", np.round(S, 3))
    print("Loads:", loads, "Assigned greens:", defaultGreen)


# ----------------------------
# Helper: check if a direction still has waiting/approaching vehicles
# ----------------------------
def has_waiting_or_approaching(dir_idx):
    """
    Returns True if there exists a vehicle of this direction that
    has NOT exited and has NOT yet crossed the stop line.
    Used to end green early when the direction has no remaining demand.
    """
    g = geom[dir_idx]
    stop_line_s = g['stop_line_s']
    with vehicles_lock:
        for lane in range(3):
            for v in vehicles[dir_idx][lane]:
                if getattr(v, 'exited', False):
                    continue
                # not yet crossed and still before/at stop line → needs service
                if v.crossed == 0 and (v.s + v.body_len) <= stop_line_s + 5:
                    return True
    return False


# ----------------------------
# Controller & signals (same structure as 4-way, with early cut-off)
# ----------------------------
def initialize_signals():
    global signals
    signals = []
    signals.append(TrafficSignal(0, defaultYellow, defaultGreen[0]))
    for i in range(1, noOfSignals):
        signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[i]))


def controller_loop():
    global currentGreen, nextGreen, currentYellow, signals, defaultGreen
    initialize_signals()
    while True:
        # recompute green times based on current congestion pattern
        updateGreenTimesFromQuantum()
        signals[currentGreen].green = defaultGreen[currentGreen]
        signals[nextGreen].red = defaultYellow

        # GREEN phase with early cut-off if lane is empty
        while signals[currentGreen].green > 0:
            if not has_waiting_or_approaching(currentGreen):
                break
            updateValues()
            time.sleep(1)

        # YELLOW phase
        currentYellow = 1
        with vehicles_lock:
            # reset queue stop for the green approach (like defaultStop in original)
            for lane in range(3):
                for v in vehicles[currentGreen][lane]:
                    v.stop_s = geom[currentGreen]['queue_stop_s']
        while signals[currentGreen].yellow > 0:
            updateValues()
            time.sleep(1)

        # reset and rotate
        currentYellow = 0
        signals[currentGreen].green = defaultGreen[currentGreen]
        signals[currentGreen].yellow = defaultYellow
        signals[currentGreen].red = defaultRed

        currentGreen = nextGreen
        nextGreen = (currentGreen + 1) % noOfSignals


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
    """
    Random vehicles from all 6 directions, continuously.
    Red approaches keep generating and moving toward queue, but only green crosses.
    """
    while True:
        vehicle_type = random.randint(0, 3)
        lane_number  = random.randint(0, 2)
        dir_index    = random.randint(0, noOfSignals - 1)
        Vehicle(lane_number, vehicleTypes[vehicle_type], dir_index)
        time.sleep(0.2 + random.random() * 0.6)


# ----------------------------
# Metrics & HUD
# ----------------------------
def getLiveVehicleCounts():
    with vehicles_lock:
        return {directionFolders[i]: sum(len(vehicles[i][lane]) for lane in range(3)) for i in range(noOfSignals)}


def getThroughput():
    global throughput_count
    with vehicles_lock:
        return throughput_count


def getAverageWaitTime():
    with vehicles_lock:
        return (total_wait_time / exited_vehicles) if exited_vehicles > 0 else 0.0


# ----------------------------
# Main loop
# ----------------------------
def main():
    threading.Thread(target=controller_loop, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("SIM - Quantum Swap Traffic (6-Way Star)")

    background = load_background()
    font = pygame.font.Font(None, 30)
    infoFont = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    metric_timer = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(background, (0, 0))

        # signals
        for i in range(noOfSignals):
            sig = signals[i]
            if i == currentGreen and currentYellow:
                img = sig_yellow
            elif i == currentGreen and currentYellow == 0:
                img = sig_green
            else:
                img = sig_red
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")

        # timers
        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255, 255, 255), (0, 0, 0))
            screen.blit(text, signalTimerCoods[i])

        # vehicles
        with vehicles_lock:
            all_vehicles = list(simulation.sprites())
        for v in all_vehicles:
            rect = v.image.get_rect(center=(int(v.x), int(v.y)))
            screen.blit(v.image, rect.topleft)
            v.move()

        # HUD
        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for i in range(noOfSignals):
            lab = directionLabels[i]
            val = vehicleCounts[directionFolders[i]]
            text = infoFont.render(f"{lab} vehicles: {val}", True, (255, 255, 0))
            screen.blit(text, (10, y_offset))
            y_offset += 22

        pygame.display.update()
        clock.tick(FPS)

        if time.time() - metric_timer >= 10.0:
            metric_timer = time.time()
            print(f"[METRICS @ {time.strftime('%H:%M:%S')}] Throughput: {getThroughput()}, Avg Wait Time: {getAverageWaitTime():.2f}s")


if __name__ == "__main__":
    print("Qiskit available:", HAVE_QISKIT, "USE_QUANTUM:", USE_QUANTUM)
    main()
