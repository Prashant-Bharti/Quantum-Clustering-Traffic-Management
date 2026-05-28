# 6way_kmeans.py
# 6-way (star) traffic sim with K-Means-based green time allocation.
# Same geometry, movement, and timing logic as 6-way quantum version;
# only the controller (green-time computation) is replaced with a K-Means baseline.

import os
import random
import time
import threading
import sys
import math

import pygame
import numpy as np

# K-Means (optional)
try:
    from sklearn.cluster import KMeans
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ----------------------------
# Config
# ----------------------------
SHOTS = 512  # unused here; kept for parallelism with quantum version if you compare code

# SHORTER green range
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
    Same movement logic as 6-way quantum version:
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
# Feature vectors & loads
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
# K-Means-based green allocation
# ----------------------------
def allocate_green_times_from_kmeans(features, loads, k=2):
    """
    K-Means baseline controller.
    features: list of 6 vectors (dimension 4)
    loads: list of 6 non-negative scalars
    Steps:
      - cluster non-empty directions using KMeans
      - compute cluster total loads
      - heavier clusters get a multiplicative boost
      - then same dynamic alpha logic as quantum version
    """
    loads = np.array(loads, dtype=float)
    total_load = loads.sum()
    N = len(loads)

    if total_load <= 0.0:
        return {i: BASE_GREEN for i in range(N)}

    X = np.vstack(features)  # shape (6,4)
    # indices of dirs that actually have vehicles
    nonempty = np.where(loads > 0)[0]

    # default raw = load (pure load-based if sklearn not available or trivial case)
    raw = loads.copy()

    if HAVE_SKLEARN and len(nonempty) >= 2:
        k_eff = min(k, len(nonempty))
        kmeans = KMeans(n_clusters=k_eff, n_init=10, random_state=0)
        kmeans.fit(X[nonempty])

        labels = -np.ones(N, dtype=int)
        labels[nonempty] = kmeans.labels_

        # cluster total loads
        cluster_loads = np.zeros(k_eff, dtype=float)
        for idx in nonempty:
            c = labels[idx]
            cluster_loads[c] += loads[idx]

        if cluster_loads.sum() > 0:
            # normalize cluster weights to [0,1]
            cluster_weights = cluster_loads / cluster_loads.max()
            # Dir-level boost based on cluster weight
            for i in range(N):
                if loads[i] <= 0:
                    raw[i] = 0.0
                else:
                    c = labels[i]
                    w_c = cluster_weights[c] if c >= 0 else 0.0
                    # base 0.5 + boosted by up to 1.0
                    raw[i] = loads[i] * (0.5 + 0.5 * w_c)
    else:
        # fallback: just use loads as raw
        raw = loads.copy()

    if raw.sum() <= 0.0:
        return {i: BASE_GREEN for i in range(N)}

    norm = raw / raw.sum()

    # adaptive alpha: for low load, dynamic part small, so all greens ~ BASE_GREEN
    alpha_max = MAX_GREEN - BASE_GREEN
    L_ref = 30.0   # vehicles at which we use full alpha
    alpha = alpha_max * min(1.0, total_load / L_ref)

    greens = {}
    for i in range(N):
        g = BASE_GREEN + alpha * norm[i]
        g_int = int(round(g))
        g_int = max(BASE_GREEN, min(MAX_GREEN, g_int))
        greens[i] = g_int

    return greens


def updateGreenTimesFromKMeans():
    global defaultGreen
    feats = direction_feature_vectors()
    loads = [vec.sum() for vec in feats]
    newTimes = allocate_green_times_from_kmeans(feats, loads, k=2)
    defaultGreen = newTimes
    print("[KMEANS] Loads:", loads, "Assigned greens:", defaultGreen)


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
# Controller & signals (same structure as 6-way quantum, just calls KMeans)
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
        # recompute green times based on current congestion pattern via K-Means
        updateGreenTimesFromKMeans()
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
    pygame.display.set_caption("SIM - KMeans Traffic (6-Way Star)")

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
    print("Sklearn available:", HAVE_SKLEARN)
    main()
