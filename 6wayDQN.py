# 6way_dqn.py
# 6-way (star) traffic sim with a DQN agent selecting next approach and green duration.
# Keeps the same vehicle / geometry / visualization logic as your other scripts
# so the DQN-vs-Quantum comparison is consistent.

import os
import random
import time
import threading
import sys
import math
from collections import deque, namedtuple
import numpy as np

# PyGame
import pygame
import pygame
pygame.init()
pygame.font.init()

# PyTorch (DQN)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    raise RuntimeError("PyTorch is required for the DQN script. Install torch and try again.")

# ----------------------------
# Config & tunables
# ----------------------------
SEED = 0
random.seed(SEED); np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)

BASE_GREEN = 3
MAX_GREEN = 12   # discrete durations: 3..12 inclusive
DURATION_BUCKETS = list(range(BASE_GREEN, MAX_GREEN + 1))
NUM_DURATIONS = len(DURATION_BUCKETS)

FPS = 30

noOfSignals = 6

# DQN hyperparameters
STATE_DIM = 24 + 6 + 6   # counts (6*4) + avg waits (6) + last greens (6)
NUM_ACTIONS = noOfSignals * NUM_DURATIONS
HIDDEN = 256
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 20000
MIN_REPLAY = 500
TARGET_UPDATE_FREQ = 500  # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000  # steps (linear)

USE_TRAINING = True   # set False to only act with greedy policy (no learning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Movement / sim params (same as your 6way files)
speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}

# geometry constants (same layout)
SCREEN_W, SCREEN_H = 1400, 800
CENTER = np.array([SCREEN_W // 2, SCREEN_H // 2], dtype=float)

directionFolders = {
    0: 'right',
    1: 'upright',
    2: 'upleft',
    3: 'left',
    4: 'downleft',
    5: 'downright',
}
directionLabels = {k: v.upper() for k, v in directionFolders.items()}

R_SPAWN = 900
R_STOP_LINE = 120
R_QUEUE_STOP = 160
R_EXIT = 1000
LANE_WIDTH = 16
STOPPING_GAP = 10
MOVING_GAP = 10
R_CROSS = 200
CROSS_SHIFT = 2 * LANE_WIDTH

# environment
vehicles = {i: {0: [], 1: [], 2: [], 'crossed': 0} for i in range(noOfSignals)}
geom = {}
signalCoods = []
signalTimerCoods = []

# threads & locks
simulation = pygame.sprite.Group()
vehicles_lock = threading.Lock()

# metrics
throughput_count = 0
total_wait_time = 0.0
exited_vehicles = 0

# controller meta
signals = []
currentGreen = None  # controlled by DQN
currentYellow = 0

# ----------------------------
# geometry builder (same as before)
# ----------------------------
def build_geometry():
    global geom, signalCoods, signalTimerCoods
    geom = {}
    signalCoods = []
    signalTimerCoods = []
    for k in range(noOfSignals):
        theta = math.radians(60.0 * k)
        out = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        u = out
        n = np.array([-u[1], u[0]], dtype=float)
        lane_offsets = {0: -LANE_WIDTH, 1: 0.0, 2: +LANE_WIDTH}
        starts = {}
        for lane in range(3):
            starts[lane] = CENTER - u * R_SPAWN + n * lane_offsets[lane]
        stop_line_s = R_SPAWN - R_STOP_LINE
        queue_stop_s = R_SPAWN - R_QUEUE_STOP
        end_s = R_SPAWN + R_EXIT
        geom[k] = dict(theta=theta, u=u, n=n, lane_offsets=lane_offsets,
                       starts=starts, stop_line_s=stop_line_s,
                       queue_stop_s=queue_stop_s, end_s=end_s)
        sig_pos = CENTER - u * (R_STOP_LINE + 30)
        timer_pos = CENTER - u * (R_STOP_LINE + 55)
        signalCoods.append((int(sig_pos[0]), int(sig_pos[1])))
        signalTimerCoods.append((int(timer_pos[0]), int(timer_pos[1])))

build_geometry()

# ----------------------------
# assets loader (same style)
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

sig_red = load_signal_image('red')
sig_yellow = load_signal_image('yellow')
sig_green = load_signal_image('green')

def load_background():
    path = os.path.join('images', 'intersection.png')
    try:
        return pygame.image.load(path)
    except Exception:
        bg = pygame.Surface((SCREEN_W, SCREEN_H)); bg.fill((50,50,50))
        guide = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for k in range(noOfSignals):
            u = geom[k]['u']
            p1 = (int(CENTER[0] - u[0]*R_SPAWN), int(CENTER[1] - u[1]*R_SPAWN))
            p2 = (int(CENTER[0] + u[0]*R_SPAWN), int(CENTER[1] + u[1]*R_SPAWN))
            pygame.draw.line(guide, (80,80,80,180), p1, p2, 3)
        bg.blit(guide, (0,0))
        return bg

def load_vehicle_sprite(vehicleClass, dir_index):
    preferred = [directionFolders.get(dir_index)]
    preferred.extend(['upleft','upright','downleft','downright','up','down','left','right'])
    for folder in preferred:
        if folder is None:
            continue
        p = os.path.join('images', folder, f'{vehicleClass}.png')
        if os.path.exists(p):
            try:
                img = pygame.image.load(p)
                img = pygame.transform.scale(img, (int(img.get_width()*0.5), int(img.get_height()*0.5)))
                return img.convert_alpha()
            except Exception:
                pass
    surf = pygame.Surface((30,15)); surf.fill((0,200,0))
    return surf

# ----------------------------
# Sprite & vehicle logic (same movement rules)
# ----------------------------
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, dir_index):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.dir_index = dir_index
        g = geom[dir_index]
        self.u = g['u']; self.n = g['n']
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
            if self.index > 0 and vehicles[dir_index][lane][self.index - 1].crossed == 0:
                prev = vehicles[dir_index][lane][self.index - 1]
                self.stop_s = prev.stop_s - (prev.body_len + STOPPING_GAP)
            else:
                self.stop_s = g['queue_stop_s']
        simulation.add(self)

    def _world_from_scalar(self, s, lateral_extra=0.0):
        return self.base_start + self.u * s + self.n * lateral_extra

    def move(self):
        global throughput_count, total_wait_time, exited_vehicles
        dir_i = self.dir_index; g = geom[dir_i]
        front_s = self.s + self.body_len
        if self.crossed == 0 and front_s > g['stop_line_s']:
            self.crossed = 1
        with vehicles_lock:
            can_move = False
            green_here = (currentGreen == dir_i and currentYellow == 0)
            if self.index == 0:
                if (front_s <= self.stop_s) or self.crossed or green_here:
                    can_move = True
            else:
                prev = vehicles[dir_i][self.lane][self.index - 1]
                gap_ok = (self.s + self.body_len) < (prev.s - MOVING_GAP)
                if ((front_s <= self.stop_s) or self.crossed or green_here) and gap_ok:
                    can_move = True
        if can_move:
            self.s += self.speed
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
        if not hasattr(self, 'exited') and self.s > g['end_s'] + 50:
            self.exited = True; self.exit_time = time.time()
        if hasattr(self, 'exited') and self.exited and self.exit_time is not None:
            if not hasattr(self, '_counted') or not self._counted:
                self._counted = True
                with vehicles_lock:
                    throughput_count += 1
                    wait = max(0.0, self.exit_time - self.spawn_time)
                    total_wait_time += wait
                    exited_vehicles += 1

# ----------------------------
# Observations & helpers
# ----------------------------
def direction_feature_vectors():
    F = []
    with vehicles_lock:
        for dir_idx in range(noOfSignals):
            counts = [0,0,0,0]
            for lane in range(3):
                for v in vehicles[dir_idx][lane]:
                    for k,name in vehicleTypes.items():
                        if name == v.vehicleClass:
                            counts[k] += 1; break
            F.append(np.array(counts, dtype=float))
    return F

def average_wait_per_direction():
    waits=[]; now=time.time()
    with vehicles_lock:
        for dir_idx in range(noOfSignals):
            acc=0.0; cnt=0
            for lane in range(3):
                for v in vehicles[dir_idx][lane]:
                    if getattr(v,'exited',False): continue
                    if v.crossed==0:
                        acc += (now - v.spawn_time); cnt += 1
            waits.append((acc/cnt) if cnt>0 else 0.0)
    return waits

def getLiveVehicleCounts():
    with vehicles_lock:
        return {directionFolders[i]: sum(len(vehicles[i][lane]) for lane in range(3)) for i in range(noOfSignals)}

def getThroughput():
    global throughput_count
    with vehicles_lock:
        return throughput_count

def getAverageWaitTime():
    with vehicles_lock:
        return (total_wait_time / exited_vehicles) if exited_vehicles>0 else 0.0

# ----------------------------
# DQN implementation (PyTorch)
# ----------------------------
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.device = DEVICE
        self.policy_net = QNet(state_dim, n_actions).to(self.device)
        self.target_net = QNet(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.steps = 0
        self.eps = EPS_START

    def select_action(self, state, training=True):
        # state: numpy array
        self.steps += 1
        if training:
            self.eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (self.steps / EPS_DECAY))
        if training and random.random() < self.eps:
            return random.randrange(NUM_ACTIONS)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            action = int(q.argmax(dim=1).item())
            return action

    def push_transition(self, *args):
        self.replay.push(*args)

    def learn_step(self):
        if len(self.replay) < MIN_REPLAY:
            return
        batch = self.replay.sample(BATCH_SIZE)
        state = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + GAMMA * q_next * (1.0 - done)
        loss = nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        # periodic target update
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# ----------------------------
# Action helpers
# ----------------------------
def action_to_pair(action):
    # action in [0, NUM_ACTIONS-1] -> (dir_idx, duration)
    dir_idx = action // NUM_DURATIONS
    dur_idx = action % NUM_DURATIONS
    duration = DURATION_BUCKETS[dur_idx]
    return dir_idx, duration

def pair_to_action(dir_idx, duration):
    dur_idx = DURATION_BUCKETS.index(duration)
    return dir_idx * NUM_DURATIONS + dur_idx

# ----------------------------
# Environment-controller that uses the DQN agent
# ----------------------------
agent = DQNAgent(STATE_DIM, NUM_ACTIONS)

# last assigned greens (for state)
last_assigned_greens = [BASE_GREEN] * noOfSignals

def build_state():
    # counts (6x4), waits (6), last greens (6) => 36 dim
    feats = direction_feature_vectors()  # list of arrays (4,)
    counts = np.hstack([f for f in feats]).astype(np.float32)  # 24
    waits = np.array(average_wait_per_direction(), dtype=np.float32)  # 6
    last = np.array(last_assigned_greens, dtype=np.float32)  # 6
    state = np.concatenate([counts, waits, last])
    # normalize scales roughly: counts may be up to dozens, waits seconds up to hundreds; simple scaling:
    # divide counts by 20, waits by 20, greens by MAX_GREEN to keep values within reasonable range
    state[:24] = state[:24] / 20.0
    state[24:30] = state[24:30] / 20.0
    state[30:] = state[30:] / float(MAX_GREEN)
    return state

# reward weights
W_WAIT = 1.0
W_TP = 1.0  # throughput delta weight

# control flags
training_steps = 0

def has_waiting_or_approaching(dir_idx):
    g = geom[dir_idx]
    stop_line_s = g['stop_line_s']
    with vehicles_lock:
        for lane in range(3):
            for v in vehicles[dir_idx][lane]:
                if getattr(v,'exited',False): continue
                if v.crossed==0 and (v.s + v.body_len) <= stop_line_s + 5:
                    return True
    return False

def controller_loop():
    global currentGreen, currentYellow, last_assigned_greens, training_steps
    # initial wait a moment for vehicles to populate
    time.sleep(1.0)
    while True:
        # build current state
        state = build_state()
        # choose action
        action = agent.select_action(state, training=USE_TRAINING)
        dir_idx, duration = action_to_pair(action)
        # set currentGreen and duration
        currentGreen = dir_idx
        currentYellow = 0
        last_assigned_greens[dir_idx] = duration

        # record throughput before phase
        tp_before = getThroughput()

        # run green countdown in seconds, with early cutoff
        seconds_remaining = duration
        while seconds_remaining > 0:
            # early exit if no waiting vehicles
            if not has_waiting_or_approaching(currentGreen):
                break
            updateValues_for_dqn(currentGreen, seconds_remaining)
            time.sleep(1.0)
            seconds_remaining -= 1

        # yellow phase
        currentYellow = 1
        # reset queue stop for the green approach
        with vehicles_lock:
            for lane in range(3):
                for v in vehicles[currentGreen][lane]:
                    v.stop_s = geom[currentGreen]['queue_stop_s']
        yellow_time = 2
        while yellow_time > 0:
            updateValues_for_dqn(currentGreen, yellow_time)
            time.sleep(1.0)
            yellow_time -= 1

        # compute reward & next_state
        tp_after = getThroughput()
        tp_delta = tp_after - tp_before
        avg_wait = getAverageWaitTime()
        # reward: throughput increase rewarded, wait penalized
        reward = - W_WAIT * avg_wait + W_TP * float(tp_delta)

        next_state = build_state()
        done = False  # episode termination not defined here (continuous sim) -> False

        # push transition & train
        if USE_TRAINING:
            agent.push_transition(state, action, reward, next_state, done)
            agent.learn_step()
            training_steps += 1

        # loop continues selecting next action

def updateValues_for_dqn(green_idx, seconds_left):
    # just decrement nothing else; we keep per-signal timers to show in UI if needed
    # We'll store "signals" array used by display; minimal mimic of earlier interface.
    # Keep signal timers as attributes for HUD only.
    global signals
    # if signals array not init, create dummy timers
    if len(signals) != noOfSignals:
        initialize_signals_for_dqn()
    # set timers such that currentGreen displays seconds_left
    for i in range(noOfSignals):
        if i == green_idx:
            signals[i].green = seconds_left
        else:
            signals[i].red = max(signals[i].red - 1, 0)

def initialize_signals_for_dqn():
    global signals
    signals = []
    signals.append(TrafficSignal(0, 2, last_assigned_greens[0]))
    for i in range(1, noOfSignals):
        signals.append(TrafficSignal(150, 2, last_assigned_greens[i]))

# ----------------------------
# Generator & main loop (visual)
# ----------------------------
def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(0,2)
        dir_index = random.randint(0, noOfSignals-1)
        Vehicle(lane_number, vehicleTypes[vehicle_type], dir_index)
        time.sleep(0.25 + random.random()*0.5)

def initialize_display_objects():
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("SIM - DQN Traffic (6-Way Star)")
    return screen

def main():
    # threads: controller, generator
    threading.Thread(target=controller_loop, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()

    screen = initialize_display_objects()
    background = load_background()
    font = pygame.font.Font(None, 30)
    infoFont = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    metric_timer = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        screen.blit(background, (0,0))

        # draw signals (visual helper)
        for i in range(noOfSignals):
            sig = signals[i] if i < len(signals) else TrafficSignal(150,2,last_assigned_greens[i])
            img = sig_green if (i == currentGreen and currentYellow == 0) else (sig_yellow if (i == currentGreen and currentYellow == 1) else sig_red)
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")

        # timers
        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])

        # vehicles snapshot
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
            text = infoFont.render(f"{lab} vehicles: {val}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 22

        pygame.display.update()
        clock.tick(FPS)

        # metrics print
        if time.time() - metric_timer >= 10.0:
            metric_timer = time.time()
            print(f"[METRICS @ {time.strftime('%H:%M:%S')}] Throughput: {getThroughput()}, Avg Wait Time: {getAverageWaitTime():.2f}s, Replay:{len(agent.replay)}")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    # create initial signals for display
    initialize_signals_for_dqn()
    # start pygame and main loop
    main()
