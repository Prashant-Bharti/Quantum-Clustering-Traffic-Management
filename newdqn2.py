# traffic_dqn_sim.py
# Traffic simulation identical to quantum_traffic_swap.py, but with DQN-based green time control

import random, time, threading, sys, math, collections
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Config
# ----------------------------
FPS = 30
MAX_GREEN = 30
BASE_GREEN = 5
defaultRed = 150
defaultYellow = 2

# Pygame setup
pygame.init()
simulation = pygame.sprite.Group()

# Directions and globals
signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

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

vehicles_lock = threading.Lock()

# Metrics
throughput_count = 0
total_wait_time = 0.0
exited_vehicles = 0

# ----------------------------
# Pygame Classes
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
        with vehicles_lock:
            vehicles[direction][lane].append(self)
            self.index = len(vehicles[direction][lane]) - 1
        try:
            path = "images/" + direction + "/" + vehicleClass + ".png"
            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (int(img.get_width()*0.5), int(img.get_height()*0.5)))
            self.image = img.convert_alpha()
        except Exception:
            surf = pygame.Surface((30,15))
            surf.fill((0,200,0))
            self.image = surf
        if self.index > 0 and vehicles[direction][lane][self.index-1].crossed == 0:
            prev = vehicles[direction][lane][self.index-1]
            if direction == 'right': self.stop = prev.stop - prev.image.get_rect().width - 10
            elif direction == 'left': self.stop = prev.stop + prev.image.get_rect().width + 10
            elif direction == 'down': self.stop = prev.stop - prev.image.get_rect().height - 10
            elif direction == 'up': self.stop = prev.stop + prev.image.get_rect().height + 10
        else:
            self.stop = defaultStop[direction]
        if direction == 'right': x[direction][lane] -= self.image.get_rect().width + 10
        elif direction == 'left': x[direction][lane] += self.image.get_rect().width + 10
        elif direction == 'down': y[direction][lane] -= self.image.get_rect().height + 10
        elif direction == 'up': y[direction][lane] += self.image.get_rect().height + 10
        simulation.add(self)

    def move(self):
        global throughput_count, total_wait_time, exited_vehicles
        d = self.direction
        w, h = self.image.get_rect().width, self.image.get_rect().height
        if d == 'right':
            if self.crossed == 0 and self.x + w > stopLines[d]: self.crossed = 1
            if (self.x + w <= self.stop or self.crossed or (currentGreen == 0 and currentYellow == 0)): self.x += self.speed
            if not hasattr(self, 'exited') and self.x > 1400:
                self.exited = True; self.exit_time = time.time()
        elif d == 'down':
            if self.crossed == 0 and self.y + h > stopLines[d]: self.crossed = 1
            if (self.y + h <= self.stop or self.crossed or (currentGreen == 1 and currentYellow == 0)): self.y += self.speed
            if not hasattr(self, 'exited') and self.y > 900:
                self.exited = True; self.exit_time = time.time()
        elif d == 'left':
            if self.crossed == 0 and self.x < stopLines[d]: self.crossed = 1
            if (self.x >= self.stop or self.crossed or (currentGreen == 2 and currentYellow == 0)): self.x -= self.speed
            if not hasattr(self, 'exited') and self.x < -200:
                self.exited = True; self.exit_time = time.time()
        elif d == 'up':
            if self.crossed == 0 and self.y < stopLines[d]: self.crossed = 1
            if (self.y >= self.stop or self.crossed or (currentGreen == 3 and currentYellow == 0)): self.y -= self.speed
            if not hasattr(self, 'exited') and self.y < -200:
                self.exited = True; self.exit_time = time.time()
        if hasattr(self, 'exited') and self.exited and self.exit_time is not None:
            if not hasattr(self, '_counted') or not self._counted:
                self._counted = True
                with vehicles_lock:
                    throughput_count += 1
                    total_wait_time += max(0.0, self.exit_time - self.spawn_time)
                    exited_vehicles += 1

# ----------------------------
# DQN Model
# ----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.layers(x)

# Replay buffer
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory:
    def __init__(self, capacity=5000):
        self.memory = collections.deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    def __init__(self):
        self.state_dim = 4   # vehicle counts per direction
        self.action_dim = 4  # which signal to turn green
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(3000)
        self.gamma = 0.95
        self.epsilon = 0.2
        self.batch_size = 64
        self.update_target_every = 100
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_vals = self.policy_net(torch.tensor(state, dtype=torch.float32))
            return int(torch.argmax(q_vals).item())

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q = reward_batch + self.gamma * next_q
        loss = nn.functional.mse_loss(q_values.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done += 1

agent = DQNAgent()

# ----------------------------
# Helpers
# ----------------------------
def get_vehicle_counts():
    with vehicles_lock:
        return np.array([sum(len(vehicles[d][l]) for l in range(3)) for d in ['right','down','left','up']], dtype=float)

def compute_reward(prev_counts, new_counts):
    return float(np.sum(prev_counts - new_counts))  # reward for clearing traffic

def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(0,2)
        direction_number = random.randint(0,3)
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(0.5)

# ----------------------------
# Controller Loop with DQN
# ----------------------------
def controller_loop():
    global currentGreen, nextGreen, currentYellow
    initialize_signals()
    prev_counts = get_vehicle_counts()
    while True:
        state = prev_counts / (1 + np.max(prev_counts))
        action = agent.select_action(state)
        currentGreen = action
        nextGreen = (currentGreen + 1) % 4
        signals[currentGreen].green = BASE_GREEN + random.randint(0, MAX_GREEN - BASE_GREEN)
        while signals[currentGreen].green > 0:
            updateValues()
            time.sleep(1)
        currentYellow = 1
        while signals[currentGreen].yellow > 0:
            updateValues()
            time.sleep(1)
        currentYellow = 0
        new_counts = get_vehicle_counts()
        reward = compute_reward(prev_counts, new_counts)
        agent.memory.push(state, action, reward, new_counts / (1 + np.max(new_counts)))
        agent.optimize()
        prev_counts = new_counts

def initialize_signals():
    global signals
    signals.clear()
    signals.append(TrafficSignal(0, defaultYellow, BASE_GREEN))
    for _ in range(3):
        signals.append(TrafficSignal(defaultRed, defaultYellow, BASE_GREEN))

def updateValues():
    global signals, currentGreen, currentYellow
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0: signals[i].green -= 1
            else: signals[i].yellow -= 1
        else: signals[i].red -= 1

# ----------------------------
# Main Pygame loop
# ----------------------------
def main():
    threading.Thread(target=controller_loop, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()
    screen = pygame.display.set_mode((1400,800))
    pygame.display.set_caption("SIM - DQN Traffic Controller")
    try: background = pygame.image.load('images/intersection.png')
    except: background = pygame.Surface((1400,800)); background.fill((50,50,50))
    try:
        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
    except:
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
        for i in range(noOfSignals):
            sig = signals[i]
            img = yellowSignal if i == currentGreen and currentYellow else greenSignal if i == currentGreen else redSignal
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")
            text = font.render(str(sig.signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])
        with vehicles_lock:
            all_vehicles = list(simulation.sprites())
        for v in all_vehicles:
            screen.blit(v.image, (v.x, v.y))
            v.move()
        vehicleCounts = get_vehicle_counts()
        y_offset = 10
        for i, dir_name in directionNumbers.items():
            text = infoFont.render(f"{dir_name.upper()} vehicles: {int(vehicleCounts[i])}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25
        pygame.display.update()
        clock.tick(FPS)
        if time.time() - metric_timer >= 10.0:
            metric_timer = time.time()
            print(f"[DQN METRICS] Avg Wait Time: {total_wait_time/max(1,exited_vehicles):.2f}s, Throughput: {throughput_count}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("Running DQN-based adaptive traffic signal control (UI identical to quantum model)...")
    main()
