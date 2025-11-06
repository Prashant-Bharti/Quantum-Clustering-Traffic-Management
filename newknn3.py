# normal_clustering_traffic.py
# Classical KNN-based clustering for adaptive traffic signal allocation
# Uses same simulation, vehicles, and UI as quantum_traffic_swap.py

import random, time, threading, sys, math
import pygame
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------
# Config
# ----------------------------
BASE_GREEN = 5
MAX_GREEN = 30
FPS = 30

defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 2

signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Speeds and positions
speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[],1:[],2:[], 'crossed':0},
            'down':  {0:[],1:[],2:[], 'crossed':0},
            'left':  {0:[],1:[],2:[], 'crossed':0},
            'up':    {0:[],1:[],2:[], 'crossed':0}}

vehicleTypes = {0:'car',1:'bus',2:'truck',3:'bike'}
directionNumbers = {0:'right',1:'down',2:'left',3:'up'}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

stopLines = {'right':590, 'down':330, 'left':800, 'up':535}
defaultStop = {'right':570, 'down':310, 'left':820, 'up':555}

stoppingGap = 10
movingGap = 10

# Pygame init
pygame.init()
simulation = pygame.sprite.Group()

vehicles_lock = threading.Lock()
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
        global throughput_count, total_wait_time, exited_vehicles
        d = self.direction
        w, h = self.image.get_rect().width, self.image.get_rect().height

        if d == 'right':
            if self.crossed == 0 and self.x + w > stopLines[d]:
                self.crossed = 1
            if (self.x + w <= self.stop or self.crossed or (currentGreen == 0 and currentYellow == 0)):
                self.x += self.speed
            if not hasattr(self, 'exited') and self.x > 1400:
                self.exited = True
                self.exit_time = time.time()

        elif d == 'down':
            if self.crossed == 0 and self.y + h > stopLines[d]:
                self.crossed = 1
            if (self.y + h <= self.stop or self.crossed or (currentGreen == 1 and currentYellow == 0)):
                self.y += self.speed
            if not hasattr(self, 'exited') and self.y > 900:
                self.exited = True
                self.exit_time = time.time()

        elif d == 'left':
            if self.crossed == 0 and self.x < stopLines[d]:
                self.crossed = 1
            if (self.x >= self.stop or self.crossed or (currentGreen == 2 and currentYellow == 0)):
                self.x -= self.speed
            if not hasattr(self, 'exited') and self.x < -200:
                self.exited = True
                self.exit_time = time.time()

        elif d == 'up':
            if self.crossed == 0 and self.y < stopLines[d]:
                self.crossed = 1
            if (self.y >= self.stop or self.crossed or (currentGreen == 3 and currentYellow == 0)):
                self.y -= self.speed
            if not hasattr(self, 'exited') and self.y < -200:
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
# Classical clustering-based allocation
# ----------------------------
def updateGreenTimesFromKNN():
    global defaultGreen

    # Define initial 4 lane cluster centers (on-road waiting positions)
    k_centers = np.array([
        [200, 360],   # right
        [750, 200],   # down
        [1200, 480],  # left
        [650, 700]    # up
    ])

    # Get all current vehicle coordinates
    with vehicles_lock:
        coords = []
        for dir_name in ['right','down','left','up']:
            for lane in range(3):
                for v in vehicles[dir_name][lane]:
                    coords.append([v.x, v.y])
    if not coords:
        return

    coords = np.array(coords)
    labels = []

    # Assign each vehicle to nearest cluster center
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(k_centers, [0,1,2,3])
    assigned = neigh.predict(coords)

    # Count vehicles per cluster
    counts = [np.sum(assigned==i) for i in range(4)]

    total = sum(counts)
    if total == 0:
        total = 1

    # Convert to green time proportionally
    greens = {}
    for i in range(4):
        greens[i] = int(BASE_GREEN + (MAX_GREEN - BASE_GREEN) * (counts[i] / total))
        greens[i] = max(BASE_GREEN, min(MAX_GREEN, greens[i]))

    defaultGreen = greens
    print("[KNN] Cluster counts:", counts, "→ Green times:", greens)

# ----------------------------
# Controller & signal logic
# ----------------------------
def initialize_signals():
    global signals
    signals = []
    signals.append(TrafficSignal(0, defaultYellow, defaultGreen[0]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[1]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[2]))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen[3]))

def controller_loop():
    global currentGreen, nextGreen, currentYellow, signals
    initialize_signals()
    while True:
        updateGreenTimesFromKNN()
        signals[currentGreen].green = defaultGreen[currentGreen]
        while signals[currentGreen].green > 0:
            updateValues()
            time.sleep(1)
        currentYellow = 1
        while signals[currentGreen].yellow > 0:
            updateValues()
            time.sleep(1)
        currentYellow = 0
        signals[currentGreen].green = defaultGreen[currentGreen]
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
# Vehicle generation
# ----------------------------
def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(0,2)
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
# Metrics
# ----------------------------
def getLiveVehicleCounts():
    with vehicles_lock:
        return {direction: sum(len(vehicles[direction][lane]) for lane in range(3)) for direction in directionNumbers.values()}

def getThroughput():
    with vehicles_lock:
        return throughput_count

def getAverageWaitTime():
    with vehicles_lock:
        return (total_wait_time / exited_vehicles) if exited_vehicles > 0 else 0.0

# ----------------------------
# Main simulation loop
# ----------------------------
def main():
    threading.Thread(target=controller_loop, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()

    screen = pygame.display.set_mode((1400,800))
    pygame.display.set_caption("SIM - Classical KNN Traffic")
    try:
        background = pygame.image.load('images/intersection.png')
    except Exception:
        background = pygame.Surface((1400,800)); background.fill((50,50,50))

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

        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])

        with vehicles_lock:
            all_vehicles = list(simulation.sprites())
        for v in all_vehicles:
            screen.blit(v.image, (v.x, v.y))
            v.move()

        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for dir_idx, direction in directionNumbers.items():
            text = infoFont.render(f"{direction.upper()} vehicles: {vehicleCounts[direction]}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.update()
        clock.tick(FPS)

        if time.time() - metric_timer >= 10.0:
            metric_timer = time.time()
            print(f"[METRICS @ {time.strftime('%H:%M:%S')}] Throughput: {getThroughput()}, Avg Wait: {getAverageWaitTime():.2f}s")

if __name__ == "__main__":
    main()
