# Required Libraries
import random
import time
import threading
import pygame
import sys
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# === Global Simulation Parameters and Metrics ===

vehicle_wait_times = []           # Stores wait time of individual vehicles
vehicle_crossed_count = 0         # Total number of vehicles that have crossed intersection
vehicle_entry_times = {}          # Entry timestamps for individual vehicles (not used currently)

# Signal timing setup
defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 2

signals = []                      # List of signal objects
noOfSignals = 4
currentGreen = 0                  # Index of currently green signal
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0

# Vehicle speeds (pixels per frame)
speeds = {'car': 8.0, 'bus': 7.2, 'truck': 7.0, 'bike': 9.5}

# Spawn positions for vehicles by direction and lane
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

# Store vehicles per direction and lane
vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0},
            'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}

# Mappings for vehicle and direction naming
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Signal display positions
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

# Stop line locations for traffic light logic
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 570, 'down': 310, 'left': 820, 'up': 555}

# Gap thresholds
stoppingGap = 10
movingGap = 10

# Initialize PyGame and simulation group
pygame.init()
simulation = pygame.sprite.Group()

# === Traffic Signal Class ===
class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

# === Vehicle Class ===
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
        self.created_time = datetime.now()
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        # Load vehicle image
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * 0.5), int(self.image.get_height() * 0.5)))

        # Set stopping position based on previous vehicle in same lane
        if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
            prev = vehicles[direction][lane][self.index-1]
            if direction == 'right': self.stop = prev.stop - prev.image.get_rect().width - stoppingGap
            elif direction == 'left': self.stop = prev.stop + prev.image.get_rect().width + stoppingGap
            elif direction == 'down': self.stop = prev.stop - prev.image.get_rect().height - stoppingGap
            elif direction == 'up': self.stop = prev.stop + prev.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # Update spawn location for next vehicle
        if direction == 'right': x[direction][lane] -= self.image.get_rect().width + stoppingGap
        elif direction == 'left': x[direction][lane] += self.image.get_rect().width + stoppingGap
        elif direction == 'down': y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == 'up': y[direction][lane] += self.image.get_rect().height + stoppingGap

        simulation.add(self)

    # === Vehicle Movement Logic ===
    def move(self):
        global vehicle_crossed_count, vehicle_wait_times
        d, w, h = self.direction, self.image.get_rect().width, self.image.get_rect().height
        # Update position based on current signal and vehicle state
        if d == 'right':
            if self.crossed == 0 and self.x + w > stopLines[d]:
                self.crossed = 1
                vehicle_crossed_count += 1
                wait_time = (datetime.now() - self.created_time).total_seconds()
                vehicle_wait_times.append(wait_time)
            if (self.x + w <= self.stop or self.crossed or (currentGreen == 0 and currentYellow == 0)) and \
               (self.index == 0 or self.x + w < vehicles[d][self.lane][self.index-1].x - movingGap): self.x += self.speed

        elif d == 'down':
            if self.crossed == 0 and self.y + h > stopLines[d]:
                self.crossed = 1
                vehicle_crossed_count += 1
                wait_time = (datetime.now() - self.created_time).total_seconds()
                vehicle_wait_times.append(wait_time)
            if (self.y + h <= self.stop or self.crossed or (currentGreen == 1 and currentYellow == 0)) and \
               (self.index == 0 or self.y + h < vehicles[d][self.lane][self.index-1].y - movingGap): self.y += self.speed

        elif d == 'left':
            if self.crossed == 0 and self.x < stopLines[d]:
                self.crossed = 1
                vehicle_crossed_count += 1
                wait_time = (datetime.now() - self.created_time).total_seconds()
                vehicle_wait_times.append(wait_time)
            if (self.x >= self.stop or self.crossed or (currentGreen == 2 and currentYellow == 0)) and \
               (self.index == 0 or self.x > vehicles[d][self.lane][self.index-1].x + vehicles[d][self.lane][self.index-1].image.get_rect().width + movingGap): self.x -= self.speed

        elif d == 'up':
            if self.crossed == 0 and self.y < stopLines[d]:
                self.crossed = 1
                vehicle_crossed_count += 1
                wait_time = (datetime.now() - self.created_time).total_seconds()
                vehicle_wait_times.append(wait_time)
            if (self.y >= self.stop or self.crossed or (currentGreen == 3 and currentYellow == 0)) and \
               (self.index == 0 or self.y > vehicles[d][self.lane][self.index-1].y + vehicles[d][self.lane][self.index-1].image.get_rect().height + movingGap): self.y -= self.speed

# === Update green signal times using live KMeans clustering ===
def updateGreenTimesFromClustering():
    global defaultGreen
    newTimes = {}
    for dir_idx, direction in directionNumbers.items():
        coords = []
        for lane in range(3):
            for v in vehicles[direction][lane]:
                coords.append([v.x, v.y])
        count = len(coords)
        if count > 0:
            kmeans = KMeans(n_clusters=min(count, 5), n_init='auto').fit(coords)
            labels = kmeans.labels_
            cluster_sizes = [list(labels).count(i) for i in range(len(set(labels)))]
            green_time = int(max(5, min(30, int(sum(cluster_sizes) * 0.7)))/1.8)
        else:
            green_time = 5
        newTimes[dir_idx] = green_time
    defaultGreen = newTimes

# === Count vehicles in real-time ===
def getLiveVehicleCounts():
    counts = {}
    for dir_idx, direction in directionNumbers.items():
        total = sum(len(vehicles[direction][lane]) for lane in range(3))
        counts[direction] = total
    return counts

# === Initialize traffic signals ===
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)
    repeat()

# === Main traffic light control loop ===
def repeat():
    global currentGreen, currentYellow, nextGreen
    updateGreenTimesFromClustering()
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[nextGreen].red = defaultYellow

    while signals[currentGreen].green > 0:
        if all(v.crossed for lane in range(3) for v in vehicles[directionNumbers[currentGreen]][lane]):
            break
        updateValues()
        time.sleep(1)

    currentYellow = 1
    for i in range(3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]

    while signals[currentGreen].yellow > 0:
        updateValues()
        time.sleep(1)

    currentYellow = 0
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    currentGreen = nextGreen
    nextGreen = (currentGreen+1)%noOfSignals
    repeat()

# === Decrement signal timers ===
def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

# === Generate vehicles continuously ===
def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(1,2)
        temp = random.randint(0,99)
        dist = [40,70,90,100]
        if temp < dist[0]: direction_number = 0
        elif temp < dist[1]: direction_number = 1
        elif temp < dist[2]: direction_number = 2
        else: direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(0.5)

# === Print performance metrics periodically ===
def printMetrics():
    while True:
        time.sleep(10)
        if vehicle_wait_times:
            avg_wait = sum(vehicle_wait_times) / len(vehicle_wait_times)
        else:
            avg_wait = 0
        print(f"[METRICS @ {datetime.now().strftime('%H:%M:%S')}] Throughput: {vehicle_crossed_count}, Average Wait Time: {avg_wait:.2f}s")

# === Main simulation function ===
def main():
    threading.Thread(target=initialize, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()
    threading.Thread(target=printMetrics, daemon=True).start()

    screen = pygame.display.set_mode((1400,800))
    pygame.display.set_caption("SIMULATION")
    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    infoFont = pygame.font.Font(None, 26)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        screen.blit(background, (0,0))
        for i in range(noOfSignals):
            sig = signals[i]
            img = yellowSignal if i == currentGreen and currentYellow else greenSignal if i == currentGreen else redSignal
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")
        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for dir_idx, direction in directionNumbers.items():
            text = infoFont.render(f"{direction.upper()} vehicles: {vehicleCounts[direction]}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25
        pygame.display.update()

# === Start Simulation ===
main()
