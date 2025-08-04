# Import necessary libraries
import random, time, threading, pygame, sys
import numpy as np
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from sklearn.preprocessing import normalize

# ---------------------------------------------
# INITIALIZATION OF SIGNAL TIMINGS AND VEHICLE DATA STRUCTURES
# ---------------------------------------------

# Default traffic light durations
defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 2

# Signal and simulation metadata
signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Vehicle speed settings for different types
speeds = {'car': 6.0, 'bus': 5.2, 'truck': 5.0, 'bike': 7.5}

# Initial spawn positions for each lane and direction
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

# Vehicles dictionary to track vehicles per direction and lane
vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0},
            'down':  {0:[], 1:[], 2:[], 'crossed':0},
            'left':  {0:[], 1:[], 2:[], 'crossed':0},
            'up':    {0:[], 1:[], 2:[], 'crossed':0}}

# Mapping dictionaries
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinate mapping for signal rendering
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

# Line at which vehicles stop
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 570, 'down': 310, 'left': 820, 'up': 555}

# Gaps between vehicles
stoppingGap = 10
movingGap = 10

# Pygame setup
pygame.init()
simulation = pygame.sprite.Group()

# ---------------------------------------------
# CLASS DEFINITIONS
# ---------------------------------------------

class TrafficSignal:
    """Represents a single traffic signal with red, yellow, and green durations."""
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class Vehicle(pygame.sprite.Sprite):
    """Represents a vehicle in the simulation."""
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

        # Append to global vehicle tracker
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        # Load vehicle image
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * 0.5), int(self.image.get_height() * 0.5)))

        # Set stopping point based on vehicle in front
        if self.index > 0 and vehicles[direction][lane][self.index-1].crossed == 0:
            prev = vehicles[direction][lane][self.index-1]
            if direction == 'right': self.stop = prev.stop - prev.image.get_rect().width - stoppingGap
            elif direction == 'left': self.stop = prev.stop + prev.image.get_rect().width + stoppingGap
            elif direction == 'down': self.stop = prev.stop - prev.image.get_rect().height - stoppingGap
            elif direction == 'up': self.stop = prev.stop + prev.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # Update initial spawn position
        if direction == 'right': x[direction][lane] -= self.image.get_rect().width + stoppingGap
        elif direction == 'left': x[direction][lane] += self.image.get_rect().width + stoppingGap
        elif direction == 'down': y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == 'up': y[direction][lane] += self.image.get_rect().height + stoppingGap
        simulation.add(self)

    def move(self):
        """Move the vehicle if allowed by signal and traffic conditions."""
        d, w, h = self.direction, self.image.get_rect().width, self.image.get_rect().height
        if d == 'right':
            if self.crossed == 0 and self.x + w > stopLines[d]: self.crossed = 1
            if (self.x + w <= self.stop or self.crossed or (currentGreen == 0 and currentYellow == 0)) and \
               (self.index == 0 or self.x + w < vehicles[d][self.lane][self.index-1].x - movingGap): self.x += self.speed
        elif d == 'down':
            if self.crossed == 0 and self.y + h > stopLines[d]: self.crossed = 1
            if (self.y + h <= self.stop or self.crossed or (currentGreen == 1 and currentYellow == 0)) and \
               (self.index == 0 or self.y + h < vehicles[d][self.lane][self.index-1].y - movingGap): self.y += self.speed
        elif d == 'left':
            if self.crossed == 0 and self.x < stopLines[d]: self.crossed = 1
            if (self.x >= self.stop or self.crossed or (currentGreen == 2 and currentYellow == 0)) and \
               (self.index == 0 or self.x > vehicles[d][self.lane][self.index-1].x + vehicles[d][self.lane][self.index-1].image.get_rect().width + movingGap): self.x -= self.speed
        elif d == 'up':
            if self.crossed == 0 and self.y < stopLines[d]: self.crossed = 1
            if (self.y >= self.stop or self.crossed or (currentGreen == 3 and currentYellow == 0)) and \
               (self.index == 0 or self.y > vehicles[d][self.lane][self.index-1].y + vehicles[d][self.lane][self.index-1].image.get_rect().height + movingGap): self.y -= self.speed

# ---------------------------------------------
# QUANTUM-BASED GREEN TIME ADJUSTMENT
# ---------------------------------------------

def swap_test_similarity(vec1, vec2, shots=256):
    """Performs the quantum swap test to compute similarity between two normalized vectors."""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    n = int(np.ceil(np.log2(len(vec1))))
    pad = 2**n
    vec1 = np.pad(vec1, (0, pad - len(vec1)))
    vec2 = np.pad(vec2, (0, pad - len(vec2)))

    qc = QuantumCircuit(1 + 2*n, 1)
    qc.h(0)
    qc.append(Initialize(vec1), list(range(1, 1+n)))
    qc.append(Initialize(vec2), list(range(1+n, 1+2*n)))
    for i in range(n):
        qc.cswap(0, 1+i, 1+n+i)
    qc.h(0)
    qc.measure(0, 0)

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    prob0 = counts.get('0', 0) / shots
    return 2 * prob0 - 1

def updateGreenTimesFromQuantumClustering():
    """Dynamically adjusts green signal durations based on quantum clustering using traffic density vectors."""
    global defaultGreen
    newTimes = {}
    for dir_idx, direction in directionNumbers.items():
        coords = []
        for lane in range(3):
            for v in vehicles[direction][lane]:
                coords.append([v.x, v.y])
        if not coords:
            newTimes[dir_idx] = 5
            continue
        coords = normalize(np.array(coords))
        centroids = coords[np.random.choice(len(coords), min(len(coords), 3), replace=False)]
        cluster_sizes = [0] * len(centroids)
        for pt in coords:
            sims = [swap_test_similarity(pt, c) for c in centroids]
            cluster_sizes[np.argmax(sims)] += 1
        green_time = int(max(3, min(30, int(sum(cluster_sizes) * 0.7))) / 2)
        newTimes[dir_idx] = green_time
    print("New quantum green times:", newTimes)
    defaultGreen = newTimes

# ---------------------------------------------
# SIGNAL TIMING LOGIC
# ---------------------------------------------

def getLiveVehicleCounts():
    """Returns current vehicle counts per direction."""
    return {direction: sum(len(vehicles[direction][lane]) for lane in range(3)) for direction in directionNumbers.values()}

def initialize():
    """Initializes the traffic signal objects and starts the control loop."""
    signals.extend([
        TrafficSignal(0, defaultYellow, defaultGreen[0]),
        TrafficSignal(defaultRed, defaultYellow, defaultGreen[1]),
        TrafficSignal(defaultRed, defaultYellow, defaultGreen[2]),
        TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    ])
    repeat()

def repeat():
    """Controls the traffic signal transitions and updates."""
    global currentGreen, currentYellow, nextGreen

    updateGreenTimesFromQuantumClustering()
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[nextGreen].red = defaultYellow

    # GREEN PHASE
    while signals[currentGreen].green > 0:
        updateValues()
        time.sleep(1)

    # YELLOW PHASE
    currentYellow = 1
    for i in range(3):
        for v in vehicles[directionNumbers[currentGreen]][i]:
            v.stop = defaultStop[directionNumbers[currentGreen]]
    while signals[currentGreen].yellow > 0:
        updateValues()
        time.sleep(1)

    # RESET SIGNAL
    currentYellow = 0
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed

    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    repeat()

def updateValues():
    """Updates signal counters every second."""
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

def generateVehicles():
    """Spawns new vehicles randomly at fixed intervals."""
    while True:
        vehicle_type = random.randint(0, 3)
        lane_number = random.randint(1, 2)
        temp = random.randint(0, 99)
        dist = [40, 70, 90, 100]
        direction_number = 0 if temp < dist[0] else 1 if temp < dist[1] else 2 if temp < dist[2] else 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(0.5)

# ---------------------------------------------
# MAIN LOOP WITH GRAPHICS RENDERING
# ---------------------------------------------

class Main:
    threading.Thread(target=initialize, daemon=True).start()
    threading.Thread(target=generateVehicles, daemon=True).start()
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

        # Draw signals and their timers
        for i in range(noOfSignals):
            sig = signals[i]
            img = yellowSignal if i == currentGreen and currentYellow else greenSignal if i == currentGreen else redSignal
            screen.blit(img, signalCoods[i])
            sig.signalText = sig.yellow if currentYellow else sig.green if i == currentGreen else (sig.red if sig.red <= 10 else "---")
        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])

        # Draw vehicles
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()

        # Show vehicle counts on screen
        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for dir_idx, direction in directionNumbers.items():
            text = infoFont.render(f"{direction.upper()} vehicles: {vehicleCounts[direction]}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.update()

Main()
