import random
import time
import threading
import pygame
import sys
from sklearn.cluster import KMeans


defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 2

signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0

speeds = {'car': 8.0, 'bus': 7.2, 'truck': 7.0, 'bike': 9.5}

x = {
    'right':[0, 60, 120],
    'down':[755, 815, 875],
    'left':[1400, 1340, 1280],
    'up':[602, 542, 482]
}
y = {
    'right':[348, 408, 468],
    'down':[0, 60, 120],
    'left':[498, 438, 378],
    'up':[800, 740, 680]
}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0},
            'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 570, 'down': 310, 'left': 820, 'up': 555}

stoppingGap = 10
movingGap = 10

pygame.init()
simulation = pygame.sprite.Group()

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
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * 0.5), int(self.image.get_height() * 0.5)))

        if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
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

        if direction == 'right': x[direction][lane] -= self.image.get_rect().width + stoppingGap
        elif direction == 'left': x[direction][lane] += self.image.get_rect().width + stoppingGap
        elif direction == 'down': y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == 'up': y[direction][lane] += self.image.get_rect().height + stoppingGap
        simulation.add(self)

    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.image.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
            if ((self.x + self.image.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0))
                and (self.index == 0 or self.x + self.image.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - movingGap))):
                self.x += self.speed
        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.image.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
            if ((self.y + self.image.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0))
                and (self.index == 0 or self.y + self.image.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - movingGap))):
                self.y += self.speed
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
            if ((self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0))
                and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].image.get_rect().width + movingGap))):
                self.x -= self.speed
        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
            if ((self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0))
                and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].image.get_rect().height + movingGap))):
                self.y -= self.speed

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
            green_time = int(max(5, min(30, int(sum(cluster_sizes) * 0.7)))/1.8) # Making green time proportional to total clustered vehicle count
        else:
            green_time = 5
        newTimes[dir_idx] = green_time
    defaultGreen = newTimes



def getLiveVehicleCounts():
    counts = {}
    for dir_idx, direction in directionNumbers.items():
        total = sum(len(vehicles[direction][lane]) for lane in range(3))
        counts[direction] = total
    return counts

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

def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

def generateVehicles():
    while True:
        vehicle_type = random.randint(0,3)
        lane_number = random.randint(1,2)
        temp = random.randint(0,99)
        dist = [40,70,90,100]  # Weighted direction chance
        if temp < dist[0]: direction_number = 0
        elif temp < dist[1]: direction_number = 1
        elif temp < dist[2]: direction_number = 2
        else: direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(0.5)

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
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0,0))

        for i in range(noOfSignals):
            sig = signals[i]
            if i == currentGreen:
                if currentYellow: screen.blit(yellowSignal, signalCoods[i]); sig.signalText = sig.yellow
                else: screen.blit(greenSignal, signalCoods[i]); sig.signalText = sig.green
            else:
                screen.blit(redSignal, signalCoods[i])
                sig.signalText = sig.red if sig.red <= 10 else "---"

        for i in range(noOfSignals):
            text = font.render(str(signals[i].signalText), True, (255,255,255), (0,0,0))
            screen.blit(text, signalTimerCoods[i])

        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()

        # Draw live vehicle counts
        vehicleCounts = getLiveVehicleCounts()
        y_offset = 10
        for dir_idx, direction in directionNumbers.items():
            text = infoFont.render(f"{direction.upper()} vehicles: {vehicleCounts[direction]}", True, (255,255,0))
            screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.update()

Main()
