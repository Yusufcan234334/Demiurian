#This code is AI

import pygame
import random
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import torch

# =====================================================================
# CONFIG
# =====================================================================
WORLD_W, WORLD_H = 4800, 3200
GRID_SIZE = 100
GRID_COLS = WORLD_W // GRID_SIZE
GRID_ROWS = WORLD_H // GRID_SIZE

FPS = 60
CAMERA_SPEED = 900.0  # px/sec

DAY_LENGTH_TICKS = 480          # 1 gün-gece döngüsü
SEASON_LENGTH_TICKS = 2400      # 1 mevsim
YEAR_LENGTH_TICKS = SEASON_LENGTH_TICKS * 4
SEASONS = ["İlkbahar", "Yaz", "Sonbahar", "Kış"]
SEASON_TEMP_MOD = [0.0, 0.18, -0.05, -0.30]
SEASON_HUMID_MOD = [0.05, -0.10, 0.10, -0.05]
SEASON_GROWTH_MOD = [1.1, 1.3, 0.8, 0.35]

MAX_PLANTS_PER_CELL = 6
BASE_GROWTH_PROB = 0.010

MAX_CREATURES = 220
INITIAL_SPECIES = 7
INITIAL_PER_SPECIES = 10

SIGHT_MIN, SIGHT_MAX = 120, 420
LEG_OPTIONS = [0, 2, 4, 6, 8]

EQ_WINDOW = 300          # kaç tick'lik pencereyle ölçülecek
EQ_HISTORY_LEN = 12      # kaç pencere art arda stabil olmalı
EQ_CV_THRESHOLD = 0.12   # varyasyon katsayısı eşiği (düşük = stabil)

RECORD_INTERVAL = 20     # kaç tick'te bir yeni x örnekleri alınsın
RECORD_DELTA_TICKS = 90  # x alındıktan kaç tick sonra y (delta) hesaplansın
SAMPLES_PER_RECORD = 14  # her seferinde kaç hücre örneklensin
FLUSH_EVERY_SAMPLES = 200  # buffer bu kadar dolunca diske yaz

SENSE_RADIUS = 260  # nearby_* hesaplarken kullanılan yarıçap (px)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
X_PATH = os.path.join(OUTPUT_DIR, "x.pt")
Y_PATH = os.path.join(OUTPUT_DIR, "y.pt")

random.seed()

# =====================================================================
# ENUMS
# =====================================================================
class Diet(Enum):
    HERBIVORE = 0
    CARNIVORE = 1
    OMNIVORE = 2

class ReproMode(Enum):
    SEXUAL = 0
    ASEXUAL = 1

class BirthMode(Enum):
    MAMMAL = 0
    EGG = 1

# =====================================================================
# WORLD / GRID
# =====================================================================
class Cell:
    __slots__ = ("is_water", "base_temp", "base_humid", "capacity", "plants")

    def __init__(self, is_water, base_temp, base_humid, capacity):
        self.is_water = is_water
        self.base_temp = base_temp
        self.base_humid = base_humid
        self.capacity = capacity
        self.plants = 0 if is_water else random.randint(0, 2)


def clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def build_grid():
    grid = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    # basit göl(ler) tanımı
    lake_centers = [
        (GRID_COLS * 0.72, GRID_ROWS * 0.28, 6.5, 4.5),
        (GRID_COLS * 0.22, GRID_ROWS * 0.75, 4.0, 3.0),
    ]
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nx, ny = c / GRID_COLS, r / GRID_ROWS
            is_water = False
            for (lcx, lcy, rx, ry) in lake_centers:
                if ((c - lcx) / rx) ** 2 + ((r - lcy) / ry) ** 2 <= 1.0:
                    is_water = True
                    break
            base_temp = clamp01(0.55 + 0.35 * math.sin(nx * 3.1) * math.cos(ny * 2.0) - (ny - 0.5) * 0.35)
            base_humid = clamp01(0.5 + 0.35 * math.sin(nx * 1.7 + 1.3) + 0.25 * math.cos(ny * 2.6))
            dist_water = min(
                math.hypot((c - lcx), (r - lcy)) / max(rx, ry) for (lcx, lcy, rx, ry) in lake_centers
            )
            base_humid = clamp01(base_humid + max(0.0, 0.25 - dist_water * 0.06))
            capacity = clamp01(base_humid * 0.65 + (1 - abs(base_temp - 0.55)) * 0.45)
            grid[r][c] = Cell(is_water, base_temp, base_humid, capacity)
    return grid


class World:
    def __init__(self):
        self.grid = build_grid()
        self.tick = 0

    def season_index(self):
        return (self.tick % YEAR_LENGTH_TICKS) // SEASON_LENGTH_TICKS

    def light(self):
        phase = (self.tick % DAY_LENGTH_TICKS) / DAY_LENGTH_TICKS
        return clamp01(0.5 + 0.5 * math.sin(2 * math.pi * phase - math.pi / 2))

    def cell_at(self, x, y):
        c = min(max(int(x // GRID_SIZE), 0), GRID_COLS - 1)
        r = min(max(int(y // GRID_SIZE), 0), GRID_ROWS - 1)
        return self.grid[r][c], c, r

    def temperature_at(self, c, r):
        cell = self.grid[r][c]
        return clamp01(cell.base_temp + SEASON_TEMP_MOD[self.season_index()])

    def humidity_at(self, c, r):
        cell = self.grid[r][c]
        return clamp01(cell.base_humid + SEASON_HUMID_MOD[self.season_index()])

    def update_plants(self):
        s = self.season_index()
        growth_mod = SEASON_GROWTH_MOD[s]
        for r in range(GRID_ROWS):
            row = self.grid[r]
            for c in range(GRID_COLS):
                cell = row[c]
                if cell.is_water:
                    continue
                cap = int(cell.capacity * MAX_PLANTS_PER_CELL)
                if cell.plants < cap:
                    prob = BASE_GROWTH_PROB * cell.capacity * growth_mod * (0.4 + self.humidity_at(c, r))
                    if random.random() < prob:
                        cell.plants += 1


# =====================================================================
# GENOME
# =====================================================================
_next_species_id = [0]


def new_species_id():
    _next_species_id[0] += 1
    return _next_species_id[0]


@dataclass
class Genome:
    aquatic: bool
    legs: int
    sight_range: float
    repro_rate: float
    repro_mode: ReproMode
    birth_mode: BirthMode
    intelligence: float
    diet: Diet
    size: float
    species_id: int
    color: tuple

    def speed(self):
        base = 40 + self.legs * 9
        if self.aquatic:
            base += 25
        return base / (0.6 + self.size)

    def max_energy(self):
        return 60 + self.size * 90

    def max_age(self):
        return 1400 + self.intelligence * 900 + self.size * 400

    def mutated_copy(self, new_species=False):
        def jitter(v, amt):
            return v + random.uniform(-amt, amt)

        g = Genome(
            aquatic=self.aquatic if random.random() > 0.03 else (not self.aquatic),
            legs=max(0, min(8, self.legs + random.choice([-2, 0, 0, 0, 2]))),
            sight_range=clamp01(jitter(self.sight_range / SIGHT_MAX, 0.06)) * SIGHT_MAX,
            repro_rate=clamp01(jitter(self.repro_rate, 0.05)),
            repro_mode=self.repro_mode if random.random() > 0.02 else random.choice(list(ReproMode)),
            birth_mode=self.birth_mode if random.random() > 0.02 else random.choice(list(BirthMode)),
            intelligence=clamp01(jitter(self.intelligence, 0.05)),
            diet=self.diet if random.random() > 0.02 else random.choice(list(Diet)),
            size=max(0.3, min(2.2, jitter(self.size, 0.08))),
            species_id=new_species_id() if new_species else self.species_id,
            color=self.color,
        )
        return g

    @staticmethod
    def random_new():
        aquatic = random.random() < 0.3
        diet = random.choice(list(Diet))
        color = (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
        return Genome(
            aquatic=aquatic,
            legs=0 if aquatic else random.choice(LEG_OPTIONS),
            sight_range=random.uniform(SIGHT_MIN, SIGHT_MAX),
            repro_rate=random.uniform(0.2, 0.9),
            repro_mode=random.choice(list(ReproMode)),
            birth_mode=random.choice(list(BirthMode)),
            intelligence=random.uniform(0.1, 0.95),
            diet=diet,
            size=random.uniform(0.5, 1.8),
            species_id=new_species_id(),
            color=color,
        )

    @staticmethod
    def crossover(a: "Genome", b: "Genome"):
        pick = lambda x, y: x if random.random() < 0.5 else y
        g = Genome(
            aquatic=pick(a.aquatic, b.aquatic),
            legs=pick(a.legs, b.legs),
            sight_range=(a.sight_range + b.sight_range) / 2,
            repro_rate=(a.repro_rate + b.repro_rate) / 2,
            repro_mode=pick(a.repro_mode, b.repro_mode),
            birth_mode=pick(a.birth_mode, b.birth_mode),
            intelligence=(a.intelligence + b.intelligence) / 2,
            diet=pick(a.diet, b.diet),
            size=(a.size + b.size) / 2,
            species_id=a.species_id,
            color=pick(a.color, b.color),
        )
        return g.mutated_copy(new_species=False)


# =====================================================================
# CREATURES / EGGS
# =====================================================================
class Egg:
    __slots__ = ("x", "y", "genome", "incubation", "max_incubation")

    def __init__(self, x, y, genome, incubation=260):
        self.x, self.y = x, y
        self.genome = genome
        self.incubation = incubation
        self.max_incubation = incubation


class Creature:
    __slots__ = (
        "x", "y", "genome", "energy", "thirst", "age", "alive",
        "repro_cooldown", "vx", "vy", "state",
    )

    def __init__(self, x, y, genome: Genome):
        self.x, self.y = x, y
        self.genome = genome
        self.energy = genome.max_energy() * 0.7
        self.thirst = 30.0
        self.age = 0
        self.alive = True
        self.repro_cooldown = random.uniform(0, 150)
        self.vx = self.vy = 0.0
        self.state = "wander"

    def energy_ratio(self):
        return clamp01(self.energy / self.genome.max_energy())


# =====================================================================
# SPATIAL HASH (komşu bulmayı hızlandırmak için)
# =====================================================================
class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.buckets = defaultdict(list)

    def clear(self):
        self.buckets.clear()

    def key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj, x, y):
        self.buckets[self.key(x, y)].append(obj)

    def query_radius(self, x, y, radius):
        cr = int(radius // self.cell_size) + 1
        cx, cy = int(x // self.cell_size), int(y // self.cell_size)
        result = []
        for dx in range(-cr, cr + 1):
            for dy in range(-cr, cr + 1):
                result.extend(self.buckets.get((cx + dx, cy + dy), ()))
        return result


# =====================================================================
# SIMULATION
# =====================================================================
class Simulation:
    def __init__(self):
        self.world = World()
        self.creatures = []
        self.eggs = []
        self.spawn_initial_population()

        self.pop_history = deque(maxlen=EQ_WINDOW)
        self.food_history = deque(maxlen=EQ_WINDOW)
        self.stable_windows = 0
        self.is_stable = False
        self.tick_since_last_check = 0

        self.pending_samples = []   # (fire_tick, x_vec, snapshot(food,pop,pred), wx, wy)
        self.x_buffer = []
        self.y_buffer = []
        self.total_saved = 0

    # ---------------- spawning ----------------
    def spawn_initial_population(self):
        for _ in range(INITIAL_SPECIES):
            template = Genome.random_new()
            for _ in range(INITIAL_PER_SPECIES):
                x, y = self.random_valid_spawn(template.aquatic)
                g = template.mutated_copy(new_species=False)
                self.creatures.append(Creature(x, y, g))

    def random_valid_spawn(self, aquatic):
        for _ in range(50):
            x = random.uniform(0, WORLD_W - 1)
            y = random.uniform(0, WORLD_H - 1)
            cell, c, r = self.world.cell_at(x, y)
            if cell.is_water == aquatic:
                return x, y
        return random.uniform(0, WORLD_W - 1), random.uniform(0, WORLD_H - 1)

    # ---------------- main tick ----------------
    def step(self):
        w = self.world
        w.tick += 1
        w.update_plants()

        spatial = SpatialHash(GRID_SIZE)
        for cr in self.creatures:
            if cr.alive:
                spatial.insert(cr, cr.x, cr.y)

        for cr in self.creatures:
            if cr.alive:
                self.update_creature(cr, spatial)

        self.creatures = [c for c in self.creatures if c.alive]
        if len(self.creatures) > MAX_CREATURES:
            random.shuffle(self.creatures)
            self.creatures = self.creatures[:MAX_CREATURES]

        self.update_eggs()
        self.update_equilibrium()
        self.update_dataset_recording()

    # ---------------- creature AI (basit, deterministik/rastgele kural tabanlı) ----------------
    def update_creature(self, cr: Creature, spatial: SpatialHash):
        g = cr.genome
        cr.age += 1
        cr.energy -= 0.045 * (0.6 + g.size) * (1.0 + g.legs * 0.02)
        cr.thirst += 0.05

        if cr.age >= g.max_age() or cr.energy <= 0 or cr.thirst >= 100:
            cr.alive = False
            return

        nearby = spatial.query_radius(cr.x, cr.y, g.sight_range)
        predators, prey, mates, food_dir = [], [], [], None

        for other in nearby:
            if other is cr or not other.alive:
                continue
            d = math.hypot(other.x - cr.x, other.y - cr.y)
            if d > g.sight_range:
                continue
            is_predator_of_me = (
                other.genome.diet in (Diet.CARNIVORE, Diet.OMNIVORE)
                and other.genome.size > g.size * 1.15
                and g.diet != Diet.CARNIVORE
            )
            is_my_prey = (
                g.diet in (Diet.CARNIVORE, Diet.OMNIVORE)
                and other.genome.size < g.size * 0.9
                and other.genome.species_id != g.species_id
            )
            if is_predator_of_me:
                predators.append((other, d))
            if is_my_prey:
                prey.append((other, d))
            if other.genome.species_id == g.species_id and other is not cr:
                mates.append((other, d))

        cell, cc, cr_row = self.world.cell_at(cr.x, cr.y)
        target_dx, target_dy = 0.0, 0.0
        speed = g.speed()

        if predators:
            predators.sort(key=lambda t: t[1])
            p, _ = predators[0]
            target_dx, target_dy = cr.x - p.x, cr.y - p.y
            cr.state = "flee"
        elif cr.thirst > 55:
            wx, wy = self.nearest_water(cr.x, cr.y, g.sight_range)
            if wx is not None:
                target_dx, target_dy = wx - cr.x, wy - cr.y
                cr.state = "seek_water"
            else:
                target_dx, target_dy = random.uniform(-1, 1), random.uniform(-1, 1)
                cr.state = "wander"
        elif cr.energy_ratio() < 0.6:
            if g.diet in (Diet.CARNIVORE, Diet.OMNIVORE) and prey:
                prey.sort(key=lambda t: t[1])
                target, dist = prey[0]
                target_dx, target_dy = target.x - cr.x, target.y - cr.y
                cr.state = "hunt"
                if dist < 14:
                    target.alive = False
                    cr.energy = min(g.max_energy(), cr.energy + target.genome.size * 55)
            elif g.diet in (Diet.HERBIVORE, Diet.OMNIVORE):
                fx, fy = self.nearest_food(cr.x, cr.y, g.sight_range)
                if fx is not None:
                    target_dx, target_dy = fx - cr.x, fy - cr.y
                    cr.state = "seek_food"
                    if math.hypot(fx - cr.x, fy - cr.y) < 30:
                        fcell, fc, frow = self.world.cell_at(fx, fy)
                        if fcell.plants > 0:
                            fcell.plants -= 1
                            cr.energy = min(g.max_energy(), cr.energy + 35)
                else:
                    target_dx, target_dy = random.uniform(-1, 1), random.uniform(-1, 1)
                    cr.state = "wander"
            else:
                target_dx, target_dy = random.uniform(-1, 1), random.uniform(-1, 1)
                cr.state = "wander"
        elif cr.repro_cooldown <= 0 and cr.energy_ratio() > 0.65:
            cr.state = "seek_mate"
            if g.repro_mode == ReproMode.ASEXUAL:
                self.reproduce_asexual(cr)
                cr.repro_cooldown = 260 * (1.4 - g.repro_rate)
            else:
                if mates:
                    mates.sort(key=lambda t: t[1])
                    mate, dist = mates[0]
                    if dist < 20 and mate.repro_cooldown <= 0:
                        self.reproduce_sexual(cr, mate)
                        cr.repro_cooldown = 260 * (1.4 - g.repro_rate)
                        mate.repro_cooldown = 260 * (1.4 - mate.genome.repro_rate)
                    else:
                        target_dx, target_dy = mate.x - cr.x, mate.y - cr.y
                else:
                    target_dx, target_dy = random.uniform(-1, 1), random.uniform(-1, 1)
        else:
            if random.random() < 0.03:
                cr.vx = random.uniform(-1, 1)
                cr.vy = random.uniform(-1, 1)
            target_dx, target_dy = cr.vx, cr.vy
            cr.state = "wander"

        if cr.repro_cooldown > 0:
            cr.repro_cooldown -= 1

        # zeka düşükse hareket daha rastgele (kararı bazen görmezden gelir)
        if random.random() > g.intelligence and cr.state not in ("flee",):
            target_dx += random.uniform(-1, 1)
            target_dy += random.uniform(-1, 1)

        dist = math.hypot(target_dx, target_dy)
        if dist > 0.001:
            target_dx, target_dy = target_dx / dist, target_dy / dist

        step = speed / FPS
        nx = cr.x + target_dx * step
        ny = cr.y + target_dy * step
        nx = min(max(nx, 0), WORLD_W - 1)
        ny = min(max(ny, 0), WORLD_H - 1)

        ncell, _, _ = self.world.cell_at(nx, ny)
        if ncell.is_water == g.aquatic:
            cr.x, cr.y = nx, ny

        if ncell.is_water and not g.aquatic:
            cr.thirst = max(0, cr.thirst - 3)

    def nearest_food(self, x, y, radius):
        cell_r = int(radius // GRID_SIZE) + 1
        cc, rr = int(x // GRID_SIZE), int(y // GRID_SIZE)
        best, best_d = None, radius + 1
        for dc in range(-cell_r, cell_r + 1):
            for dr in range(-cell_r, cell_r + 1):
                c, r = cc + dc, rr + dr
                if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
                    cell = self.world.grid[r][c]
                    if not cell.is_water and cell.plants > 0:
                        px, py = c * GRID_SIZE + GRID_SIZE / 2, r * GRID_SIZE + GRID_SIZE / 2
                        d = math.hypot(px - x, py - y)
                        if d < best_d:
                            best_d, best = d, (px, py)
        return best if best else (None, None)

    def nearest_water(self, x, y, radius):
        cell_r = int(radius // GRID_SIZE) + 1
        cc, rr = int(x // GRID_SIZE), int(y // GRID_SIZE)
        best, best_d = None, radius + 1
        for dc in range(-cell_r, cell_r + 1):
            for dr in range(-cell_r, cell_r + 1):
                c, r = cc + dc, rr + dr
                if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
                    cell = self.world.grid[r][c]
                    if cell.is_water:
                        px, py = c * GRID_SIZE + GRID_SIZE / 2, r * GRID_SIZE + GRID_SIZE / 2
                        d = math.hypot(px - x, py - y)
                        if d < best_d:
                            best_d, best = d, (px, py)
        return best if best else (None, None)

    # ---------------- reproduction ----------------
    def reproduce_asexual(self, cr: Creature):
        if len(self.creatures) + len(self.eggs) >= MAX_CREATURES:
            return
        cost = cr.genome.max_energy() * 0.35
        if cr.energy < cost:
            return
        cr.energy -= cost
        child_genome = cr.genome.mutated_copy(new_species=random.random() < 0.01)
        self.birth(cr.x, cr.y, child_genome)

    def reproduce_sexual(self, a: Creature, b: Creature):
        if len(self.creatures) + len(self.eggs) >= MAX_CREATURES:
            return
        cost = a.genome.max_energy() * 0.3
        if a.energy < cost or b.energy < cost:
            return
        a.energy -= cost
        b.energy -= cost
        child_genome = Genome.crossover(a.genome, b.genome)
        self.birth(a.x, a.y, child_genome)

    def birth(self, x, y, genome: Genome):
        if genome.birth_mode == BirthMode.MAMMAL:
            self.creatures.append(Creature(x + random.uniform(-10, 10), y + random.uniform(-10, 10), genome))
        else:
            self.eggs.append(Egg(x + random.uniform(-10, 10), y + random.uniform(-10, 10), genome))

    def update_eggs(self):
        remaining = []
        for egg in self.eggs:
            egg.incubation -= 1
            eaten = False
            for cr in self.creatures:
                if cr.alive and cr.genome.diet in (Diet.CARNIVORE, Diet.OMNIVORE):
                    if math.hypot(cr.x - egg.x, cr.y - egg.y) < 12:
                        cr.energy = min(cr.genome.max_energy(), cr.energy + 20)
                        eaten = True
                        break
            if eaten:
                continue
            if egg.incubation <= 0:
                if len(self.creatures) < MAX_CREATURES:
                    self.creatures.append(Creature(egg.x, egg.y, egg.genome))
            else:
                remaining.append(egg)
        self.eggs = remaining

    # ---------------- equilibrium detection ----------------
    def update_equilibrium(self):
        total_food = sum(cell.plants for row in self.world.grid for cell in row)
        self.pop_history.append(len(self.creatures))
        self.food_history.append(total_food)
        self.tick_since_last_check += 1

        if self.tick_since_last_check >= EQ_WINDOW and len(self.pop_history) == EQ_WINDOW:
            self.tick_since_last_check = 0
            pop_mean = sum(self.pop_history) / EQ_WINDOW
            food_mean = sum(self.food_history) / EQ_WINDOW
            pop_std = (sum((p - pop_mean) ** 2 for p in self.pop_history) / EQ_WINDOW) ** 0.5
            food_std = (sum((f - food_mean) ** 2 for f in self.food_history) / EQ_WINDOW) ** 0.5
            pop_cv = pop_std / pop_mean if pop_mean > 1 else 1.0
            food_cv = food_std / food_mean if food_mean > 1 else 1.0

            if pop_mean > 5 and pop_cv < EQ_CV_THRESHOLD and food_cv < EQ_CV_THRESHOLD:
                self.stable_windows += 1
            else:
                self.stable_windows = 0

            if self.stable_windows >= EQ_HISTORY_LEN:
                self.is_stable = True

    # ---------------- dataset recording ----------------
    def region_counts(self, wx, wy, radius):
        food = 0
        cell_r = int(radius // GRID_SIZE) + 1
        cc, rr = int(wx // GRID_SIZE), int(wy // GRID_SIZE)
        for dc in range(-cell_r, cell_r + 1):
            for dr in range(-cell_r, cell_r + 1):
                c, r = cc + dc, rr + dr
                if 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS:
                    px, py = c * GRID_SIZE + GRID_SIZE / 2, r * GRID_SIZE + GRID_SIZE / 2
                    if math.hypot(px - wx, py - wy) <= radius:
                        food += self.world.grid[r][c].plants

        pop, pred = 0, 0
        for cr in self.creatures:
            if not cr.alive:
                continue
            if math.hypot(cr.x - wx, cr.y - wy) <= radius:
                pop += 1
                if cr.genome.diet in (Diet.CARNIVORE, Diet.OMNIVORE):
                    pred += 1
        return food, pop, pred

    def build_x_vector(self, wx, wy):
        cell, c, r = self.world.cell_at(wx, wy)
        temperature = self.world.temperature_at(c, r)
        humidity = self.world.humidity_at(c, r)
        food_density = clamp01(cell.plants / MAX_PLANTS_PER_CELL)
        water_density = 1.0 if cell.is_water else 0.0
        light = self.world.light()
        season = self.world.season_index() / 3.0

        food, pop, pred = self.region_counts(wx, wy, SENSE_RADIUS)
        nearby_food = clamp01(food / 30.0)
        nearby_water = 1.0 if water_density == 1.0 else clamp01(
            1.0 - (self.nearest_water_dist(wx, wy, SENSE_RADIUS) / SENSE_RADIUS)
        )
        nearby_creatures = clamp01(pop / 20.0)
        nearby_predators = clamp01(pred / 10.0)
        danger = clamp01(0.7 * nearby_predators + 0.3 * (1.0 - nearby_food))

        world_reserved = 0.0  # bkz. dosya başındaki not (3)
        x_vec = [
            world_reserved, temperature, humidity, food_density, water_density,
            light, season, nearby_food, nearby_water, nearby_creatures,
            nearby_predators, danger,
        ]
        return x_vec, (food, pop, pred)

    def nearest_water_dist(self, x, y, radius):
        wx, wy = self.nearest_water(x, y, radius)
        if wx is None:
            return radius
        return math.hypot(wx - x, wy - y)

    def update_dataset_recording(self):
        if not self.is_stable:
            return

        if self.world.tick % RECORD_INTERVAL == 0:
            for _ in range(SAMPLES_PER_RECORD):
                wx = random.uniform(0, WORLD_W - 1)
                wy = random.uniform(0, WORLD_H - 1)
                x_vec, snapshot = self.build_x_vector(wx, wy)
                fire_tick = self.world.tick + RECORD_DELTA_TICKS
                self.pending_samples.append((fire_tick, x_vec, snapshot, wx, wy))

        still_pending = []
        for (fire_tick, x_vec, snapshot, wx, wy) in self.pending_samples:
            if self.world.tick >= fire_tick:
                food0, pop0, pred0 = snapshot
                food1, pop1, pred1 = self.region_counts(wx, wy, SENSE_RADIUS)
                food_delta = clamp01((food1 - food0) / 15.0 + 0.5) * 2 - 1
                pop_delta = clamp01((pop1 - pop0) / 8.0 + 0.5) * 2 - 1
                pred_delta = clamp01((pred1 - pred0) / 5.0 + 0.5) * 2 - 1
                self.x_buffer.append(x_vec)
                self.y_buffer.append([food_delta, pop_delta, pred_delta])
            else:
                still_pending.append((fire_tick, x_vec, snapshot, wx, wy))
        self.pending_samples = still_pending

        if len(self.x_buffer) >= FLUSH_EVERY_SAMPLES:
            self.flush_dataset()

    def flush_dataset(self):
        if not self.x_buffer:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        new_x = torch.tensor(self.x_buffer, dtype=torch.float32)
        new_y = torch.tensor(self.y_buffer, dtype=torch.float32)

        if os.path.exists(X_PATH):
            old_x = torch.load(X_PATH)
            new_x = torch.cat([old_x, new_x], dim=0)
        if os.path.exists(Y_PATH):
            old_y = torch.load(Y_PATH)
            new_y = torch.cat([old_y, new_y], dim=0)

        torch.save(new_x, X_PATH)
        torch.save(new_y, Y_PATH)
        self.total_saved = new_x.shape[0]
        self.x_buffer.clear()
        self.y_buffer.clear()


# =====================================================================
# RENDERING
# =====================================================================
def biome_color(temp, humid, is_water):
    if is_water:
        return (40, 90, 170)
    r = int(90 + temp * 120 - humid * 30)
    g = int(80 + humid * 140)
    b = int(60 + (1 - humid) * 40)
    return (max(20, min(255, r)), max(20, min(255, g)), max(20, min(255, b)))


def build_background_surface(world: World):
    surf = pygame.Surface((WORLD_W, WORLD_H))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = world.grid[r][c]
            color = biome_color(cell.base_temp, cell.base_humid, cell.is_water)
            pygame.draw.rect(surf, color, (c * GRID_SIZE, r * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    return surf


def draw_plants(surf, world: World, cam_x, cam_y, screen_w, screen_h):
    c0 = max(0, int(cam_x // GRID_SIZE))
    r0 = max(0, int(cam_y // GRID_SIZE))
    c1 = min(GRID_COLS, int((cam_x + screen_w) // GRID_SIZE) + 1)
    r1 = min(GRID_ROWS, int((cam_y + screen_h) // GRID_SIZE) + 1)
    for r in range(r0, r1):
        for c in range(c0, c1):
            cell = world.grid[r][c]
            if cell.is_water or cell.plants <= 0:
                continue
            base_x = c * GRID_SIZE - cam_x
            base_y = r * GRID_SIZE - cam_y
            for i in range(cell.plants):
                px = base_x + 10 + (i % 3) * 28
                py = base_y + 10 + (i // 3) * 28
                pygame.draw.circle(surf, (40, 170, 60), (int(px), int(py)), 5)


DIET_OUTLINE = {Diet.HERBIVORE: (60, 220, 90), Diet.CARNIVORE: (230, 60, 60), Diet.OMNIVORE: (230, 200, 50)}


def draw_creature(surf, cr: Creature, cam_x, cam_y):
    x, y = cr.x - cam_x, cr.y - cam_y
    if x < -30 or y < -30 or x > surf.get_width() + 30 or y > surf.get_height() + 30:
        return
    radius = int(5 + cr.genome.size * 5)
    color = cr.genome.color
    outline = DIET_OUTLINE[cr.genome.diet]
    if cr.genome.aquatic:
        pygame.draw.ellipse(surf, color, (x - radius, y - radius * 0.7, radius * 2, radius * 1.4))
    else:
        pygame.draw.circle(surf, color, (int(x), int(y)), radius)
        legs = cr.genome.legs
        for i in range(legs):
            ang = (i / max(1, legs)) * 2 * math.pi
            lx = x + math.cos(ang) * (radius + 4)
            ly = y + math.sin(ang) * (radius + 4)
            pygame.draw.line(surf, color, (x, y), (lx, ly), 2)
    pygame.draw.circle(surf, outline, (int(x), int(y)), radius, 2)


def draw_egg(surf, egg: Egg, cam_x, cam_y):
    x, y = egg.x - cam_x, egg.y - cam_y
    pygame.draw.ellipse(surf, (240, 240, 210), (x - 5, y - 7, 10, 14))
    pygame.draw.ellipse(surf, (150, 150, 120), (x - 5, y - 7, 10, 14), 1)


# =====================================================================
# MAIN LOOP
# =====================================================================
def main():
    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Ekosistem Simülasyonu - Veri Toplama")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)
    big_font = pygame.font.SysFont("arial", 28, bold=True)

    sim = Simulation()
    background = build_background_surface(sim.world)

    cam_x, cam_y = WORLD_W / 2 - screen_w / 2, WORLD_H / 2 - screen_h / 2
    turbo = False
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    turbo = not turbo
                elif event.key == pygame.K_s:
                    sim.flush_dataset()
                elif event.key == pygame.K_r:
                    sim = Simulation()
                    background = build_background_surface(sim.world)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cam_x -= CAMERA_SPEED * dt
        if keys[pygame.K_RIGHT]:
            cam_x += CAMERA_SPEED * dt
        if keys[pygame.K_UP]:
            cam_y -= CAMERA_SPEED * dt
        if keys[pygame.K_DOWN]:
            cam_y += CAMERA_SPEED * dt
        cam_x = max(0, min(cam_x, WORLD_W - screen_w))
        cam_y = max(0, min(cam_y, WORLD_H - screen_h))

        steps = 4 if turbo else 1
        for _ in range(steps):
            sim.step()

        screen.blit(background, (-cam_x, -cam_y))
        draw_plants(screen, sim.world, cam_x, cam_y, screen_w, screen_h)
        for egg in sim.eggs:
            draw_egg(screen, egg, cam_x, cam_y)
        for cr in sim.creatures:
            draw_creature(screen, cr, cam_x, cam_y)

        # HUD
        season_name = SEASONS[sim.world.season_index()]
        hud_lines = [
            f"Tick: {sim.world.tick}   Mevsim: {season_name}   Işık: {sim.world.light():.2f}",
            f"Popülasyon: {len(sim.creatures)}   Yumurta: {len(sim.eggs)}   Toplam yiyecek: {sum(c.plants for row in sim.world.grid for c in row)}",
            f"Kayıtlı örnek sayısı: {sim.total_saved + len(sim.x_buffer)}   Bekleyen: {len(sim.pending_samples)}",
            f"Turbo: {'AÇIK' if turbo else 'KAPALI'} (SPACE)   Elle kaydet: S   Sıfırla: R   Çıkış: ESC",
        ]
        for i, line in enumerate(hud_lines):
            txt = font.render(line, True, (255, 255, 255))
            screen.blit(txt, (14, 10 + i * 22))

        status_text = "DENGE BULUNDU - VERİ KAYDI AÇIK" if sim.is_stable else f"DENGE BEKLENİYOR... ({sim.stable_windows}/{EQ_HISTORY_LEN})"
        status_color = (90, 230, 120) if sim.is_stable else (230, 200, 90)
        status_surf = big_font.render(status_text, True, status_color)
        screen.blit(status_surf, (14, screen_h - 40))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()