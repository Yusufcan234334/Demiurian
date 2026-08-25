#this code is human made.  AI only handled some of the difficult parts and cleaned up the code to make it readable.

import pygame
import random
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import torch
import torch.nn as siniragi

# =====================================================================
# TEMEL AYARLAR
# =====================================================================
SCALE = math.sqrt(10)  # toplam alan eskisinin ~10 katı olsun diye her eksen sqrt(10) büyütülüyor
BASE_W, BASE_H = 4800, 3200
WORLD_W, WORLD_H = int(BASE_W * SCALE), int(BASE_H * SCALE)
GRID_SIZE = 100
GRID_COLS = WORLD_W // GRID_SIZE
GRID_ROWS = WORLD_H // GRID_SIZE

FPS = 60
PAN_SPEED = 1400.0

DAY_LENGTH_TICKS = 480
SEASON_LENGTH_TICKS = 2400
YEAR_LENGTH_TICKS = SEASON_LENGTH_TICKS * 4
SEASONS = ["İlkbahar", "Yaz", "Sonbahar", "Kış"]
SEASON_TEMP_MOD = [0.0, 0.18, -0.05, -0.30]
SEASON_HUMID_MOD = [0.05, -0.10, 0.10, -0.05]
SEASON_GROWTH_MOD = [1.1, 1.3, 0.8, 0.35]

MAX_PLANTS_PER_CELL = 6
BASE_GROWTH_PROB = 0.010

MAX_CREATURES = 420
INITIAL_SPECIES = 9
INITIAL_PER_SPECIES = 10

SIGHT_MIN, SIGHT_MAX = 120, 420
LEG_OPTIONS = [0, 2, 4, 6, 8]
EYE_OPTIONS = [1, 2, 2, 2, 2, 3, 4, 6, 8]

SENSE_RADIUS = 260
WORLD_STATS_RECALC_INTERVAL = 240
POLLUTION_PER_CREATURE = 0.0000025
POLLUTION_DECAY = 0.0004

# --- TANRI / AI AYARLARI ---
GOD_DECISION_INTERVAL = 480       # her 480 tick'te bir karar
REGION_COLS, REGION_ROWS = 9, 9   # harita 9x9 makro bölgeye ayrılıyor
GOD_SPAWN_COUNT = 3                # her karar sonrası kaç canlı hediye edilecek
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demiurian.pth")

# --- KAMERA / ZOOM / LOD AYARLARI ---
ZOOM_MIN, ZOOM_MAX = 0.04, 3.0
ZOOM_STEP = 1.12
LOD_EMOJI_ZOOM = 0.35     # bunun altında canlılar tek piksel/kare olarak çizilir
LOD_PLANT_ZOOM = 0.55     # bunun altında tek tek bitki ikonu çizilmez
MINIMAP_W, MINIMAP_H = 230, 160
MINIMAP_MARGIN = 14

random.seed()


def clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


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


class Habitat(Enum):
    LAND = 0
    WATER = 1
    AMPHIBIOUS = 2
    UNDERGROUND = 3
    ARBOREAL = 4


class Terrain(Enum):
    WATER = 0
    CORAL_REEF = 1
    DESERT = 2
    VOLCANIC = 3
    PLAINS = 4
    SAVANNA = 5
    FOREST = 6
    JUNGLE = 7
    SWAMP = 8
    TAIGA = 9
    TUNDRA = 10
    MOUNTAIN = 11
    BEACH = 12


TERRAIN_BASE_COLOR = {
    Terrain.WATER: (40, 90, 170),
    Terrain.CORAL_REEF: (40, 150, 190),
    Terrain.DESERT: (222, 197, 120),
    Terrain.VOLCANIC: (120, 55, 40),
    Terrain.PLAINS: (150, 195, 90),
    Terrain.SAVANNA: (200, 180, 90),
    Terrain.FOREST: (55, 130, 65),
    Terrain.JUNGLE: (25, 110, 55),
    Terrain.SWAMP: (75, 100, 65),
    Terrain.TAIGA: (70, 120, 105),
    Terrain.TUNDRA: (210, 220, 225),
    Terrain.MOUNTAIN: (120, 115, 110),
    Terrain.BEACH: (230, 215, 170),
}


# =====================================================================
# DÜNYA / GRID
# =====================================================================
class Cell:
    __slots__ = (
        "is_water", "base_temp", "base_humid", "elevation", "capacity", "plants",
        "terrain", "vegetation_density", "oxygen", "pollution", "rainfall",
        "depth", "resource_regen_rate",
    )

    def __init__(self, is_water, base_temp, base_humid, elevation, capacity):
        self.is_water = is_water
        self.base_temp = base_temp
        self.base_humid = base_humid
        self.elevation = elevation
        self.capacity = capacity
        self.plants = 0 if is_water else random.randint(0, 2)
        self.terrain = classify_terrain(is_water, base_temp, base_humid, elevation)
        self.vegetation_density = 0.0 if is_water else clamp01(capacity)
        self.rainfall = clamp01(base_humid * 0.8 + random.uniform(-0.05, 0.05))
        self.depth = clamp01(0.3 + random.uniform(0, 0.7)) if is_water else 0.0
        self.oxygen = clamp01(0.55 + base_humid * 0.25 - (self.depth * 0.15 if is_water else 0.0))
        self.pollution = 0.0
        self.resource_regen_rate = clamp01(capacity)


def classify_terrain(is_water, temp, humid, elevation):
    if is_water:
        return Terrain.CORAL_REEF if temp > 0.62 and humid > 0.5 else Terrain.WATER
    if elevation > 0.82:
        return Terrain.VOLCANIC if temp > 0.7 else Terrain.MOUNTAIN
    if elevation < 0.16 and humid > 0.55:
        return Terrain.SWAMP
    if elevation < 0.12:
        return Terrain.BEACH
    if temp < 0.28:
        return Terrain.TUNDRA
    if temp < 0.42:
        return Terrain.TAIGA
    if humid > 0.72 and temp > 0.55:
        return Terrain.JUNGLE
    if humid > 0.55:
        return Terrain.FOREST
    if humid < 0.22 and temp > 0.55:
        return Terrain.DESERT
    if humid < 0.4:
        return Terrain.SAVANNA
    return Terrain.PLAINS


def value_noise(nx, ny, freq, seed_off):
    return (
        math.sin(nx * freq + seed_off) * math.cos(ny * freq * 0.8 - seed_off)
        + math.sin(nx * freq * 2.3 - seed_off * 1.7) * 0.5
        + math.cos(ny * freq * 1.9 + seed_off * 0.6) * 0.5
    ) / 2.0


def build_grid():
    grid = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    n_lakes = random.randint(4, 8)
    lake_centers = []
    for _ in range(n_lakes):
        lake_centers.append((
            random.uniform(0.06, 0.94) * GRID_COLS,
            random.uniform(0.06, 0.94) * GRID_ROWS,
            random.uniform(2.5, 7.0),
            random.uniform(2.0, 5.5),
        ))
    seed_off = random.uniform(0, 1000)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nx, ny = c / GRID_COLS, r / GRID_ROWS
            is_water = False
            for (lcx, lcy, rx, ry) in lake_centers:
                if ((c - lcx) / rx) ** 2 + ((r - lcy) / ry) ** 2 <= 1.0:
                    is_water = True
                    break
            base_temp = clamp01(0.55 + 0.35 * value_noise(nx, ny, 2.6, seed_off) - (ny - 0.5) * 0.35)
            base_humid = clamp01(0.5 + 0.4 * value_noise(nx, ny, 1.9, seed_off + 50))
            elevation = clamp01(0.5 + 0.5 * value_noise(nx, ny, 3.4, seed_off + 130))
            dist_water = min(
                math.hypot((c - lcx), (r - lcy)) / max(rx, ry) for (lcx, lcy, rx, ry) in lake_centers
            )
            base_humid = clamp01(base_humid + max(0.0, 0.22 - dist_water * 0.05))
            if is_water:
                elevation = 0.0
            capacity = clamp01(base_humid * 0.6 + (1 - abs(base_temp - 0.55)) * 0.4 - elevation * 0.25)
            grid[r][c] = Cell(is_water, base_temp, base_humid, elevation, capacity)
    return grid


class World:
    def __init__(self):
        self.grid = build_grid()
        self.tick = 0
        self.average_temperature = 0.5
        self.temperature_variance = 0.0
        self.average_pollution = 0.0
        self.available_niches = 1.0
        self._recalc_world_stats()

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
        return clamp01(self.grid[r][c].base_temp + SEASON_TEMP_MOD[self.season_index()])

    def humidity_at(self, c, r):
        return clamp01(self.grid[r][c].base_humid + SEASON_HUMID_MOD[self.season_index()])

    def water_temperature_at(self, c, r):
        cell = self.grid[r][c]
        base = clamp01(cell.base_temp + SEASON_TEMP_MOD[self.season_index()] * 0.4)
        return clamp01(base - cell.depth * 0.2)

    def update_plants(self):
        s = self.season_index()
        growth_mod = SEASON_GROWTH_MOD[s]
        for r in range(GRID_ROWS):
            row = self.grid[r]
            for c in range(GRID_COLS):
                cell = row[c]
                if cell.is_water:
                    continue
                regen = clamp01(cell.capacity * growth_mod * (1.0 - cell.pollution * 0.6))
                cell.resource_regen_rate = regen
                cap = int(cell.capacity * MAX_PLANTS_PER_CELL)
                if cell.plants < cap:
                    prob = BASE_GROWTH_PROB * regen * (0.4 + self.humidity_at(c, r))
                    if random.random() < prob:
                        cell.plants += 1

    def update_pollution(self, population):
        add = population * POLLUTION_PER_CREATURE
        for row in self.grid:
            for cell in row:
                cell.pollution = clamp01(cell.pollution + add - POLLUTION_DECAY)
                cell.oxygen = clamp01(cell.oxygen - add * 0.5 + POLLUTION_DECAY * 0.3)

    def _recalc_world_stats(self):
        temps = [cell.base_temp for row in self.grid for cell in row]
        mean = sum(temps) / len(temps)
        var = sum((t - mean) ** 2 for t in temps) / len(temps)
        self.average_temperature = mean
        self.temperature_variance = var
        pollutions = [cell.pollution for row in self.grid for cell in row]
        self.average_pollution = sum(pollutions) / len(pollutions)

    def maybe_recalc_stats(self):
        if self.tick % WORLD_STATS_RECALC_INTERVAL == 0:
            self._recalc_world_stats()


# =====================================================================
# GENOME (x_vector / y_vector formatı orijinal eğitim koduyla AYNI)
# =====================================================================
_next_species_id = [0]


def new_species_id():
    _next_species_id[0] += 1
    return _next_species_id[0]


def make_eye_positions(count):
    positions = []
    if count <= 2:
        spread = 25
        base_angles = [-spread, spread][:count] if count == 2 else [0]
        for a in base_angles:
            positions.append((a + random.uniform(-4, 4), random.uniform(0.55, 0.85)))
    else:
        for i in range(count):
            angle = (360 / count) * i + random.uniform(-8, 8)
            positions.append((angle, random.uniform(0.5, 0.9)))
    return positions


def make_leg_positions(count):
    if count == 0:
        return []
    return [(360 / count) * i + random.uniform(-6, 6) for i in range(count)]


@dataclass
class Genome:
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
    habitat: Habitat
    is_gods_creation: bool = False

    weight: float = 1.0
    strength: float = 0.5
    hearing_range: float = 200.0
    smell_range: float = 150.0
    swim_ability: float = 0.2
    fly_ability: float = 0.0
    armor: float = 0.1
    camouflage: float = 0.2
    poison: float = 0.0
    attack_power: float = 0.4
    defense: float = 0.4
    energy_consumption_mult: float = 1.0

    eye_count: int = 2
    eye_positions: list = field(default_factory=list)
    leg_positions: list = field(default_factory=list)

    aggression: float = 0.4
    curiosity: float = 0.4
    sociality: float = 0.4
    fear: float = 0.4
    territoriality: float = 0.3
    exploration: float = 0.4
    memory: float = 0.4
    learning_ability: float = 0.4
    problem_solving: float = 0.4

    maturity_age: float = 200.0
    offspring_count: int = 1
    parental_care: float = 0.3
    mutation_rate: float = 0.06

    def to_y_vector(self):
        aquatic = 1.0 if self.habitat in (Habitat.WATER, Habitat.AMPHIBIOUS) else 0.0
        diet_h = 1.0 if self.diet == Diet.HERBIVORE else 0.0
        diet_c = 1.0 if self.diet == Diet.CARNIVORE else 0.0
        diet_o = 1.0 if self.diet == Diet.OMNIVORE else 0.0
        legs_norm = self.legs / 8.0
        size_norm = clamp01(self.size / 2.5)
        speed_norm = clamp01(self.speed() / 150.0)
        vision_norm = clamp01(self.sight_range / SIGHT_MAX)
        return [aquatic, diet_h, diet_c, diet_o, legs_norm, size_norm, speed_norm,
                vision_norm, self.intelligence, self.strength, self.armor,
                self.camouflage, self.aggression]

    def can_occupy(self, cell):
        if self.habitat == Habitat.WATER:
            return cell.is_water
        if self.habitat == Habitat.AMPHIBIOUS:
            return True
        return not cell.is_water

    def speed(self):
        base = 40 + self.legs * 9
        if self.habitat in (Habitat.WATER, Habitat.AMPHIBIOUS):
            base += 25 * self.swim_ability
        if self.habitat == Habitat.ARBOREAL:
            base += 10
        base += self.fly_ability * 40
        return base / (0.6 + self.size)

    def max_energy(self):
        return 60 + self.size * 90

    def max_age(self):
        return 1400 + self.intelligence * 900 + self.size * 400

    def mutated_copy(self, new_species=False):
        def jitter(v, amt):
            return v + random.uniform(-amt, amt)

        m = self.mutation_rate
        new_habitat = self.habitat
        if random.random() < 0.02:
            new_habitat = random.choice(list(Habitat))
        new_legs = max(0, min(8, self.legs + random.choice([-2, 0, 0, 0, 2])))
        new_eyes = max(1, min(8, self.eye_count + random.choice([-1, 0, 0, 0, 1])))

        return Genome(
            legs=new_legs,
            sight_range=clamp01(jitter(self.sight_range / SIGHT_MAX, 0.06 * (0.5 + m))) * SIGHT_MAX,
            repro_rate=clamp01(jitter(self.repro_rate, 0.05)),
            repro_mode=self.repro_mode if random.random() > 0.02 else random.choice(list(ReproMode)),
            birth_mode=self.birth_mode if random.random() > 0.02 else random.choice(list(BirthMode)),
            intelligence=clamp01(jitter(self.intelligence, 0.05)),
            diet=self.diet if random.random() > 0.02 else random.choice(list(Diet)),
            size=max(0.3, min(2.2, jitter(self.size, 0.08))),
            species_id=new_species_id() if new_species else self.species_id,
            color=self.color, habitat=new_habitat, is_gods_creation=self.is_gods_creation,
            weight=max(0.1, jitter(self.weight, 0.1)), strength=clamp01(jitter(self.strength, 0.05)),
            hearing_range=max(50, jitter(self.hearing_range, 15)), smell_range=max(30, jitter(self.smell_range, 15)),
            swim_ability=clamp01(jitter(self.swim_ability, 0.06)),
            fly_ability=clamp01(jitter(self.fly_ability, 0.03)) if random.random() > 0.05 else self.fly_ability,
            armor=clamp01(jitter(self.armor, 0.05)), camouflage=clamp01(jitter(self.camouflage, 0.05)),
            poison=clamp01(jitter(self.poison, 0.04)), attack_power=clamp01(jitter(self.attack_power, 0.05)),
            defense=clamp01(jitter(self.defense, 0.05)),
            energy_consumption_mult=max(0.5, jitter(self.energy_consumption_mult, 0.05)),
            eye_count=new_eyes, eye_positions=make_eye_positions(new_eyes), leg_positions=make_leg_positions(new_legs),
            aggression=clamp01(jitter(self.aggression, 0.05)), curiosity=clamp01(jitter(self.curiosity, 0.05)),
            sociality=clamp01(jitter(self.sociality, 0.05)), fear=clamp01(jitter(self.fear, 0.05)),
            territoriality=clamp01(jitter(self.territoriality, 0.05)), exploration=clamp01(jitter(self.exploration, 0.05)),
            memory=clamp01(jitter(self.memory, 0.05)), learning_ability=clamp01(jitter(self.learning_ability, 0.05)),
            problem_solving=clamp01(jitter(self.problem_solving, 0.05)), maturity_age=max(30, jitter(self.maturity_age, 20)),
            offspring_count=max(1, min(12, self.offspring_count + random.choice([-1, 0, 0, 0, 1]))),
            parental_care=clamp01(jitter(self.parental_care, 0.05)), mutation_rate=clamp01(jitter(self.mutation_rate, 0.02)),
        )

    @staticmethod
    def random_new():
        habitat = random.choice(list(Habitat))
        aquatic = habitat in (Habitat.WATER, Habitat.AMPHIBIOUS)
        diet = random.choice(list(Diet))
        color = (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
        legs = 0 if habitat == Habitat.WATER else random.choice(LEG_OPTIONS)
        eyes = random.choice(EYE_OPTIONS)
        sid = new_species_id()
        size = random.uniform(0.5, 1.8)
        return Genome(
            legs=legs, sight_range=random.uniform(SIGHT_MIN, SIGHT_MAX), repro_rate=random.uniform(0.2, 0.9),
            repro_mode=random.choice(list(ReproMode)), birth_mode=random.choice(list(BirthMode)),
            intelligence=random.uniform(0.1, 0.95), diet=diet, size=size, species_id=sid, color=color,
            habitat=habitat, weight=size * random.uniform(0.7, 1.4), strength=random.uniform(0.2, 0.9),
            hearing_range=random.uniform(100, 350), smell_range=random.uniform(80, 300),
            swim_ability=random.uniform(0.6, 1.0) if aquatic else random.uniform(0.0, 0.4),
            fly_ability=random.uniform(0.0, 0.15) if random.random() < 0.08 else 0.0,
            armor=random.uniform(0.0, 0.6), camouflage=random.uniform(0.1, 0.7),
            poison=random.uniform(0.0, 0.4) if random.random() < 0.15 else 0.0,
            attack_power=random.uniform(0.1, 0.9), defense=random.uniform(0.1, 0.9),
            energy_consumption_mult=random.uniform(0.8, 1.3), eye_count=eyes,
            eye_positions=make_eye_positions(eyes), leg_positions=make_leg_positions(legs),
            aggression=random.uniform(0.1, 0.9), curiosity=random.uniform(0.1, 0.9), sociality=random.uniform(0.1, 0.9),
            fear=random.uniform(0.1, 0.9), territoriality=random.uniform(0.0, 0.8), exploration=random.uniform(0.1, 0.9),
            memory=random.uniform(0.1, 0.9), learning_ability=random.uniform(0.1, 0.9), problem_solving=random.uniform(0.1, 0.9),
            maturity_age=random.uniform(80, 400), offspring_count=random.choice([1, 1, 2, 2, 3, 4]),
            parental_care=random.uniform(0.0, 0.8), mutation_rate=random.uniform(0.02, 0.15),
        )

    @staticmethod
    def crossover(a: "Genome", b: "Genome"):
        pick = lambda x, y: x if random.random() < 0.5 else y
        avg = lambda x, y: (x + y) / 2
        legs = pick(a.legs, b.legs)
        eyes = pick(a.eye_count, b.eye_count)
        g = Genome(
            legs=legs, sight_range=avg(a.sight_range, b.sight_range), repro_rate=avg(a.repro_rate, b.repro_rate),
            repro_mode=pick(a.repro_mode, b.repro_mode), birth_mode=pick(a.birth_mode, b.birth_mode),
            intelligence=avg(a.intelligence, b.intelligence), diet=pick(a.diet, b.diet), size=avg(a.size, b.size),
            species_id=a.species_id, color=pick(a.color, b.color), habitat=pick(a.habitat, b.habitat),
            weight=avg(a.weight, b.weight), strength=avg(a.strength, b.strength),
            hearing_range=avg(a.hearing_range, b.hearing_range), smell_range=avg(a.smell_range, b.smell_range),
            swim_ability=avg(a.swim_ability, b.swim_ability), fly_ability=avg(a.fly_ability, b.fly_ability),
            armor=avg(a.armor, b.armor), camouflage=avg(a.camouflage, b.camouflage), poison=avg(a.poison, b.poison),
            attack_power=avg(a.attack_power, b.attack_power), defense=avg(a.defense, b.defense),
            energy_consumption_mult=avg(a.energy_consumption_mult, b.energy_consumption_mult),
            eye_count=eyes, eye_positions=make_eye_positions(eyes), leg_positions=make_leg_positions(legs),
            aggression=avg(a.aggression, b.aggression), curiosity=avg(a.curiosity, b.curiosity),
            sociality=avg(a.sociality, b.sociality), fear=avg(a.fear, b.fear),
            territoriality=avg(a.territoriality, b.territoriality), exploration=avg(a.exploration, b.exploration),
            memory=avg(a.memory, b.memory), learning_ability=avg(a.learning_ability, b.learning_ability),
            problem_solving=avg(a.problem_solving, b.problem_solving), maturity_age=avg(a.maturity_age, b.maturity_age),
            offspring_count=pick(a.offspring_count, b.offspring_count), parental_care=avg(a.parental_care, b.parental_care),
            mutation_rate=avg(a.mutation_rate, b.mutation_rate),
        )
        return g.mutated_copy(new_species=False)


# =====================================================================
# CANLILAR / YUMURTA / SPATIAL HASH
# =====================================================================
class Egg:
    __slots__ = ("x", "y", "genome", "incubation", "max_incubation")

    def __init__(self, x, y, genome, incubation=260):
        self.x, self.y = x, y
        self.genome = genome
        self.incubation = incubation
        self.max_incubation = incubation


class Creature:
    __slots__ = ("x", "y", "genome", "energy", "thirst", "age", "alive", "repro_cooldown", "vx", "vy", "state")

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

    def is_mature(self):
        return self.age >= self.genome.maturity_age


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
# TANRI MODELİ (eğitim kodundakiyle birebir aynı mimari)
# =====================================================================
class muhtisimmodel(siniragi.Module):
    def __init__(self, giris, genislemecikis, katmansayisi, branchsayisi, cikis):
        super().__init__()
        self.branchler = siniragi.ModuleList()
        self.fusion_weights = siniragi.Parameter(torch.ones(branchsayisi))
        self.residual_weight = siniragi.Parameter(torch.tensor(1.0))

        neuroncountoutson = genislemecikis
        for branch in range(branchsayisi):
            katmanlar = siniragi.ModuleList()
            neuroncountin = giris
            neuroncountout = genislemecikis
            for i in range(katmansayisi):
                katmanlar.append(siniragi.Linear(neuroncountin, neuroncountout))
                if i == katmansayisi - 1:
                    break
                neuroncountin = neuroncountout
                neuroncountout = neuroncountout * 2
            for i in range(katmansayisi):
                if i == katmansayisi - 1:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2
                    neuroncountoutson = neuroncountout
                else:
                    neuroncountin = neuroncountout
                    neuroncountout //= 2
                katmanlar.append(siniragi.Linear(neuroncountin, neuroncountout))
            self.branchler.append(katmanlar)

        heads = 4
        while neuroncountoutson % heads != 0 and heads > 1:
            heads //= 2
        self.attention = siniragi.MultiheadAttention(embed_dim=neuroncountoutson, num_heads=heads, batch_first=True)
        self.output1 = siniragi.Linear(neuroncountoutson, cikis)

    def forward(self, x):
        ilkx = x
        branchciktilari = []
        for katmanlar in self.branchler:
            x = ilkx
            for katmannum, katman in enumerate(katmanlar):
                x = katman(x)
                if katmannum != len(katmanlar) - 1:
                    x = torch.relu(x)
            branchciktilari.append(x)
        branchciktilari = torch.stack(branchciktilari, dim=1)
        attended, _ = self.attention(branchciktilari, branchciktilari, branchciktilari)
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = attended + self.residual_weight * branchciktilari
        fused = fused * weights.view(1, -1, 1)
        fused = fused.sum(dim=1)
        return self.output1(fused)


class AIGod:
    """Haritayı 9x9 makro bölgeye ayırır, her GOD_DECISION_INTERVAL tick'te
    sıradaki bölgeye bakar, modele o bölgenin ortalama koşullarını (x_vector)
    verir ve dönen y_vector'ü yeni bir canlı tasarımına çevirip o bölgeye
    'hediye eder'."""

    def __init__(self, sim):
        self.sim = sim
        self.model = muhtisimmodel(22, 44, 2, 3, 13)
        self.loaded = False
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location="cpu")
                self.model.load_state_dict(state)
                self.loaded = True
                print(f"[Tanrı] Eğitilmiş model yüklendi: {MODEL_PATH}")
            except Exception as e:
                print(f"[Tanrı] Model yüklenemedi ({e}), rastgele (eğitilmemiş) ağırlıklarla devam.")
        else:
            print("[Tanrı] demiurian.pth bulunamadı — Tanrı şu an kaotik/eğitilmemiş davranıyor.")
        self.model.eval()

        self.regions = []
        rw = GRID_COLS / REGION_COLS
        rh = GRID_ROWS / REGION_ROWS
        for ry in range(REGION_ROWS):
            for rx in range(REGION_COLS):
                c0, c1 = int(rx * rw), int((rx + 1) * rw)
                r0, r1 = int(ry * rh), int((ry + 1) * rh)
                self.regions.append((max(c0, 0), max(r0, 0), min(c1, GRID_COLS), min(r1, GRID_ROWS)))
        self.turn_index = 0
        self.last_decision_text = "Tanrı henüz karar vermedi..."
        self.active_region = self.regions[0]
        self.last_decision_tick = -9999

    def maybe_decide(self):
        w = self.sim.world
        if w.tick % GOD_DECISION_INTERVAL != 0:
            return
        if w.tick == self.last_decision_tick:
            return
        self.last_decision_tick = w.tick

        region = self.regions[self.turn_index % len(self.regions)]
        self.active_region = region
        self.turn_index += 1

        x_vec = self._region_x_vector(region)
        with torch.no_grad():
            raw = self.model(torch.tensor([x_vec], dtype=torch.float32))
            y_vec = torch.sigmoid(raw)[0].tolist()

        genome = self._genome_from_y(y_vec)
        spawned = 0
        for _ in range(GOD_SPAWN_COUNT):
            pos = self._valid_spawn_in_region(genome, region)
            if pos is None:
                continue
            x, y = pos
            child = genome.mutated_copy(new_species=False)
            self.sim.creatures.append(Creature(x, y, child))
            spawned += 1

        c0, r0, c1, r1 = region
        self.last_decision_text = (
            f"Bölge ({c0//max(1,int(GRID_COLS/REGION_COLS))},{r0//max(1,int(GRID_ROWS/REGION_ROWS))}) -> "
            f"{genome.habitat.name}/{genome.diet.name}, {spawned} canlı hediye edildi"
        )

    def _region_x_vector(self, region):
        c0, r0, c1, r1 = region
        c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
        samples = []
        for _ in range(6):
            c = random.randint(c0, c1 - 1)
            r = random.randint(r0, r1 - 1)
            wx = c * GRID_SIZE + GRID_SIZE / 2
            wy = r * GRID_SIZE + GRID_SIZE / 2
            samples.append(self.sim.build_x_vector(wx, wy))
        avg = [sum(vals) / len(vals) for vals in zip(*samples)]
        return avg

    def _genome_from_y(self, y):
        aquatic, diet_h, diet_c, diet_o, legs_n, size_n, speed_n, vision_n, intel, strength, armor, camo, aggr = y
        base = Genome.random_new()
        if aquatic > 0.6:
            habitat = Habitat.WATER
        elif aquatic > 0.4:
            habitat = Habitat.AMPHIBIOUS
        else:
            habitat = random.choices(
                [Habitat.LAND, Habitat.ARBOREAL, Habitat.UNDERGROUND], weights=[0.7, 0.2, 0.1]
            )[0]
        diet_scores = {Diet.HERBIVORE: diet_h, Diet.CARNIVORE: diet_c, Diet.OMNIVORE: diet_o}
        diet = max(diet_scores, key=diet_scores.get)
        legs = min(LEG_OPTIONS, key=lambda v: abs(v / 8.0 - legs_n)) if habitat != Habitat.WATER else 0
        eyes = random.choice(EYE_OPTIONS)
        sid = new_species_id()
        golden = (255, 215, int(60 + camo * 100))

        base.legs = legs
        base.eye_count = eyes
        base.eye_positions = make_eye_positions(eyes)
        base.leg_positions = make_leg_positions(legs)
        base.habitat = habitat
        base.diet = diet
        base.size = clamp(size_n * 2.5, 0.3, 2.2)
        base.sight_range = clamp(vision_n * SIGHT_MAX, SIGHT_MIN, SIGHT_MAX)
        base.intelligence = clamp01(intel)
        base.strength = clamp01(strength)
        base.armor = clamp01(armor)
        base.camouflage = clamp01(camo)
        base.aggression = clamp01(aggr)
        base.species_id = sid
        base.color = golden
        base.is_gods_creation = True
        return base

    def _valid_spawn_in_region(self, genome, region):
        c0, r0, c1, r1 = region
        c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)
        for _ in range(40):
            c = random.randint(c0, c1 - 1)
            r = random.randint(r0, r1 - 1)
            cell = self.sim.world.grid[r][c]
            if genome.can_occupy(cell):
                return c * GRID_SIZE + random.uniform(10, GRID_SIZE - 10), r * GRID_SIZE + random.uniform(10, GRID_SIZE - 10)
        return None


# =====================================================================
# SİMÜLASYON
# =====================================================================
class Simulation:
    def __init__(self):
        self.world = World()
        self.creatures = []
        self.eggs = []
        self.spawn_initial_population()
        self.god = AIGod(self)

    def spawn_initial_population(self):
        for _ in range(INITIAL_SPECIES):
            template = Genome.random_new()
            for _ in range(INITIAL_PER_SPECIES):
                x, y = self.random_valid_spawn(template)
                g = template.mutated_copy(new_species=False)
                self.creatures.append(Creature(x, y, g))

    def random_valid_spawn(self, genome: Genome):
        for _ in range(50):
            x = random.uniform(0, WORLD_W - 1)
            y = random.uniform(0, WORLD_H - 1)
            cell, c, r = self.world.cell_at(x, y)
            if genome.can_occupy(cell):
                return x, y
        return random.uniform(0, WORLD_W - 1), random.uniform(0, WORLD_H - 1)

    def step(self):
        w = self.world
        w.tick += 1
        w.update_plants()
        w.update_pollution(len(self.creatures))
        w.maybe_recalc_stats()

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
        self.god.maybe_decide()

    def update_creature(self, cr: Creature, spatial: SpatialHash):
        g = cr.genome
        cr.age += 1
        cr.energy -= 0.045 * (0.6 + g.size) * (1.0 + g.legs * 0.02) * g.energy_consumption_mult
        cr.thirst += 0.05

        if cr.age >= g.max_age() or cr.energy <= 0 or cr.thirst >= 100:
            cr.alive = False
            return

        detect_radius = max(g.sight_range, g.hearing_range * 0.6, g.smell_range * 0.5)
        nearby = spatial.query_radius(cr.x, cr.y, detect_radius)
        predators, prey, mates = [], [], []

        for other in nearby:
            if other is cr or not other.alive:
                continue
            d = math.hypot(other.x - cr.x, other.y - cr.y)
            if d > detect_radius:
                continue
            is_predator_of_me = (
                other.genome.diet in (Diet.CARNIVORE, Diet.OMNIVORE)
                and other.genome.size > g.size * 1.15 and g.diet != Diet.CARNIVORE
            )
            is_my_prey = (
                g.diet in (Diet.CARNIVORE, Diet.OMNIVORE)
                and other.genome.size < g.size * 0.9
                and other.genome.species_id != g.species_id
                and (g.aggression > 0.15 or random.random() < 0.3)
            )
            if is_predator_of_me:
                predators.append((other, d))
            if is_my_prey:
                prey.append((other, d))
            if other.genome.species_id == g.species_id and other is not cr:
                mates.append((other, d))

        target_dx, target_dy = 0.0, 0.0
        speed = g.speed()
        flee_trigger = g.sight_range * (0.4 + g.fear * 0.6)

        if predators and min(d for _, d in predators) < flee_trigger:
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
                    attack_score = g.attack_power + g.problem_solving * 0.3
                    defense_score = target.genome.defense + target.genome.armor * 0.5 + target.genome.camouflage * 0.2
                    success_chance = clamp01(0.5 + (attack_score - defense_score) * 0.6)
                    if random.random() < success_chance:
                        target.alive = False
                        cr.energy = min(g.max_energy(), cr.energy + target.genome.size * 55)
                        if target.genome.poison > 0:
                            cr.energy -= target.genome.poison * 40
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
        elif cr.repro_cooldown <= 0 and cr.energy_ratio() > 0.65 and cr.is_mature():
            cr.state = "seek_mate"
            if g.repro_mode == ReproMode.ASEXUAL:
                self.reproduce_asexual(cr)
                cr.repro_cooldown = 260 * (1.4 - g.repro_rate)
            else:
                if mates:
                    mates.sort(key=lambda t: t[1])
                    mate, dist = mates[0]
                    if dist < 20 and mate.repro_cooldown <= 0 and mate.is_mature():
                        self.reproduce_sexual(cr, mate)
                        cr.repro_cooldown = 260 * (1.4 - g.repro_rate)
                        mate.repro_cooldown = 260 * (1.4 - mate.genome.repro_rate)
                    else:
                        target_dx, target_dy = mate.x - cr.x, mate.y - cr.y
                else:
                    target_dx, target_dy = random.uniform(-1, 1), random.uniform(-1, 1)
        else:
            change_prob = 0.02 + g.curiosity * 0.05 + g.exploration * 0.03
            if random.random() < change_prob:
                cr.vx = random.uniform(-1, 1)
                cr.vy = random.uniform(-1, 1)
            target_dx, target_dy = cr.vx, cr.vy
            cr.state = "wander"

        if cr.repro_cooldown > 0:
            cr.repro_cooldown -= 1

        if random.random() > g.intelligence and cr.state != "flee":
            target_dx += random.uniform(-1, 1)
            target_dy += random.uniform(-1, 1)

        dist = math.hypot(target_dx, target_dy)
        if dist > 0.001:
            target_dx, target_dy = target_dx / dist, target_dy / dist

        step = speed / FPS
        nx = clamp(cr.x + target_dx * step, 0, WORLD_W - 1)
        ny = clamp(cr.y + target_dy * step, 0, WORLD_H - 1)

        ncell, _, _ = self.world.cell_at(nx, ny)
        if g.can_occupy(ncell):
            cr.x, cr.y = nx, ny

        if ncell.is_water and g.habitat != Habitat.WATER:
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

    def reproduce_asexual(self, cr: Creature):
        if len(self.creatures) + len(self.eggs) >= MAX_CREATURES:
            return
        cost = cr.genome.max_energy() * 0.35
        if cr.energy < cost:
            return
        cr.energy -= cost
        for _ in range(cr.genome.offspring_count):
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
        offspring_n = max(1, round((a.genome.offspring_count + b.genome.offspring_count) / 2))
        for _ in range(offspring_n):
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
                    protect = clamp01(egg.genome.parental_care) * 0.5
                    if math.hypot(cr.x - egg.x, cr.y - egg.y) < 12 and random.random() > protect:
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

    def nearest_water_dist(self, x, y, radius):
        wx, wy = self.nearest_water(x, y, radius)
        if wx is None:
            return radius
        return math.hypot(wx - x, wy - y)

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
            1.0 - (self.nearest_water_dist(wx, wy, SENSE_RADIUS) / SENSE_RADIUS))
        nearby_creatures = clamp01(pop / 20.0)
        nearby_predators = clamp01(pred / 10.0)
        danger = clamp01(0.7 * nearby_predators + 0.3 * (1.0 - nearby_food))

        terrain_code = cell.terrain.value / (len(Terrain) - 1)
        return [
            temperature, humidity, food_density, water_density, light, season,
            nearby_food, nearby_water, nearby_creatures, nearby_predators, danger,
            terrain_code, cell.vegetation_density, cell.oxygen, cell.pollution, cell.rainfall,
            cell.depth, (self.world.water_temperature_at(c, r) if cell.is_water else 0.0),
            cell.resource_regen_rate, self.world.available_niches,
            self.world.average_temperature, self.world.temperature_variance,
        ]


# =====================================================================
# EMOJI RENDER SİSTEMİ
# =====================================================================
EMOJI_FONT_CANDIDATES = ["Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", "Noto Emoji", "DejaVu Sans"]

BODY_EMOJI = {
    (Habitat.WATER, Diet.HERBIVORE): "🐠", (Habitat.WATER, Diet.CARNIVORE): "🦈", (Habitat.WATER, Diet.OMNIVORE): "🐡",
    (Habitat.AMPHIBIOUS, Diet.HERBIVORE): "🐸", (Habitat.AMPHIBIOUS, Diet.CARNIVORE): "🐊", (Habitat.AMPHIBIOUS, Diet.OMNIVORE): "🦎",
    (Habitat.ARBOREAL, Diet.HERBIVORE): "🐒", (Habitat.ARBOREAL, Diet.CARNIVORE): "🐆", (Habitat.ARBOREAL, Diet.OMNIVORE): "🐿️",
    (Habitat.UNDERGROUND, Diet.HERBIVORE): "🐀", (Habitat.UNDERGROUND, Diet.CARNIVORE): "🦂", (Habitat.UNDERGROUND, Diet.OMNIVORE): "🐛",
    (Habitat.LAND, Diet.HERBIVORE): "🐑", (Habitat.LAND, Diet.CARNIVORE): "🐺", (Habitat.LAND, Diet.OMNIVORE): "🐻",
}
FOOT_EMOJI = "🦶"
EYE_EMOJI = "👁️"
WING_EMOJI = "🪽"
ARMOR_EMOJI = "🛡️"
POISON_EMOJI = "☠️"
GOD_MARK_EMOJI = "✨"
EGG_EMOJI = "🥚"
PLANT_EMOJI = "🌿"


class EmojiRenderer:
    def __init__(self):
        self.font_name = None
        available = pygame.font.get_fonts()
        for cand in EMOJI_FONT_CANDIDATES:
            key = cand.lower().replace(" ", "")
            for f in available:
                if key in f.replace(" ", ""):
                    self.font_name = f
                    break
            if self.font_name:
                break
        self.cache = {}
        self.enabled = True

    def _font(self, size):
        size = max(6, int(size))
        if size not in self.cache:
            try:
                if self.font_name:
                    fnt = pygame.font.SysFont(self.font_name, size)
                else:
                    fnt = pygame.font.SysFont(None, size)
            except Exception:
                fnt = pygame.font.SysFont(None, size)
            self.cache[size] = {}
            self.cache[size]["_font"] = fnt
        return self.cache[size]["_font"]

    def draw(self, surf, emoji, cx, cy, size):
        size_q = max(6, int(round(size / 2.0) * 2))
        bucket = self.cache.get(size_q)
        if bucket is None:
            self._font(size_q)
            bucket = self.cache[size_q]
        img = bucket.get(emoji)
        if img is None:
            try:
                fnt = bucket["_font"]
                img = fnt.render(emoji, True, (255, 255, 255))
            except Exception:
                img = None
            bucket[emoji] = img
        if img is not None and img.get_width() > 1:
            surf.blit(img, (cx - img.get_width() / 2, cy - img.get_height() / 2))
            return True
        return False


def draw_creature_emoji(surf, renderer: EmojiRenderer, cr: Creature, cam_x, cam_y, zoom):
    x = (cr.x - cam_x) * zoom
    y = (cr.y - cam_y) * zoom
    if x < -40 or y < -40 or x > surf.get_width() + 40 or y > surf.get_height() + 40:
        return
    g = cr.genome
    body_size = (10 + g.size * 10) * zoom

    key = (g.habitat, g.diet)
    body_emoji = BODY_EMOJI.get(key, "🐾")
    ok = renderer.draw(surf, body_emoji, x, y, body_size)
    if not ok:
        pygame.draw.circle(surf, g.color, (int(x), int(y)), max(2, int(body_size / 2)))

    foot_size = body_size * 0.42
    radius = body_size * 0.55
    for angle in g.leg_positions:
        rad = math.radians(angle)
        fx = x + math.cos(rad) * radius
        fy = y + math.sin(rad) * radius
        if not renderer.draw(surf, FOOT_EMOJI, fx, fy, foot_size):
            pygame.draw.line(surf, g.color, (x, y), (fx, fy), 1)

    eye_size = body_size * 0.34
    for angle, dist_ratio in g.eye_positions:
        rad = math.radians(angle - 90)
        ex = x + math.cos(rad) * radius * dist_ratio * 0.7
        ey = y + math.sin(rad) * radius * dist_ratio * 0.7
        if not renderer.draw(surf, EYE_EMOJI, ex, ey, eye_size):
            pygame.draw.circle(surf, (255, 255, 255), (int(ex), int(ey)), 1)

    if g.fly_ability > 0.1:
        renderer.draw(surf, WING_EMOJI, x - body_size * 0.5, y, body_size * 0.7)
        renderer.draw(surf, WING_EMOJI, x + body_size * 0.5, y, body_size * 0.7)
    if g.armor > 0.45:
        renderer.draw(surf, ARMOR_EMOJI, x, y - body_size * 0.7, body_size * 0.5)
    if g.poison > 0.25:
        renderer.draw(surf, POISON_EMOJI, x + body_size * 0.5, y - body_size * 0.5, body_size * 0.45)
    if g.is_gods_creation:
        renderer.draw(surf, GOD_MARK_EMOJI, x, y + body_size * 0.75, body_size * 0.5)


# =====================================================================
# SEED / METİN GİRİŞİ EKRANI
# =====================================================================
def get_seed_input(screen, font):
    text = ""
    clock = pygame.time.Clock()
    w, h = screen.get_size()
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return text.strip() if text.strip() else None
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    ch = event.unicode
                    if ch and (ch.isalnum()):
                        text += ch

        screen.fill((18, 20, 28))
        title = font.render("TANRI EKOSİSTEM SİMÜLASYONU", True, (240, 220, 120))
        prompt = font.render("Bir seed girin (harf/rakam) ve ENTER'a basın:", True, (220, 220, 220))
        hint = font.render("(Boş bırakıp ENTER'a basarsanız rastgele seed kullanılır, ESC de aynı işi yapar)", True, (150, 150, 160))
        box = font.render(text + "|", True, (120, 230, 160))
        screen.blit(title, (w / 2 - title.get_width() / 2, h / 2 - 100))
        screen.blit(prompt, (w / 2 - prompt.get_width() / 2, h / 2 - 40))
        screen.blit(hint, (w / 2 - hint.get_width() / 2, h / 2 - 10))
        screen.blit(box, (w / 2 - box.get_width() / 2, h / 2 + 30))
        pygame.display.flip()


# =====================================================================
# ANA DÖNGÜ
# =====================================================================
def draw_minimap(screen, sim: Simulation, cam_x, cam_y, visible_w, visible_h, region_cols, region_rows):
    w, h = screen.get_size()
    mx0 = w - MINIMAP_W - MINIMAP_MARGIN
    my0 = MINIMAP_MARGIN
    mm = pygame.Surface((MINIMAP_W, MINIMAP_H))
    mm.fill((15, 16, 22))

    sx = MINIMAP_W / WORLD_W
    sy = MINIMAP_H / WORLD_H

    for r in range(0, GRID_ROWS, max(1, GRID_ROWS // 60)):
        for c in range(0, GRID_COLS, max(1, GRID_COLS // 60)):
            cell = sim.world.grid[r][c]
            color = TERRAIN_BASE_COLOR.get(cell.terrain, (90, 90, 90))
            px = int(c * GRID_SIZE * sx)
            py = int(r * GRID_SIZE * sy)
            mm.set_at((min(px, MINIMAP_W - 1), min(py, MINIMAP_H - 1)), color)

    for i in range(1, region_cols):
        xpix = int(i * (WORLD_W / region_cols) * sx)
        pygame.draw.line(mm, (60, 60, 70), (xpix, 0), (xpix, MINIMAP_H), 1)
    for i in range(1, region_rows):
        ypix = int(i * (WORLD_H / region_rows) * sy)
        pygame.draw.line(mm, (60, 60, 70), (0, ypix), (MINIMAP_W, ypix), 1)

    ac0, ar0, ac1, ar1 = sim.god.active_region
    ax0 = ac0 * GRID_SIZE * sx
    ay0 = ar0 * GRID_SIZE * sy
    aw = (ac1 - ac0) * GRID_SIZE * sx
    ah = (ar1 - ar0) * GRID_SIZE * sy
    pygame.draw.rect(mm, (255, 210, 60), (ax0, ay0, max(aw, 1), max(ah, 1)), 2)

    diet_color = {Diet.HERBIVORE: (90, 220, 110), Diet.CARNIVORE: (230, 70, 70), Diet.OMNIVORE: (230, 200, 60)}
    for cr in sim.creatures:
        px = int(cr.x * sx)
        py = int(cr.y * sy)
        if 0 <= px < MINIMAP_W and 0 <= py < MINIMAP_H:
            mm.set_at((px, py), diet_color.get(cr.genome.diet, (255, 255, 255)))

    vx0, vy0 = cam_x * sx, cam_y * sy
    vw, vh = visible_w * sx, visible_h * sy
    pygame.draw.rect(mm, (255, 255, 255), (vx0, vy0, max(vw, 2), max(vh, 2)), 1)

    pygame.draw.rect(mm, (200, 200, 210), (0, 0, MINIMAP_W, MINIMAP_H), 1)
    screen.blit(mm, (mx0, my0))


def run():
    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Tanrı AI Ekosistem Simülasyonu")
    clock = pygame.time.Clock()
    ui_font = pygame.font.SysFont("arial", 22)
    hud_font = pygame.font.SysFont("arial", 18)
    big_font = pygame.font.SysFont("arial", 26, bold=True)

    seed_text = get_seed_input(screen, ui_font)
    if seed_text is not None:
        try:
            random.seed(int(seed_text))
        except ValueError:
            random.seed(seed_text)
        print(f"[Seed] Kullanılan seed: {seed_text}")
    else:
        print("[Seed] Rastgele seed kullanılıyor.")

    sim = Simulation()
    emoji_renderer = EmojiRenderer()

    zoom = 0.25
    cam_x = WORLD_W / 2 - (screen_w / zoom) / 2
    cam_y = WORLD_H / 2 - (screen_h / zoom) / 2
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
                elif event.key == pygame.K_r:
                    sim = Simulation()
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_x_before = cam_x + mouse_x / zoom
                world_y_before = cam_y + mouse_y / zoom
                if event.y > 0:
                    zoom = min(ZOOM_MAX, zoom * ZOOM_STEP)
                elif event.y < 0:
                    zoom = max(ZOOM_MIN, zoom / ZOOM_STEP)
                cam_x = world_x_before - mouse_x / zoom
                cam_y = world_y_before - mouse_y / zoom

        keys = pygame.key.get_pressed()
        pan = (PAN_SPEED / zoom) * dt
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            cam_x -= pan
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            cam_x += pan
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            cam_y -= pan
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            cam_y += pan

        visible_w = screen_w / zoom
        visible_h = screen_h / zoom
        cam_x = clamp(cam_x, 0, max(0, WORLD_W - visible_w))
        cam_y = clamp(cam_y, 0, max(0, WORLD_H - visible_h))

        steps = 4 if turbo else 1
        for _ in range(steps):
            sim.step()

        # --- ÇİZİM ---
        screen.fill((10, 10, 14))
        cell_px = GRID_SIZE * zoom
        c0 = max(0, int(cam_x // GRID_SIZE))
        r0 = max(0, int(cam_y // GRID_SIZE))
        c1 = min(GRID_COLS, int((cam_x + visible_w) // GRID_SIZE) + 1)
        r1 = min(GRID_ROWS, int((cam_y + visible_h) // GRID_SIZE) + 1)

        show_plants = zoom >= LOD_PLANT_ZOOM
        for r in range(r0, r1):
            row = sim.world.grid[r]
            for c in range(c0, c1):
                cell = row[c]
                color = TERRAIN_BASE_COLOR.get(cell.terrain, (90, 90, 90))
                px = (c * GRID_SIZE - cam_x) * zoom
                py = (r * GRID_SIZE - cam_y) * zoom
                pygame.draw.rect(screen, color, (px, py, cell_px + 1, cell_px + 1))
                if not show_plants and not cell.is_water and cell.plants > 0:
                    tint = min(255, 60 + cell.plants * 25)
                    pygame.draw.rect(screen, (30, tint, 40), (px, py, cell_px + 1, cell_px + 1), 0)
                elif show_plants and not cell.is_water and cell.plants > 0:
                    for i in range(cell.plants):
                        ppx = px + cell_px * (0.15 + (i % 3) * 0.3)
                        ppy = py + cell_px * (0.15 + (i // 3) * 0.3)
                        if not emoji_renderer.draw(screen, PLANT_EMOJI, ppx, ppy, cell_px * 0.35):
                            pygame.draw.circle(screen, (40, 170, 60), (int(ppx), int(ppy)), max(1, int(cell_px * 0.08)))

        for egg in sim.eggs:
            ex = (egg.x - cam_x) * zoom
            ey = (egg.y - cam_y) * zoom
            if zoom < LOD_EMOJI_ZOOM:
                screen.set_at((int(ex), int(ey)), (230, 230, 200))
            else:
                if not emoji_renderer.draw(screen, EGG_EMOJI, ex, ey, 14 * zoom):
                    pygame.draw.ellipse(screen, (240, 240, 210), (ex - 4, ey - 6, 8, 12))

        if zoom < LOD_EMOJI_ZOOM:
            for cr in sim.creatures:
                px = int((cr.x - cam_x) * zoom)
                py = int((cr.y - cam_y) * zoom)
                if 0 <= px < screen_w and 0 <= py < screen_h:
                    screen.set_at((px, py), cr.genome.color)
        else:
            for cr in sim.creatures:
                draw_creature_emoji(screen, emoji_renderer, cr, cam_x, cam_y, zoom)

        draw_minimap(screen, sim, cam_x, cam_y, visible_w, visible_h, REGION_COLS, REGION_ROWS)

        active_species = len(set(cr.genome.species_id for cr in sim.creatures if cr.alive))
        season_name = SEASONS[sim.world.season_index()]
        hud_lines = [
            f"Tick: {sim.world.tick}   Mevsim: {season_name}   Zoom: {zoom:.2f}x",
            f"Popülasyon: {len(sim.creatures)}   Yumurta: {len(sim.eggs)}   Aktif Tür: {active_species}",
            f"Model: {'YÜKLENDİ' if sim.god.loaded else 'EĞİTİLMEMİŞ (rastgele)'}   Sonraki karar: {GOD_DECISION_INTERVAL - (sim.world.tick % GOD_DECISION_INTERVAL)} tick",
            f"Tanrı: {sim.god.last_decision_text}",
            "Yön tuşları/WASD: kaydır   Fare tekerleği: zoom   SPACE: turbo   R: sıfırla   ESC: çıkış",
        ]
        for i, line in enumerate(hud_lines):
            txt = hud_font.render(line, True, (255, 255, 255))
            screen.blit(txt, (14, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()