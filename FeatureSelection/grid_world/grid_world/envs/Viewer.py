import pygame as pg
import random
from PIL import Image

class Viewer:

    def __init__(self, height=9, width=16, cell_size=10, wall=True):
        pg.init()
        self.height = height
        self.width = width
        self.cell_size = cell_size
        self.wall = wall
        self.screen_size = (self.width * self.cell_size, self.height * self.cell_size)

        # Position data
        self.goal_pos = []
        self.wall_pos = []
        self.char_pos = ()

        # Directions
        self.directions = {
            0: (0,  -self.cell_size), # UP
            1: (0, self.cell_size), # DOWN
            2: (self.cell_size, 0), # RIGHT
            3: (-self.cell_size, 0) # LEFT
        }

        # Main surfaces
        self.surface = pg.Surface(self.screen_size)  # Main game surface
        self.surface.fill((0, 0, 0))
        self.character = pg.Surface((self.cell_size, self.cell_size))  # Character surface
        self.character.fill((255, 255, 255))
        self.goal = pg.Surface((self.cell_size, self.cell_size)) # Goal surface
        self.goal.fill((0, 255, 255))
        self.wall = pg.Surface((self.cell_size, self.cell_size)) # Wall surface
        self.wall.fill((0, 0, 255))

        # Screen
        self.screen = None

        self.initialize()
        self.draw()

    def initialize(self):
        # Reset position data
        self.goal_pos = set()
        self.wall_pos = set()
        self.char_pos = ()

        # Init goal
        for y in range(self.height):
            self.goal_pos.add(((self.width - 1) * self.cell_size, y * self.cell_size))

        # Init wall
        if self.wall:
            x = random.randrange(1, self.width - 1) * self.cell_size # Wall can't overlap with goal
            for y in range(random.randrange(2, self.height/2)): # Wall must be at least 2 cells long
                self.wall_pos.add((x, y * self.cell_size))

        # Init character
        y = random.randrange(0, self.height) * self.cell_size
        self.char_pos = (0, y)  # Initial position is random
        self.draw()

    def draw(self):
        # Clear surface
        self.surface.fill((0, 0, 0))

        # Draw goal
        for p in self.goal_pos:
            self.surface.blit(self.goal, p)

        # Draw wall
        if self.wall:
            for p in self.wall_pos:
                self.surface.blit(self.wall, p)

        # Draw character
        self.surface.blit(self.character, self.char_pos)

    def render(self):
        if self.screen is None:
            self.screen = pg.display.set_mode(self.screen_size)
        self.screen.blit(self.surface, (0, 0))
        self.draw()
        pg.display.update()

    def move_character(self, action):
        self.char_pos = self._get_new_char_pos(action)
        self.draw()

    def get_state(self):
        str = pg.image.tostring(self.surface, 'RGB')
        out = Image.fromstring('RGB', self.screen_size, str).convert('L')
        return out

    def is_on_goal(self):
        return self.char_pos in self.goal_pos

    def close(self):
        if self.screen is not None:
            pg.display.quit()

    # HELPERS
    def _get_new_char_pos(self, action):
        addendum = self.directions[action]
        x = self.char_pos[0] + addendum[0]
        y = self.char_pos[1] + addendum[1]
        new_pos = (x, y)
        if new_pos in self.wall_pos or x < 0 or x >= self.width * self.cell_size or y < 0 or y >= self.height * self.cell_size:
            return self.char_pos
        else:
            return new_pos
