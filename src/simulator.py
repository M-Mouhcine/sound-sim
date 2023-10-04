import itertools
import numpy as np
from numpy.core.umath import pi
from numpy.ma import sin
from copy import deepcopy
from skimage.draw import random_shapes

# initial values
damping = 1 - 0.001
ca2 = 0.5
initial_P = 200
max_pressure = initial_P / 2
min_presure = -initial_P / 2


def generate_random_map(map_size, random_seed):
    """Randomly generate an array of zeros (free media) and ones (obstacles).
    The obstacles have basic geometric shapes.

    Parameters:
        map_size(tuple): shape of the map to generate
        random_seed (int): random seed for random generation of obstacles
    Returns:
        random_map (np.array): array shaped as map_size, containing random
                               obstacles.
    """
    result = random_shapes(
        map_size,
        intensity_range=(0, 1),
        min_size=8,
        max_size=15,
        min_shapes=2,
        max_shapes=10,
        random_seed=random_seed,
        num_channels=1,
        allow_overlap=False,
    )
    # result is a tuple consisting of
    # # (1) the image with the generated shapes
    # # (2) a list of label tuples with the kind of shape
    # (e.g. circle, rectangle) and ((r0, r1), (c0, c1)) coordinates.
    obstacle_map, labels = result
    # Force free media in a square of 20x20 at the center of the map
    width_center = map_size[0] // 2
    length_center = map_size[1] // 2
    obstacle_map[
        width_center - 20 : width_center + 21,
        length_center - 20 : length_center + 21,
    ] = 255
    free_media = obstacle_map == 255
    # Obstacles = 1, free media = 0
    obstacles = obstacle_map == 0
    obstacle_map[free_media] = 0
    obstacle_map[obstacles] = 1
    return obstacle_map[..., 0]


class SoundSimulator:
    """Class for the configuration and simulation of sound propagation in a map
    with obstacles.
    Adapted from https://github.com/Alexander3/wave-propagation
    Based on Komatsuzaki T. "Modelling of Incident Sound Wave Propagation
    around Sound Barriers Using Cellular Automata" (2012)

    Attributes:
        map_size (tuple): size of the map
        obstacle_map (np.array): free media = 0, obstacles = 1. If the given
            shape is different from map_size, ignore the parameters and
            generate a map with no obstacles.
        duration (int): duration (in seconds) of the simulation.
        iteration (int): current iteration of the simulation.
        size_x (int): number of cols in the grid.
        size_y (int): number of cols in the grid.
        pressure (np.array): pressure field at current iteration.
        pressure_hist (np.array): history of all simulated pressure fields.
        _velocities (np.array): velocity field at current iteration.
    """

    def __init__(self, map_size, obstacle_map=None, duration=60):
        self.duration = duration
        self.omega = 3 / (2 * pi)
        self.iteration = 0
        self.size_y, self.size_x = map_size
        # obstacle_map handling
        if obstacle_map is not None and obstacle_map.shape == map_size:
            print("** Map Accepted **")
            self.obstacle_map = obstacle_map
        elif obstacle_map is not None and obstacle_map.shape != map_size:
            print("** Map size denied **")
            self.obstacle_map = np.zeros((self.size_y, self.size_x))
        else:
            self.obstacle_map = np.zeros((self.size_y, self.size_x))
        # Source position is the center of the map
        self.s_y = self.size_y // 2
        self.s_x = self.size_x // 2
        self.pressure = np.zeros((self.size_y, self.size_x))
        self.pressure_hist = np.zeros((duration, self.size_y, self.size_x))
        # outflow velocities from each cell
        self._velocities = np.zeros((self.size_y, self.size_x, 4))

    def updateV(self):
        """Update the velocity field based on Komatsuzaki's transition rules."""
        V = self._velocities
        P = self.pressure

        old_V = deepcopy(V)

        V[..., 0] += P + np.roll(P, -1, axis=0)
        V[0, :, 0] = old_V[0, :, 0] + P[0, :]

        V[..., 1] += P + np.roll(P, 1, axis=1)
        V[:, -1, 1] = old_V[:, -1, 1] + P[:, -1]

        V[..., 2] += P + np.roll(P, 1, axis=0)
        V[-1, :, 2] = old_V[-1, :, 2] + P[-1, :]

        V[..., 3] += P + np.roll(P, -1, axis=1)
        V[:, 0, 3] = old_V[:, 0, 3] + P[:, 0]

        self._velocities = V
        self._velocities *= 1 - self.obstacle_map[..., np.newaxis]

    def updateP(self):
        """Update the pressure field based on Komatsuzaki's transition rules."""
        self.pressure -= ca2 * damping * np.sum(self._velocities, axis=2)

    def step(self):
        """Perform a simulation step, upadting the wind an pressure fields."""
        self.pressure[self.s_y, self.s_x] = initial_P * sin(
            self.omega * self.iteration
        )
        self.updateV()
        self.updateP()
        self.iteration += 1

    def spl(self, integration_interval=200):
        """Compute the sound pressure level map.
        https://en.wikipedia.org/wiki/Sound_pressure#Sound_pressure_level

        Parameters:
            integration_interval (int): interval over which the rms pressure
                                        is computed, starting from the last
                                        simulation iteration backwards.
        Returns:
            spl (np.array): map of sound pressure level (dB).
        """
        p0 = 20 * 10e-6  # Pa
        if integration_interval > self.pressure_hist.shape[0]:
            integration_interval = self.pressure_hist.shape[0]
        rms_p = np.sqrt(
            np.mean(
                np.square(self.pressure_hist[-integration_interval:-1]), axis=0
            )
        )
        return 20 * np.log10(rms_p / p0)

    def run(self):
        for iteration in range(self.duration):
            self.pressure_hist[iteration] = deepcopy(self.pressure)
            self.step()
