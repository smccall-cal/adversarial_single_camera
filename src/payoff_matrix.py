#!/bin/src/python

import numpy as np
import itertools

from typing import Set, List


class PayoffMatrix:
    def __init__(self, floor_map: np.array, weight: float = 1):
        self.floor_map: np.array = floor_map.astype(bool)
        self.dimension: int = len(self.floor_map.shape)

        self.weight: float = weight
        self.center: tuple = tuple(0 for _ in range(self.dimension))

        self.values: np.array = np.zeros((
            * self.floor_map.shape,
            * (3 for _ in range(self.dimension))
        ))

        self.payoff: np.array = np.zeros(self.floor_map.shape)

    def __getitem__(self, position):
        return self.payoff[position]

    def expected_payoff(self, strategy: np.array):
        for corner, directions in self.corners():
            for position in self.iterate_positions(corner, 0):
                self.update_position(strategy[position], position, directions)

        corner = tuple(0 for _ in range(self.dimension))
        for position in self.iterate_positions(corner, 0):
            self.payoff[position] = self.values[position].sum()

        return self.payoff

    def iterate_positions(self, corner: tuple, axis: int):
        if axis < len(corner):
            if (current := corner[axis]) > 0:
                iterable = range(current - 1, -1, -1)
            else:
                iterable = range(self.floor_map.shape[axis])

            for i in iterable:
                for j in self.iterate_positions(corner, axis + 1):
                    yield tuple((i, * j))
        else:
            yield tuple()

    def update_position(self, strategy: float, position: tuple, directions: Set[tuple]):
        if not self.is_wall(position):
            self.values[position][self.center] = strategy

            for direction in directions:
                self.values[position][direction] = self.conical_value(position, direction)

    def conical_value(self, position: tuple, direction: tuple) -> float:
        position: tuple = tuple(sum(c) for c in zip(position, direction))
        conical_value: float = 0.0

        for direction in self.cone(direction):
            conical_value = conical_value + self.values[position][direction]

        return self.weight * conical_value

    def cone(self, direction: tuple) -> Set[tuple]:
        intermediate = itertools.product(* zip(direction, self.center))
        return set(intermediate)

    def is_wall(self, position: tuple) -> bool:
        return self.floor_map[position]

    def corners(self) -> (tuple, Set[tuple]):
        corner_positions: List[tuple] = list(itertools.product(*(
            (0, extrema) for extrema in self.floor_map.shape
        )))

        visited_directions: Set[tuple] = {self.center}
        corner, index = tuple(), 0

        while corner_positions:
            corner, index = corner_positions.pop(index), 0 if corner else -1

            directions: Set[tuple] = self.compatible_directions(corner, visited_directions)
            visited_directions = visited_directions.union(directions)

            if directions:
                yield corner, directions

    @staticmethod
    def compatible_directions(corner: tuple, visited: Set[tuple]) -> Set[np.array]:
        dirs = itertools.product(*(
            (np.sign(c - 1), 0) for c in corner
        ))
        return set(dirs).difference(visited)
