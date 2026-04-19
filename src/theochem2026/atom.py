from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

ELEMENT_SYMBOLS = {atomic_number: symbol for symbol, atomic_number in ATOMIC_NUMBERS.items()}


@dataclass
class Atom:
    symbol: str
    coord: Sequence[float]

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().capitalize()
        self.coord = np.asarray(self.coord, dtype=float)
        if self.coord.shape != (3,):
            raise ValueError("Coordinates must be a 3D vector.")

    def __str__(self) -> str:
        return f"{self.symbol} at {self.coord}"

    @classmethod
    def from_string(cls, s: str) -> Atom:
        parts = s.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 'SYMBOL x y z', got: {s!r}")

        symbol = parts[0]
        try:
            coord = np.asarray(parts[1:4], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates in atom string: {s!r}") from exc
        return cls(symbol, coord)

    @property
    def atomic_number(self) -> int:
        try:
            return ATOMIC_NUMBERS[self.symbol]
        except KeyError as exc:
            raise ValueError(f"Unknown element symbol: {self.symbol!r}") from exc

    @property
    def n_electrons(self) -> int:
        return self.atomic_number

    def get_distance(self, other: Atom) -> float:
        if not isinstance(other, Atom):
            raise ValueError("Distance can only be calculated between two Atom instances.")
        return float(np.linalg.norm(self.coord - other.coord))
