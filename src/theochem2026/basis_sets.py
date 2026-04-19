from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence 
import numpy as np
from .atom import ATOMIC_NUMBERS, ELEMENT_SYMBOLS
import requests
import os
import json

# (2k-1)!! for k = 0..12  -> max angular momentum index 12
ODD_DF = np.array([
    1, 1, 3, 15, 105, 945, 10395, 135135,
    2027025, 34459425, 654729075, 13749310575, 316234143225
], dtype=np.float64)

PI = np.pi

def l_to_ijk(L):
    IJK = []
    for I in range(L, -1, -1):
        for J in range(L - I, -1, -1):
            IJK.append((I, J, L - I - J))
    return sorted(IJK, reverse=True)

def primitive_cart_norm(alpha, l, m, n):
    # normalized for (x-Ax)^l (y-Ay)^m (z-Az)^n * exp(-alpha r^2)
    alpha = np.float64(alpha)
    L = l + m + n
    num = (2.0 * alpha / PI) ** 0.75 * (4.0 * alpha) ** (0.5 * L)
    den = np.sqrt(ODD_DF[l] * ODD_DF[m] * ODD_DF[n])
    return num / den

@dataclass 
class Shell:
    l: int
    exponents: np.ndarray
    coefficients: np.ndarray
    norm_factors: np.ndarray  = field(init=False)
    center: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "exponents", np.asarray(self.exponents, dtype=float))
        object.__setattr__(self, "coefficients", np.asarray(self.coefficients, dtype=float))
        self.norm_factors = self.get_norm_factors()

    def get_norm_factors(self) -> np.ndarray:
        ijk = l_to_ijk(self.l)
        norm_factors = np.empty((len(self.exponents), len(ijk)), dtype=np.float64)
        for i, (l, m, n) in enumerate(ijk):
            norm_factors[:, i] = primitive_cart_norm(self.exponents, l, m, n)
        return norm_factors
    
    def set_center(self, center: np.ndarray) -> None:
        object.__setattr__(self, "center", np.asarray(center, dtype=float))
    
@dataclass 
class BasisSet:
    name: str
    elements: dict[str, list[Shell]] = field(default_factory=dict)

    def download_from_bse(self, element_list, timeout: int = 30):
        url = f"https://www.basissetexchange.org/api/basis/{self.name}/format/json/"
        url += f"?elements={','.join(element_list)}"
        response = requests.get(url, timeout = timeout)
        response.raise_for_status()
        data_json = response.json()
        self.parse_elements(data_json)
        json.dump(data_json, open(f"{self.name}.json", "w"), indent=4)

    def parse_elements(self, data: dict) -> None:
        self.elements = {}
        basis_data = data.get("elements", {})
        for atomic_number, element_data in basis_data.items():
            symbol = ELEMENT_SYMBOLS.get(int(atomic_number))
            shells: list[Shell] = []
            for shell_data in element_data.get("electron_shells", []):
                angular_momenta = shell_data["angular_momentum"]
                contraction_coefficients = shell_data["coefficients"]
                exponents = shell_data["exponents"]
                for l, coefficients in zip(angular_momenta, contraction_coefficients):
                    shells.append(
                        Shell(
                            l=int(l),
                            exponents=exponents,
                            coefficients=np.asarray(coefficients, dtype=float),
                        )
                    )
            self.elements[symbol] = shells

    def get_basis_set(self, element_list: Sequence[str]) -> None:
        # if name.json file exists, load it, otherwise download from BSE and save to disk
        filename = f"{self.name}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data_json = json.load(f)
            self.parse_elements(data_json)
        else:
            self.download_from_bse(element_list)  # download for all elements in element_list

    def __str__(self) -> str:
        output = f"Basis set: {self.name}\n"
        for symbol, shells in self.elements.items():
            output += f"Element: {symbol}\n"
            for shell in shells:
                output += f"  l={shell.l}, exponents={shell.exponents}, coefficients={shell.coefficients}\n"
        return output

