from dataclasses import dataclass, field
import numpy as np
import requests
import json
from .atom import ELEMENT_SYMBOL

ODD_DF = np.array([1, 1, 2, 3, 8, 15, 48, 105, 384, 945])
PI = np.pi

def l_to_ijk(L):
    return [
        (i, j, L - i - j)
        for i in range(L, -1, -1)
        for j in range(L - i, -1, -1)
    ]

def primitive_cart_norm(alpha, l, m, n):
    L = l + m + n
    num = (2 * alpha/PI)**(0.75)*(4 * alpha)**(0.5*L)
    den = np.sqrt(ODD_DF[l] * ODD_DF[m] * ODD_DF[n])
    return num/den

@dataclass
class Shell:
    l : int
    exponents: np.ndarray
    coefficients: np.ndarray
    norm_factors: np.ndarray = field(init=False)
    #center: np.ndarray =field(init = False)

    def __post_init__(self):
        self.norm_factors = self.get_norm_factors()

    def get_norm_factors(self):
        ijk = l_to_ijk(self.l)
        norm_factors = np.empty((len(self.exponents),len(ijk)), dtype = np.float64)
        for i, (l, m, n) in enumerate(ijk):
            norm_factors[: ,i] = primitive_cart_norm(self.exponents, l, m, n)
        return norm_factors
    
    def set_center(self, center: np.ndarray) -> None:
        object.__setattr__(self, "center", np.asarray(center, dtype=float))
    
@dataclass
class BasisSet:
    name: str
    elements: dict[str, list[Shell]] = field(default_factory = dict)

    def download_from_bse(self, element_list: list, timeout: int = 30) -> None:
        url = f"https://www.basissetexchange.org/api/basis/{self.name}/format/json/"
        url += f"?elements={','.join(element_list)}"
        response = requests.get(url, timeout = timeout)
        data_json = response.json()
        print(f"Dump data to {self.name}.json")
        json.dump(data_json, open(f"{self.name}.json", "w"), indent = 4)
        self.parse_elements(data_json)

    def parse_elements(self, data: dict) -> None:
        self.elements = {}
        basis_data = data.get("elements", {})
        for atomic_number, element_data in basis_data.items():
            symbol = ELEMENT_SYMBOL.get(int(atomic_number))
            shells: list[Shell] = []
            for shell_data in element_data.get("electron_shells", []):
                angular_momentum = shell_data.get("angular_momentum")
                contraction_coefficients = shell_data.get("coefficients")
                exponents = shell_data["exponents"]
                for l, coefficients in zip(angular_momentum, contraction_coefficients):
                    shells.append(Shell(l = int(l),
                                       exponents = np.asarray(exponents, dtype = float),
                                       coefficients = np.asarray(coefficients, dtype = float)))
            self.elements[symbol] = shells

    def __str__(self) -> str:
        output = f"Basis set: {self.name}\n"
        for symbol, shells in self.elements.items():
            output += f"Element: {symbol}\n"
            for shell in shells:
                output += f"l = {shell.l}, exponents = {shell.exponents}, coefficients = {shell.coefficients}\n"
                #output += f"norm_factors = {shell.norm_factors}"
        return(output)