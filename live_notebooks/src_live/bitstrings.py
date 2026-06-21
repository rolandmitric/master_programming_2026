from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


def _occupied_indices(bits: int, n_orbitals: int) -> tuple[int, ...]:
    return tuple(orbital for orbital in range(n_orbitals) if (bits >> orbital) & 1)


def _bits_from_occupied(occupied: Iterable[int], n_orbitals: int) -> int:
    bits = 0
    for orbital in occupied:
        if not 0 <= orbital < n_orbitals:
            raise ValueError(f"Orbital index {orbital} is out of range for {n_orbitals} orbitals.")
        if (bits >> orbital) & 1:
            raise ValueError(f"Orbital index {orbital} was specified more than once.")
        bits |= 1 << orbital
    return bits


@dataclass(frozen=True, slots=True)
class Determinant:
    alpha: int = 0
    beta: int = 0

    @property
    def n_alpha(self) -> int:
        return self.alpha.bit_count()

    @property
    def n_beta(self) -> int:
        return self.beta.bit_count()

    def occupied_alpha(self, n_orbitals: int) -> tuple[int, ...]:
        return _occupied_indices(self.alpha, n_orbitals)

    def occupied_beta(self, n_orbitals: int) -> tuple[int, ...]:
        return _occupied_indices(self.beta, n_orbitals)

    def as_bitstrings(self, n_orbitals: int) -> tuple[str, str]:
        return (
            format(self.alpha, f"0{n_orbitals}b"),
            format(self.beta, f"0{n_orbitals}b"),
        )


class BitstringHamiltonian:
    """Small determinant helper built from spatial-orbital integrals."""

    def __init__(
        self,
        one_electron_integrals: np.ndarray,
        two_electron_integrals: np.ndarray,
        nuclear_repulsion: float = 0.0,
    ) -> None:
        h1 = np.asarray(one_electron_integrals, dtype=float)
        g2 = np.asarray(two_electron_integrals, dtype=float)

        if h1.ndim != 2 or h1.shape[0] != h1.shape[1]:
            raise ValueError("One-electron integrals must have shape (n, n).")
        if g2.ndim != 4 or g2.shape != (h1.shape[0],) * 4:
            raise ValueError("Two-electron integrals must have shape (n, n, n, n).")

        self.h1 = h1
        self.g2 = g2
        self.nuclear_repulsion = float(nuclear_repulsion)
        self.n_spatial_orbitals = h1.shape[0]
        self.n_spin_orbitals = 2 * self.n_spatial_orbitals

    def determinant(
        self,
        alpha_occupied: Iterable[int] = (),
        beta_occupied: Iterable[int] = (),
    ) -> Determinant:
        return Determinant(
            alpha=_bits_from_occupied(alpha_occupied, self.n_spatial_orbitals),
            beta=_bits_from_occupied(beta_occupied, self.n_spatial_orbitals),
        )

    def reference(self, n_alpha: int, n_beta: int) -> Determinant:
        if not 0 <= n_alpha <= self.n_spatial_orbitals:
            raise ValueError("n_alpha is out of range.")
        if not 0 <= n_beta <= self.n_spatial_orbitals:
            raise ValueError("n_beta is out of range.")
        return self.determinant(range(n_alpha), range(n_beta))

    def generate_determinants(self, n_alpha: int, n_beta: int) -> list[Determinant]:
        if not 0 <= n_alpha <= self.n_spatial_orbitals:
            raise ValueError("n_alpha is out of range.")
        if not 0 <= n_beta <= self.n_spatial_orbitals:
            raise ValueError("n_beta is out of range.")

        determinants: list[Determinant] = []
        orbitals = range(self.n_spatial_orbitals)
        for alpha_occ in combinations(orbitals, n_alpha):
            for beta_occ in combinations(orbitals, n_beta):
                determinants.append(self.determinant(alpha_occ, beta_occ))
        return determinants

    def create(self, determinant: Determinant, orbital: int, spin: str) -> tuple[int, Determinant | None]:
        spin_orbital = self.spin_orbital_index(orbital, spin)
        phase, bits = self._apply_creation(self._spin_bits(determinant), spin_orbital)
        if phase == 0:
            return 0, None
        return phase, self._from_spin_bits(bits)

    def annihilate(
        self,
        determinant: Determinant,
        orbital: int,
        spin: str,
    ) -> tuple[int, Determinant | None]:
        spin_orbital = self.spin_orbital_index(orbital, spin)
        phase, bits = self._apply_annihilation(self._spin_bits(determinant), spin_orbital)
        if phase == 0:
            return 0, None
        return phase, self._from_spin_bits(bits)

    def diagonal_energy(self, determinant: Determinant) -> float:
        return float(self.matrix_element(determinant, determinant))

    def matrix_element(self, left: Determinant, right: Determinant) -> float:
        left_bits = self._spin_bits(left)
        right_bits = self._spin_bits(right)

        created_bits = left_bits & ~right_bits
        removed_bits = right_bits & ~left_bits
        n_created = created_bits.bit_count()
        n_removed = removed_bits.bit_count()

        if n_created != n_removed or n_created > 2:
            return 0.0

        left_occ = self.occupied_spin_orbitals(left)
        if n_created == 0:
            value = self.nuclear_repulsion
            for i in left_occ:
                value += self.spin_orbital_one_body(i, i)
            for i in left_occ:
                for j in left_occ:
                    value += 0.5 * self.antisymmetrized_two_body(i, j, i, j)
            return float(value)

        created = _occupied_indices(created_bits, self.n_spin_orbitals)
        removed = _occupied_indices(removed_bits, self.n_spin_orbitals)

        if n_created == 1:
            particle = created[0]
            hole = removed[0]
            phase = self._single_excitation_phase(right_bits, particle, hole)
            if phase == 0:
                return 0.0

            common_bits = right_bits & ~(1 << hole)
            common = _occupied_indices(common_bits, self.n_spin_orbitals)
            value = self.spin_orbital_one_body(particle, hole)
            for occupied in common:
                value += self.antisymmetrized_two_body(particle, occupied, hole, occupied)
            return float(phase * value)

        particle_p, particle_q = created
        hole_r, hole_s = removed
        phase = self._double_excitation_phase(
            right_bits,
            particle_p,
            particle_q,
            hole_r,
            hole_s,
        )
        if phase == 0:
            return 0.0

        value = self.antisymmetrized_two_body(particle_p, particle_q, hole_r, hole_s)
        return float(phase * value)

    def hamiltonian(self, determinants: Sequence[Determinant]) -> np.ndarray:
        n_det = len(determinants)
        hamiltonian = np.zeros((n_det, n_det), dtype=float)

        for col, right in enumerate(determinants):
            for row in range(col + 1):
                value = self.matrix_element(determinants[row], right)
                hamiltonian[row, col] = value
                hamiltonian[col, row] = value
        return hamiltonian

    def occupied_spin_orbitals(self, determinant: Determinant) -> tuple[int, ...]:
        alpha_occ = determinant.occupied_alpha(self.n_spatial_orbitals)
        beta_occ = tuple(
            orbital + self.n_spatial_orbitals
            for orbital in determinant.occupied_beta(self.n_spatial_orbitals)
        )
        return alpha_occ + beta_occ

    def spin_orbital_index(self, orbital: int, spin: str) -> int:
        if not 0 <= orbital < self.n_spatial_orbitals:
            raise ValueError(
                f"Orbital index {orbital} is out of range for {self.n_spatial_orbitals} orbitals."
            )
        if spin == "alpha":
            return orbital
        if spin == "beta":
            return orbital + self.n_spatial_orbitals
        raise ValueError("Spin must be 'alpha' or 'beta'.")

    def spin_orbital_one_body(self, p: int, q: int) -> float:
        p_spatial, p_spin = self._split_spin_orbital(p)
        q_spatial, q_spin = self._split_spin_orbital(q)
        if p_spin != q_spin:
            return 0.0
        return float(self.h1[p_spatial, q_spatial])

    def antisymmetrized_two_body(self, p: int, q: int, r: int, s: int) -> float:
        p_spatial, p_spin = self._split_spin_orbital(p)
        q_spatial, q_spin = self._split_spin_orbital(q)
        r_spatial, r_spin = self._split_spin_orbital(r)
        s_spatial, s_spin = self._split_spin_orbital(s)

        direct = 0.0
        exchange = 0.0
        if p_spin == r_spin and q_spin == s_spin:
            direct = float(self.g2[p_spatial, q_spatial, r_spatial, s_spatial])
        if p_spin == s_spin and q_spin == r_spin:
            exchange = float(self.g2[p_spatial, q_spatial, s_spatial, r_spatial])
        return direct - exchange

    def _split_spin_orbital(self, spin_orbital: int) -> tuple[int, int]:
        if not 0 <= spin_orbital < self.n_spin_orbitals:
            raise ValueError(
                f"Spin orbital index {spin_orbital} is out of range for {self.n_spin_orbitals} orbitals."
            )
        if spin_orbital < self.n_spatial_orbitals:
            return spin_orbital, 0
        return spin_orbital - self.n_spatial_orbitals, 1

    def _spin_bits(self, determinant: Determinant) -> int:
        return determinant.alpha | (determinant.beta << self.n_spatial_orbitals)

    def _from_spin_bits(self, bits: int) -> Determinant:
        mask = (1 << self.n_spatial_orbitals) - 1
        return Determinant(alpha=bits & mask, beta=bits >> self.n_spatial_orbitals)

    @staticmethod
    def _apply_creation(bits: int, spin_orbital: int) -> tuple[int, int]:
        if (bits >> spin_orbital) & 1:
            return 0, bits
        phase = -1 if (bits & ((1 << spin_orbital) - 1)).bit_count() % 2 else 1
        return phase, bits | (1 << spin_orbital)

    @staticmethod
    def _apply_annihilation(bits: int, spin_orbital: int) -> tuple[int, int]:
        if not (bits >> spin_orbital) & 1:
            return 0, bits
        phase = -1 if (bits & ((1 << spin_orbital) - 1)).bit_count() % 2 else 1
        return phase, bits & ~(1 << spin_orbital)

    def _single_excitation_phase(self, bits: int, particle: int, hole: int) -> int:
        phase, bits = self._apply_annihilation(bits, hole)
        if phase == 0:
            return 0
        phase_create, bits = self._apply_creation(bits, particle)
        if phase_create == 0:
            return 0
        return phase * phase_create

    def _double_excitation_phase(
        self,
        bits: int,
        particle_p: int,
        particle_q: int,
        hole_r: int,
        hole_s: int,
    ) -> int:
        phase = 1

        step, bits = self._apply_annihilation(bits, hole_r)
        if step == 0:
            return 0
        phase *= step

        step, bits = self._apply_annihilation(bits, hole_s)
        if step == 0:
            return 0
        phase *= step

        step, bits = self._apply_creation(bits, particle_q)
        if step == 0:
            return 0
        phase *= step

        step, bits = self._apply_creation(bits, particle_p)
        if step == 0:
            return 0
        phase *= step

        return phase
