from __future__ import annotations

import unittest

import numpy as np

from theochem2026 import BitstringHamiltonian


class BitstringHamiltonianTests(unittest.TestCase):
    def test_reference_uses_lowest_spatial_orbitals(self) -> None:
        model = BitstringHamiltonian(np.zeros((3, 3)), np.zeros((3, 3, 3, 3)))
        determinant = model.reference(n_alpha=2, n_beta=1)

        self.assertEqual(determinant.alpha, 0b011)
        self.assertEqual(determinant.beta, 0b001)
        self.assertEqual(determinant.as_bitstrings(3), ("011", "001"))

    def test_one_electron_hamiltonian_matches_single_particle_basis(self) -> None:
        h1 = np.array([[1.0, 0.25], [0.25, 2.0]])
        g2 = np.zeros((2, 2, 2, 2))
        model = BitstringHamiltonian(h1, g2)

        determinants = model.generate_determinants(n_alpha=1, n_beta=0)
        hamiltonian = model.hamiltonian(determinants)

        np.testing.assert_allclose(hamiltonian, h1)

    def test_diagonal_energy_includes_opposite_spin_coulomb_term(self) -> None:
        h1 = np.diag([1.0, 2.0])
        g2 = np.zeros((2, 2, 2, 2))
        g2[0, 0, 0, 0] = 0.7
        model = BitstringHamiltonian(h1, g2)

        determinant = model.determinant(alpha_occupied=[0], beta_occupied=[0])
        self.assertAlmostEqual(model.diagonal_energy(determinant), 2.7)

    def test_single_excitation_matrix_element_uses_off_diagonal_h1(self) -> None:
        h1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        g2 = np.zeros((2, 2, 2, 2))
        model = BitstringHamiltonian(h1, g2)

        left = model.determinant(alpha_occupied=[1])
        right = model.determinant(alpha_occupied=[0])

        self.assertAlmostEqual(model.matrix_element(left, right), 0.5)
        self.assertAlmostEqual(model.matrix_element(right, left), 0.5)


if __name__ == "__main__":
    unittest.main()
