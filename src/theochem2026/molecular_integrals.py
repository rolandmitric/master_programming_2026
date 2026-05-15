from dataclasses import dataclass, field
import copy

import numpy as np
from numba import njit
from numba.typed import List

from .atom import ELEMENT_SYMBOLS
from .basis_sets import BasisSet, Shell, l_to_ijk
from .integrals.S import S
from .integrals.T import T
from .integrals.V import V
from .integrals.ERI import ERI, ERI_KEY_TO_CASE
from .integrals.boys import Boys012, Boys01234

from .molecule import Molecule

ANG2BOHR = 1.8897261254578281


def dim_shell(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def build_ijk_lmn(la: int, lb: int) -> np.ndarray:
    ijk = l_to_ijk(la)
    lmn = l_to_ijk(lb)
    return np.asarray(
        [(i, j, k, l, m, n) for (i, j, k) in ijk for (l, m, n) in lmn],
        dtype=np.int64,
    )


def build_eri_indices(la: int, lb: int, lc: int, ld: int) -> np.ndarray:
    ijk = l_to_ijk(la)
    lmn = l_to_ijk(lb)
    opq = l_to_ijk(lc)
    rst = l_to_ijk(ld)
    return np.asarray(
        [a + b + c + d for a in ijk for b in lmn for c in opq for d in rst],
        dtype=np.int64,
    )


def build_eri_case_ids(ijkl_data: np.ndarray) -> np.ndarray:
    case_ids = np.empty(ijkl_data.shape[0], dtype=np.int64)
    for idx in range(ijkl_data.shape[0]):
        case_ids[idx] = ERI_KEY_TO_CASE[tuple(int(x) for x in ijkl_data[idx])]
    return case_ids


LMAX = 6
DIM_SHELL = {l: dim_shell(l) for l in range(LMAX + 1)}
LA_LB_TO_IJKLMN = {
    (la, lb): build_ijk_lmn(la, lb) for la in range(LMAX + 1) for lb in range(LMAX + 1)
}



@njit(fastmath=True)
def overlap_shell_pair(exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
                       exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray, ijk_lmn: np.ndarray,) -> np.ndarray:
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    out = np.zeros((dim_a, dim_b), dtype=np.float64)

    Dx = center_a[0] - center_b[0]
    Dy = center_a[1] - center_b[1]
    Dz = center_a[2] - center_b[2]
    D2 = Dx * Dx + Dy * Dy + Dz * Dz

    for idx in range(ijk_lmn.shape[0]):
        i, j, k, l, m, n = ijk_lmn[idx]
        ia = idx // dim_b
        ib = idx - ia * dim_b

        val = 0.0
        for pa in range(exp_a.shape[0]):
            alpha_a = exp_a[pa]
            ca = coeff_a[pa] * norm_a[pa, ia]
            for pb in range(exp_b.shape[0]):
                alpha_b = exp_b[pb]
                cb = coeff_b[pb] * norm_b[pb, ib]
                ab_sum = alpha_a + alpha_b
                ab_product = alpha_a * alpha_b
                val += ca * cb * S(i,j,k,l,m,n,Dx,Dy,Dz,D2,ab_sum,ab_product,alpha_a,alpha_b,)
        out[ia, ib] = val

    return out

@njit(fastmath=True)
def kinetic_shell_pair(exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
                       exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray, ijk_lmn: np.ndarray,) -> np.ndarray:
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    out = np.zeros((dim_a, dim_b), dtype=np.float64)

    Dx = center_a[0] - center_b[0]
    Dy = center_a[1] - center_b[1]
    Dz = center_a[2] - center_b[2]
    D2 = Dx * Dx + Dy * Dy + Dz * Dz

    for idx in range(ijk_lmn.shape[0]):
        i, j, k, l, m, n = ijk_lmn[idx]
        ia = idx // dim_b
        ib = idx - ia * dim_b

        val = 0.0
        for pa in range(exp_a.shape[0]):
            alpha_a = exp_a[pa]
            ca = coeff_a[pa] * norm_a[pa, ia]
            for pb in range(exp_b.shape[0]):
                alpha_b = exp_b[pb]
                cb = coeff_b[pb] * norm_b[pb, ib]
                ab_sum = alpha_a + alpha_b
                ab_product = alpha_a * alpha_b
                val += ca * cb * T(i,j,k,l,m,n,Dx,Dy,Dz,D2,ab_sum,ab_product,alpha_a,alpha_b,)
        out[ia, ib] = val

    return out

# assumes: from .integrals.V import V

@njit(fastmath=True)
def nuclear_shell_pair(
    exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
    exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray,
    center_c: np.ndarray,  # nucleus position C
    ijk_lmn: np.ndarray,
) -> np.ndarray:
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    out = np.zeros((dim_a, dim_b), dtype=np.float64)

    Ax, Ay, Az = center_a[0], center_a[1], center_a[2]
    Bx, By, Bz = center_b[0], center_b[1], center_b[2]
    Cx, Cy, Cz = center_c[0], center_c[1], center_c[2]

    ABx = Ax - Bx
    ABy = Ay - By
    ABz = Az - Bz
    RAB2 = ABx * ABx + ABy * ABy + ABz * ABz

    for pa in range(exp_a.shape[0]):
        alpha_a = exp_a[pa]
        coeff_pa = coeff_a[pa]

        for pb in range(exp_b.shape[0]):
            alpha_b = exp_b[pb]
            coeff_pb = coeff_b[pb]

            p = alpha_a + alpha_b
            KAB = np.exp(-(alpha_a * alpha_b / p) * RAB2)

            Qx = alpha_a * Ax + alpha_b * Bx - p * Cx
            Qy = alpha_a * Ay + alpha_b * By - p * Cy
            Qz = alpha_a * Az + alpha_b * Bz - p * Cz
            u = (Qx * Qx + Qy * Qy + Qz * Qz) / p

            F0, F1, F2 = Boys012(u)

            for idx in range(ijk_lmn.shape[0]):
                i, j, k, l, m, n = ijk_lmn[idx]
                ia = idx // dim_b
                ib = idx - ia * dim_b

                ca = coeff_pa * norm_a[pa, ia]
                cb = coeff_pb * norm_b[pb, ib]
                out[ia, ib] += ca * cb * V(
                    i, j, k, l, m, n,
                    ABx, ABy, ABz,
                    p, alpha_a, alpha_b,
                    Qx, Qy, Qz, F0, F1, F2, KAB
                )

    return out


@njit(fastmath=True)
def eri_shell_quartet(
    exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
    exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray,
    exp_c: np.ndarray, coeff_c: np.ndarray, norm_c: np.ndarray, center_c: np.ndarray,
    exp_d: np.ndarray, coeff_d: np.ndarray, norm_d: np.ndarray, center_d: np.ndarray,
    eri_case_data: np.ndarray,
) -> np.ndarray:
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    dim_c = norm_c.shape[1]
    dim_d = norm_d.shape[1]
    out = np.zeros((dim_a, dim_b, dim_c, dim_d), dtype=np.float64)

    Ax, Ay, Az = center_a[0], center_a[1], center_a[2]
    Bx, By, Bz = center_b[0], center_b[1], center_b[2]
    Cx, Cy, Cz = center_c[0], center_c[1], center_c[2]
    Dx, Dy, Dz = center_d[0], center_d[1], center_d[2]

    ABx = Ax - Bx
    ABy = Ay - By
    ABz = Az - Bz
    CDx = Cx - Dx
    CDy = Cy - Dy
    CDz = Cz - Dz
    rAB = ABx * ABx + ABy * ABy + ABz * ABz
    rCD = CDx * CDx + CDy * CDy + CDz * CDz

    stride_cd = dim_c * dim_d
    stride_bcd = dim_b * stride_cd

    for pa in range(exp_a.shape[0]):
        alpha = exp_a[pa]
        coeff_pa = coeff_a[pa]

        for pb in range(exp_b.shape[0]):
            beta = exp_b[pb]
            coeff_pb = coeff_b[pb]

            p = alpha + beta
            Px = (alpha * Ax + beta * Bx) / p
            Py = (alpha * Ay + beta * By) / p
            Pz = (alpha * Az + beta * Bz) / p
            KAB = np.exp(-(alpha * beta / p) * rAB)

            for pc in range(exp_c.shape[0]):
                gamma = exp_c[pc]
                coeff_pc = coeff_c[pc]

                for pd in range(exp_d.shape[0]):
                    delta = exp_d[pd]
                    coeff_pd = coeff_d[pd]

                    q = gamma + delta
                    Qx = (gamma * Cx + delta * Dx) / q
                    Qy = (gamma * Cy + delta * Dy) / q
                    Qz = (gamma * Cz + delta * Dz) / q
                    KCD = np.exp(-(gamma * delta / q) * rCD)

                    PQx = Px - Qx
                    PQy = Py - Qy
                    PQz = Pz - Qz
                    pRPQ = (p * q / (p + q)) * (PQx * PQx + PQy * PQy + PQz * PQz)
                    u = pRPQ
                    F0, F1, F2, F3, F4 = Boys01234(pRPQ)

                    for idx in range(eri_case_data.shape[0]):
                        ao_a = idx // stride_bcd
                        rem = idx - ao_a * stride_bcd
                        ao_b = rem // stride_cd
                        rem = rem - ao_b * stride_cd
                        ao_c = rem // dim_d
                        ao_d = rem - ao_c * dim_d

                        ca = coeff_pa * norm_a[pa, ao_a]
                        cb = coeff_pb * norm_b[pb, ao_b]
                        cc = coeff_pc * norm_c[pc, ao_c]
                        cd = coeff_pd * norm_d[pd, ao_d]

                        out[ao_a, ao_b, ao_c, ao_d] += ca * cb * cc * cd * ERI(
                            eri_case_data[idx],
                            Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz, Dx, Dy, Dz,
                            ABx, ABy, ABz, CDx, CDy, CDz,
                            p, q, alpha, beta, gamma, delta,
                            Px, Py, Pz, Qx, Qy, Qz,
                            PQx, PQy, PQz,
                            rAB, rCD, pRPQ, u, F0, F1, F2, F3, F4, KAB, KCD,
                        )

    return out



@njit(fastmath=True)
def overlap_matrix_driver(n_ao: int, pair_a: np.ndarray, pair_b: np.ndarray, 
                          offsets: np.ndarray, exp_data, coeff_data, norm_data, center_data, ijk_data,) -> np.ndarray:
c
    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a + 1]
        b0 = offsets[b]
        b1 = offsets[b + 1]

        block = overlap_shell_pair(exp_data[a],coeff_data[a],norm_data[a],center_data[a],
                                   exp_data[b],coeff_data[b],norm_data[b],center_data[b],ijk_data[p])

        S_ao[a0:a1, b0:b1] = block
        if a != b:
            S_ao[b0:b1, a0:a1] = block.T

    return S_ao

@njit(fastmath=True)
def kinetic_matrix_driver(n_ao: int, pair_a: np.ndarray, pair_b: np.ndarray, 
                          offsets: np.ndarray, exp_data, coeff_data, norm_data, center_data, ijk_data,) -> np.ndarray:
    T_ao = np.zeros((n_ao, n_ao), dtype=np.float64)

    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a + 1]
        b0 = offsets[b]
        b1 = offsets[b + 1]

        block = kinetic_shell_pair(exp_data[a],coeff_data[a],norm_data[a],center_data[a],
                                   exp_data[b],coeff_data[b],norm_data[b],center_data[b],ijk_data[p])

        T_ao[a0:a1, b0:b1] = block
        if a != b:
            T_ao[b0:b1, a0:a1] = block.T

    return T_ao

@njit(fastmath=True)
def nuclear_matrix_driver(
    n_ao: int,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    offsets: np.ndarray,
    exp_data, coeff_data, norm_data, center_data, ijk_data,
    nuc_coords: np.ndarray,   # shape (n_nuc, 3), in Bohr
    nuc_charges: np.ndarray,  # shape (n_nuc,)
) -> np.ndarray:
    V_ao = np.zeros((n_ao, n_ao), dtype=np.float64)

    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a + 1]
        b0 = offsets[b]
        b1 = offsets[b + 1]

        block = np.zeros((a1 - a0, b1 - b0), dtype=np.float64)

        # sum over nuclei: V = sum_C (-Z_C) * <a|1/r_C|b>
        for c in range(nuc_coords.shape[0]):
            block += -nuc_charges[c] * nuclear_shell_pair(
                exp_data[a], coeff_data[a], norm_data[a], center_data[a],
                exp_data[b], coeff_data[b], norm_data[b], center_data[b],
                nuc_coords[c], ijk_data[p]
            )

        V_ao[a0:a1, b0:b1] = block
        if a != b:
            V_ao[b0:b1, a0:a1] = block.T

    return V_ao


@njit(fastmath=True)
def eri_tensor_driver(
    n_ao: int,
    quartet_a: np.ndarray,
    quartet_b: np.ndarray,
    quartet_c: np.ndarray,
    quartet_d: np.ndarray,
    offsets: np.ndarray,
    exp_data, coeff_data, norm_data, center_data, eri_case_data,
    symmetrize: bool = True,
) -> np.ndarray:
    eri_ao = np.zeros((n_ao, n_ao, n_ao, n_ao), dtype=np.float64)

    for p in range(quartet_a.shape[0]):
        a = quartet_a[p]
        b = quartet_b[p]
        c = quartet_c[p]
        d = quartet_d[p]

        a0 = offsets[a]
        a1 = offsets[a + 1]
        b0 = offsets[b]
        b1 = offsets[b + 1]
        c0 = offsets[c]
        c1 = offsets[c + 1]
        d0 = offsets[d]
        d1 = offsets[d + 1]

        block = eri_shell_quartet(
            exp_data[a], coeff_data[a], norm_data[a], center_data[a],
            exp_data[b], coeff_data[b], norm_data[b], center_data[b],
            exp_data[c], coeff_data[c], norm_data[c], center_data[c],
            exp_data[d], coeff_data[d], norm_data[d], center_data[d],
            eri_case_data[p]
        )

        eri_ao[a0:a1, b0:b1, c0:c1, d0:d1] = block
        if symmetrize:
            eri_ao[b0:b1, a0:a1, c0:c1, d0:d1] = block.transpose(1, 0, 2, 3)
            eri_ao[a0:a1, b0:b1, d0:d1, c0:c1] = block.transpose(0, 1, 3, 2)
            eri_ao[b0:b1, a0:a1, d0:d1, c0:c1] = block.transpose(1, 0, 3, 2)
            eri_ao[c0:c1, d0:d1, a0:a1, b0:b1] = block.transpose(2, 3, 0, 1)
            eri_ao[d0:d1, c0:c1, a0:a1, b0:b1] = block.transpose(3, 2, 0, 1)
            eri_ao[c0:c1, d0:d1, b0:b1, a0:a1] = block.transpose(2, 3, 1, 0)
            eri_ao[d0:d1, c0:c1, b0:b1, a0:a1] = block.transpose(3, 2, 1, 0)

    return eri_ao

@dataclass
class MolecularIntegrals:
    molecule: Molecule
    basis_set: BasisSet
    shells: list[Shell] = field(init=False)
    dims: np.ndarray = field(init=False)
    offsets: np.ndarray = field(init=False)
    pair_a: np.ndarray = field(init=False)
    pair_b: np.ndarray = field(init=False)
    quartet_a: np.ndarray = field(init=False)
    quartet_b: np.ndarray = field(init=False)
    quartet_c: np.ndarray = field(init=False)
    quartet_d: np.ndarray = field(init=False)
    exp_data: object = field(init=False)
    coeff_data: object = field(init=False)
    norm_data: object = field(init=False)
    center_data: object = field(init=False)
    ijk_data: object = field(init=False)
    eri_case_data: object = field(init=False)

    def __post_init__(self) -> None:
        self.shells = []
        for atom in self.molecule.atoms:
            atom_symbol = ELEMENT_SYMBOLS[atom.atomic_number]
            for template_shell in self.basis_set.elements[atom_symbol]:
                shell = copy.deepcopy(template_shell)
                shell.set_center(np.asarray(atom.coord, dtype=np.float64) * ANG2BOHR)
                self.shells.append(shell)

        self.dims = np.asarray([DIM_SHELL[shell.l] for shell in self.shells], dtype=np.int64)
        self.offsets = np.zeros(len(self.shells) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.dims)

        self.exp_data = List()
        self.coeff_data = List()
        self.norm_data = List()
        self.center_data = List()
        for shell in self.shells:
            self.exp_data.append(np.ascontiguousarray(shell.exponents, dtype=np.float64))
            self.coeff_data.append(np.ascontiguousarray(shell.coefficients, dtype=np.float64))
            self.norm_data.append(np.ascontiguousarray(shell.norm_factors, dtype=np.float64))
            self.center_data.append(np.ascontiguousarray(shell.center, dtype=np.float64))

        pair_a_list = []
        pair_b_list = []
        self.ijk_data = List()
        for a, shell_a in enumerate(self.shells):
            for b in range(a + 1):
                pair_a_list.append(a)
                pair_b_list.append(b)
                ijk_lmn = LA_LB_TO_IJKLMN[(shell_a.l, self.shells[b].l)]
                self.ijk_data.append(np.ascontiguousarray(ijk_lmn, dtype=np.int64))

        self.pair_a = np.asarray(pair_a_list, dtype=np.int64)
        self.pair_b = np.asarray(pair_b_list, dtype=np.int64)

        quartet_a_list = []
        quartet_b_list = []
        quartet_c_list = []
        quartet_d_list = []
        self.eri_case_data = List()
        for p, a in enumerate(pair_a_list):
            b = pair_b_list[p]
            for q in range(p + 1):
                c = pair_a_list[q]
                d = pair_b_list[q]
                quartet_a_list.append(a)
                quartet_b_list.append(b)
                quartet_c_list.append(c)
                quartet_d_list.append(d)
                ijkl = build_eri_indices(self.shells[a].l, self.shells[b].l, self.shells[c].l, self.shells[d].l)
                case_ids = build_eri_case_ids(ijkl)
                self.eri_case_data.append(np.ascontiguousarray(case_ids, dtype=np.int64))

        self.quartet_a = np.asarray(quartet_a_list, dtype=np.int64)
        self.quartet_b = np.asarray(quartet_b_list, dtype=np.int64)
        self.quartet_c = np.asarray(quartet_c_list, dtype=np.int64)
        self.quartet_d = np.asarray(quartet_d_list, dtype=np.int64)

    @property
    def n_ao(self) -> int:
        return int(self.offsets[-1])

    def overlap_matrix(self) -> np.ndarray:
        return overlap_matrix_driver(self.n_ao, self.pair_a, self.pair_b, self.offsets, 
                                     self.exp_data, self.coeff_data,self.norm_data,self.center_data,self.ijk_data)

    def kinetic_matrix(self) -> np.ndarray:
        return kinetic_matrix_driver(self.n_ao, self.pair_a, self.pair_b, self.offsets, 
                                     self.exp_data, self.coeff_data,self.norm_data,self.center_data,self.ijk_data)


    def nuclear_attraction_matrix(self) -> np.ndarray:
        nuc_coords = np.asarray([atom.coord for atom in self.molecule.atoms], dtype=np.float64) * ANG2BOHR
        nuc_charges = np.asarray([atom.atomic_number for atom in self.molecule.atoms], dtype=np.float64)

        return nuclear_matrix_driver(
            self.n_ao, self.pair_a, self.pair_b, self.offsets,
            self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.ijk_data,
            nuc_coords, nuc_charges
        )

    def electron_repulsion_tensor(self, symmetrize: bool = True) -> np.ndarray:
        return eri_tensor_driver(
            self.n_ao, self.quartet_a, self.quartet_b, self.quartet_c, self.quartet_d, self.offsets,
            self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.eri_case_data, symmetrize,
        )
