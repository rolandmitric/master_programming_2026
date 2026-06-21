from dataclasses import dataclass, field
import copy
import numpy as np
from numba import njit
from numba.typed import List

from src_live import Molecule, BasisSet, ELEMENT_SYMBOL, Shell, S, T, V, ERI

ANG2BOHR = 1.8897259886
DIM_SHELL = {0: 1, 1: 3, 2: 6, 3: 10}
LA_LB_TO_IJKLMN = {(0, 0): [(0, 0, 0, 0, 0, 0)],
 (0, 1): [(0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1)],
 (0, 2): [(0, 0, 0, 2, 0, 0),
          (0, 0, 0, 1, 1, 0),
          (0, 0, 0, 1, 0, 1),
          (0, 0, 0, 0, 2, 0),
          (0, 0, 0, 0, 1, 1),
          (0, 0, 0, 0, 0, 2)],
 (1, 0): [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)],
 (1, 1): [(1, 0, 0, 1, 0, 0),
          (1, 0, 0, 0, 1, 0),
          (1, 0, 0, 0, 0, 1),
          (0, 1, 0, 1, 0, 0),
          (0, 1, 0, 0, 1, 0),
          (0, 1, 0, 0, 0, 1),
          (0, 0, 1, 1, 0, 0),
          (0, 0, 1, 0, 1, 0),
          (0, 0, 1, 0, 0, 1)],
 (1, 2): [(1, 0, 0, 2, 0, 0),
          (1, 0, 0, 1, 1, 0),
          (1, 0, 0, 1, 0, 1),
          (1, 0, 0, 0, 2, 0),
          (1, 0, 0, 0, 1, 1),
          (1, 0, 0, 0, 0, 2),
          (0, 1, 0, 2, 0, 0),
          (0, 1, 0, 1, 1, 0),
          (0, 1, 0, 1, 0, 1),
          (0, 1, 0, 0, 2, 0),
          (0, 1, 0, 0, 1, 1),
          (0, 1, 0, 0, 0, 2),
          (0, 0, 1, 2, 0, 0),
          (0, 0, 1, 1, 1, 0),
          (0, 0, 1, 1, 0, 1),
          (0, 0, 1, 0, 2, 0),
          (0, 0, 1, 0, 1, 1),
          (0, 0, 1, 0, 0, 2)],
 (2, 0): [(2, 0, 0, 0, 0, 0),
          (1, 1, 0, 0, 0, 0),
          (1, 0, 1, 0, 0, 0),
          (0, 2, 0, 0, 0, 0),
          (0, 1, 1, 0, 0, 0),
          (0, 0, 2, 0, 0, 0)],
 (2, 1): [(2, 0, 0, 1, 0, 0),
          (2, 0, 0, 0, 1, 0),
          (2, 0, 0, 0, 0, 1),
          (1, 1, 0, 1, 0, 0),
          (1, 1, 0, 0, 1, 0),
          (1, 1, 0, 0, 0, 1),
          (1, 0, 1, 1, 0, 0),
          (1, 0, 1, 0, 1, 0),
          (1, 0, 1, 0, 0, 1),
          (0, 2, 0, 1, 0, 0),
          (0, 2, 0, 0, 1, 0),
          (0, 2, 0, 0, 0, 1),
          (0, 1, 1, 1, 0, 0),
          (0, 1, 1, 0, 1, 0),
          (0, 1, 1, 0, 0, 1),
          (0, 0, 2, 1, 0, 0),
          (0, 0, 2, 0, 1, 0),
          (0, 0, 2, 0, 0, 1)],
 (2, 2): [(2, 0, 0, 2, 0, 0),
          (2, 0, 0, 1, 1, 0),
          (2, 0, 0, 1, 0, 1),
          (2, 0, 0, 0, 2, 0),
          (2, 0, 0, 0, 1, 1),
          (2, 0, 0, 0, 0, 2),
          (1, 1, 0, 2, 0, 0),
          (1, 1, 0, 1, 1, 0),
          (1, 1, 0, 1, 0, 1),
          (1, 1, 0, 0, 2, 0),
          (1, 1, 0, 0, 1, 1),
          (1, 1, 0, 0, 0, 2),
          (1, 0, 1, 2, 0, 0),
          (1, 0, 1, 1, 1, 0),
          (1, 0, 1, 1, 0, 1),
          (1, 0, 1, 0, 2, 0),
          (1, 0, 1, 0, 1, 1),
          (1, 0, 1, 0, 0, 2),
          (0, 2, 0, 2, 0, 0),
          (0, 2, 0, 1, 1, 0),
          (0, 2, 0, 1, 0, 1),
          (0, 2, 0, 0, 2, 0),
          (0, 2, 0, 0, 1, 1),
          (0, 2, 0, 0, 0, 2),
          (0, 1, 1, 2, 0, 0),
          (0, 1, 1, 1, 1, 0),
          (0, 1, 1, 1, 0, 1),
          (0, 1, 1, 0, 2, 0),
          (0, 1, 1, 0, 1, 1),
          (0, 1, 1, 0, 0, 2),
          (0, 0, 2, 2, 0, 0),
          (0, 0, 2, 1, 1, 0),
          (0, 0, 2, 1, 0, 1),
          (0, 0, 2, 0, 2, 0),
          (0, 0, 2, 0, 1, 1),
          (0, 0, 2, 0, 0, 2)]}

@njit(fastmath=True)
def overlap_shell_pair(
    exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
    exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray, ijklmn
):
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]

    Dx = center_a[0] - center_b[0]
    Dy = center_a[1] - center_b[1]
    Dz = center_a[2] - center_b[2]
    RAB2 = Dx**2 + Dy**2 + Dz**2

    out = np.zeros((dim_a, dim_b), dtype=np.float64)

    for pa in range(exp_a.shape[0]):
        alpha_a = exp_a[pa]
        for pb in range(exp_b.shape[0]):
            alpha_b = exp_b[pb]

            P = alpha_a + alpha_b
            Q = alpha_a * alpha_b
            KAB = np.exp(-Q * RAB2 / P)

            for idx in range(len(ijklmn)):
                i, j, k, l, m, n = ijklmn[idx]
                ia = idx // dim_b
                ib = idx - ia * dim_b

                c_a = coeff_a[pa] * norm_a[pa, ia]
                c_b = coeff_b[pb] * norm_b[pb, ib]

                out[ia, ib] += c_a * c_b * S.S(
                    i, j, k, l, m, n,
                    Dx, Dy, Dz,
                    KAB, P, Q, alpha_a, alpha_b
                )
    return out

@njit(fastmath=True)
def kinetic_shell_pair(exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
                       exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray, ijklmn):
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    Dx = center_a[0] - center_b[0]
    Dy = center_a[1] - center_b[1]
    Dz = center_a[2] - center_b[2]
    RAB2 = Dx**2 + Dy**2 + Dz**2
    out = np.zeros((dim_a, dim_b), dtype = np.float64)

    for idx in range(len(ijklmn)):
        i, j, k, l, m, n = ijklmn[idx]
        ia = idx // dim_b
        ib = idx - ia*dim_b
        val = 0
        for pa in range(exp_a.shape[0]):
            c_a = coeff_a[pa] * norm_a[pa, ia]
            alpha_a = exp_a[pa]
            for pb in range(exp_b.shape[0]):
                c_b = coeff_b[pb] * norm_b[pb, ib]
                alpha_b = exp_b[pb]
                P = alpha_a + alpha_b
                Q = alpha_a * alpha_b
                val += c_a*c_b*T.T(i,j,k,l,m,n,Dx,Dy,Dz,P,Q,RAB2,alpha_a,alpha_b)
        out[ia, ib] = val
    return out

@njit(fastmath=True)
def nuclear_attraction_shell_pair(exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
                       exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray, center_c, ijklmn):
    dim_a = norm_a.shape[1]
    dim_b = norm_b.shape[1]
    Ax, Ay, Az = center_a[0], center_a[1], center_a[2]
    Bx, By, Bz = center_b[0], center_b[1], center_b[2]
    Cx, Cy, Cz = center_c[0], center_c[1], center_c[2]
    Dx = Ax - Bx
    Dy = Ay - By
    Dz = Az - Bz
    RAB2 = Dx**2 + Dy**2 + Dz**2
    out = np.zeros((dim_a, dim_b), dtype = np.float64)

    for idx in range(len(ijklmn)):
        i, j, k, l, m, n = ijklmn[idx]
        ia = idx // dim_b
        ib = idx - ia*dim_b
        val = 0
        for pa in range(exp_a.shape[0]):
            c_a = coeff_a[pa] * norm_a[pa, ia]
            alpha_a = exp_a[pa]
            for pb in range(exp_b.shape[0]):
                c_b = coeff_b[pb] * norm_b[pb, ib]
                alpha_b = exp_b[pb]
                P = alpha_a + alpha_b
                Q = alpha_a * alpha_b
                
                KAB = np.exp(-Q*RAB2/P)
                Qx = alpha_a * Ax + alpha_b * Bx - P*Cx
                Qy = alpha_a * Ay + alpha_b * By - P*Cy
                Qz = alpha_a * Az + alpha_b * Bz - P*Cz
                u = (Qx**2 + Qy**2 + Qz**2) / P
                val += c_a*c_b*V.V(i,j,k,l,m,n,Dx,Dy,Dz,KAB, P, Q, Qx, Qy, Qz, alpha_a, alpha_b, u)
        out[ia, ib] = val
    return out

@njit(cache = True, fastmath=True)
def eri_shell_quartet(
    exp_a: np.ndarray, coeff_a: np.ndarray, norm_a: np.ndarray, center_a: np.ndarray,
    exp_b: np.ndarray, coeff_b: np.ndarray, norm_b: np.ndarray, center_b: np.ndarray,
    exp_c: np.ndarray, coeff_c: np.ndarray, norm_c: np.ndarray, center_c: np.ndarray,
    exp_d: np.ndarray, coeff_d: np.ndarray, norm_d: np.ndarray, center_d: np.ndarray,
    ijklmnopqrst: np.ndarray,
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

    RAB2 = ABx * ABx + ABy * ABy + ABz * ABz
    RCD2 = CDx * CDx + CDy * CDy + CDz * CDz

    stride_cd = dim_c * dim_d
    stride_bcd = dim_b * stride_cd

    for idx in range(ijklmnopqrst.shape[0]):
        ao_a = idx // stride_bcd
        rem = idx - ao_a * stride_bcd
        ao_b = rem // stride_cd
        rem = rem - ao_b * stride_cd
        ao_c = rem // dim_d
        ao_d = rem - ao_c * dim_d

        i = ijklmnopqrst[idx, 0]
        j = ijklmnopqrst[idx, 1]
        k = ijklmnopqrst[idx, 2]
        l = ijklmnopqrst[idx, 3]
        m = ijklmnopqrst[idx, 4]
        n = ijklmnopqrst[idx, 5]
        o = ijklmnopqrst[idx, 6]
        p = ijklmnopqrst[idx, 7]
        q = ijklmnopqrst[idx, 8]
        r = ijklmnopqrst[idx, 9]
        s = ijklmnopqrst[idx, 10]
        t = ijklmnopqrst[idx, 11]

        val = 0.0

        for pa in range(exp_a.shape[0]):
            alpha = exp_a[pa]
            ca = coeff_a[pa] * norm_a[pa, ao_a]

            for pb in range(exp_b.shape[0]):
                beta = exp_b[pb]
                cb = coeff_b[pb] * norm_b[pb, ao_b]

                P = alpha + beta
                Px = (alpha * Ax + beta * Bx) / P
                Py = (alpha * Ay + beta * By) / P
                Pz = (alpha * Az + beta * Bz) / P
                KAB = np.exp(-(alpha * beta / P) * RAB2)

                for pc in range(exp_c.shape[0]):
                    gamma = exp_c[pc]
                    cc = coeff_c[pc] * norm_c[pc, ao_c]

                    for pd in range(exp_d.shape[0]):
                        delta = exp_d[pd]
                        cd = coeff_d[pd] * norm_d[pd, ao_d]

                        Q = gamma + delta
                        Qx = (gamma * Cx + delta * Dx) / Q
                        Qy = (gamma * Cy + delta * Dy) / Q
                        Qz = (gamma * Cz + delta * Dz) / Q
                        KCD = np.exp(-(gamma * delta / Q) * RCD2)

                        PQx = Px - Qx
                        PQy = Py - Qy
                        PQz = Pz - Qz
                        RPQ2 = PQx * PQx + PQy * PQy + PQz * PQz
                        rho = (P * Q) / (P + Q)

                        val += ca * cb * cc * cd * ERI.ERI(
                            i, j, k, l, m, n, o, p, q, r, s, t,
                            ABx, ABy, ABz, CDx, CDy, CDz,
                            KAB, KCD, P, PQx, PQy, PQz, Q, RPQ2,
                            alpha, beta, delta, gamma, rho,
                        )

        out[ao_a, ao_b, ao_c, ao_d] = val

    return out



@njit(fastmath=True)
def overlap_matrix_driver(n_ao: int, pair_a: np.ndarray, pair_b: np.ndarray, 
                          offsets: np.ndarray, exp_data, coeff_data, norm_data,
                          center_data, ijk_data):
    
    S_ao = np.zeros((n_ao, n_ao), dtype = np.float64)

    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a+1]

        b0 = offsets[b]
        b1 = offsets[b+1]

        block = overlap_shell_pair(exp_data[a], coeff_data[a], norm_data[a], center_data[a],
                           exp_data[b], coeff_data[b], norm_data[b], center_data[b], ijk_data[p])
        
        S_ao [a0:a1, b0:b1] = block
        if a!=b:
            S_ao [b0:b1, a0:a1] = block.T
    return S_ao

@njit(fastmath=True)
def kinetic_matrix_driver(n_ao: int, pair_a: np.ndarray, pair_b: np.ndarray, 
                          offsets: np.ndarray, exp_data, coeff_data, norm_data,
                          center_data, ijk_data):
    
    T_ao = np.zeros((n_ao, n_ao), dtype = np.float64)

    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a+1]

        b0 = offsets[b]
        b1 = offsets[b+1]

        block = kinetic_shell_pair(exp_data[a], coeff_data[a], norm_data[a], center_data[a],
                           exp_data[b], coeff_data[b], norm_data[b], center_data[b], ijk_data[p])
        
        T_ao [a0:a1, b0:b1] = block
        if a!=b:
            T_ao [b0:b1, a0:a1] = block.T
    return T_ao

@njit(fastmath=True)
def nuclear_attraction_matrix_driver(n_ao: int, pair_a: np.ndarray, pair_b: np.ndarray, 
                          offsets: np.ndarray, exp_data, coeff_data, norm_data,
                          center_data, ijk_data, nuc_coords, nuc_charges):
    
    V_ao = np.zeros((n_ao, n_ao), dtype = np.float64)

    for p in range(pair_a.shape[0]):
        a = pair_a[p]
        b = pair_b[p]

        a0 = offsets[a]
        a1 = offsets[a+1]

        b0 = offsets[b]
        b1 = offsets[b+1]
        
        block = np.zeros((a1 - a0, b1 - b0), dtype = np.float64)

        for c in range(nuc_coords.shape[0]):
            block -= nuc_charges[c] * nuclear_attraction_shell_pair(exp_data[a], coeff_data[a], norm_data[a], center_data[a], exp_data[b], coeff_data[b], norm_data[b], center_data[b], nuc_coords[c],  ijk_data[p])
        
        V_ao [a0:a1, b0:b1] = block
        if a!=b:
            V_ao [b0:b1, a0:a1] = block.T
    return V_ao

@njit(fastmath=True)
def eri_tensor_driver(
    n_ao: int,
    quartet_a: np.ndarray,
    quartet_b: np.ndarray,
    quartet_c: np.ndarray,
    quartet_d: np.ndarray,
    offsets: np.ndarray,
    exp_data, coeff_data, norm_data, center_data, ijklmnopqrst_data,
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
            ijklmnopqrst_data[p]
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
    shells: list[Shell] = field(init = False)

    def __post_init__(self):
        self.shells = []
        for atom in self.molecule.atoms:
            atom_symbol = ELEMENT_SYMBOL[atom.atomic_number]
            for template_shell in self.basis_set.elements[atom_symbol]:
                shell = copy.deepcopy(template_shell)
                shell.set_center(np.asarray(atom.coord * ANG2BOHR, dtype=np.float64))
                self.shells.append(shell)

        self.dims = np.asarray([DIM_SHELL[shell.l] for shell in self.shells], dtype=np.int64)
        self.offsets = np.zeros((len(self.shells) + 1), dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.dims)

        self.exp_data = List()
        self.coeff_data = List()
        self.norm_data = List()
        self.center_data = List()
        self.ijk_data = List()

        for shell in self.shells:
            self.exp_data.append(np.ascontiguousarray(shell.exponents, dtype=np.float64))
            self.coeff_data.append(np.ascontiguousarray(shell.coefficients, dtype=np.float64))
            self.norm_data.append(np.ascontiguousarray(shell.norm_factors, dtype=np.float64))
            self.center_data.append(np.ascontiguousarray(shell.center, dtype=np.float64))

        pair_a_list = []
        pair_b_list = []

        for a, shell_a in enumerate(self.shells):
            for b in range(a + 1):
                pair_a_list.append(a)
                pair_b_list.append(b)
                ijk_lmn = LA_LB_TO_IJKLMN[(self.shells[a].l, self.shells[b].l)]
                self.ijk_data.append(np.ascontiguousarray(ijk_lmn, dtype=np.int64))

        self.pair_a_list = np.ascontiguousarray(pair_a_list, dtype=np.int64)
        self.pair_b_list = np.ascontiguousarray(pair_b_list, dtype=np.int64)

        quartet_a_list = []
        quartet_b_list = []
        quartet_c_list = []
        quartet_d_list = []

        self.ijklmnopqrst_data = List()

        for p, a in enumerate(pair_a_list):
            b = pair_b_list[p]

            for q in range(p + 1):
                c = pair_a_list[q]
                d = pair_b_list[q]

                quartet_a_list.append(a)
                quartet_b_list.append(b)
                quartet_c_list.append(c)
                quartet_d_list.append(d)

                ab_data = LA_LB_TO_IJKLMN[(self.shells[a].l, self.shells[b].l)]
                cd_data = LA_LB_TO_IJKLMN[(self.shells[c].l, self.shells[d].l)]

                ijklmnopqrst = [
                    ab + cd
                    for ab in ab_data
                    for cd in cd_data
                ]

                self.ijklmnopqrst_data.append(
                    np.ascontiguousarray(ijklmnopqrst, dtype=np.int64)
                )

        self.quartet_a = np.asarray(quartet_a_list, dtype=np.int64)
        self.quartet_b = np.asarray(quartet_b_list, dtype=np.int64)
        self.quartet_c = np.asarray(quartet_c_list, dtype=np.int64)
        self.quartet_d = np.asarray(quartet_d_list, dtype=np.int64)

    @property
    def n_ao(self)->int:
        return int(self.offsets[-1])

    def overlap_matrix(self) -> np.ndarray:
        return overlap_matrix_driver(self.n_ao, self.pair_a_list, self.pair_b_list, self.offsets,
                                 self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.ijk_data)

    def kinetic_matrix(self) -> np.ndarray:
        return kinetic_matrix_driver(self.n_ao, self.pair_a_list, self.pair_b_list, self.offsets,
                                 self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.ijk_data)
    
    def nuclear_attraction_matrix(self) -> np.ndarray:
        nuc_coords = np.asarray([atom.coord * ANG2BOHR for atom in self.molecule.atoms])
        nuc_charges = np.asarray([atom.atomic_number for atom in self.molecule.atoms])

        return nuclear_attraction_matrix_driver(self.n_ao, self.pair_a_list, self.pair_b_list, self.offsets,
                                 self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.ijk_data,
                                 nuc_coords, nuc_charges)

    def electron_repulsion_tensor(self, symmetrize: bool = True) -> np.ndarray:
        return eri_tensor_driver(
            self.n_ao, self.quartet_a, self.quartet_b, self.quartet_c, self.quartet_d, self.offsets,
            self.exp_data, self.coeff_data, self.norm_data, self.center_data, self.ijklmnopqrst_data, symmetrize,
        )
