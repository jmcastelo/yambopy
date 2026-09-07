"""
cell_to_ibrav_v01.py
====================
Class ``CellToIbrav`` — determine the Bravais lattice type (QE ibrav index)
and lattice parameters from a set of basis vectors.

Accepts input in **Quantum ESPRESSO convention**:
  - The 3×3 matrix has basis vectors as **rows** (matching QE's CELL_PARAMETERS
    input-card format).  If your data comes from QE's internal ``at`` array
    where vectors are **columns** (Fortran order), pass ``transpose=True``.
  - For each ibrav QE defines a specific orientation of the primitive vectors;
    this script recognises those orientations and returns the correct
    (possibly negative) ibrav index.

Dependencies: numpy only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Transformation matrices: conventional cell → primitive cell
# Each matrix P satisfies  A_prim = P @ A_conv  (basis vectors as rows).
# ---------------------------------------------------------------------------
_CENTERING_DATA: dict[str, tuple[np.ndarray, int, str]] = {
    "P":  (np.eye(3), 1, "primitive"),
    "I":  (np.array([[-0.5,  0.5,  0.5],
                     [ 0.5, -0.5,  0.5],
                     [ 0.5,  0.5, -0.5]], float), 2, "body-centred (QE ibrav=-3)"),
    "I1": (np.array([[ 0.5,  0.5,  0.5],
                     [-0.5,  0.5,  0.5],
                     [-0.5, -0.5,  0.5]], float), 2, "body-centred (QE ibrav=3)"),
    "I2": (np.array([[-0.5,  0.5,  0.5],
                     [ 0.5, -0.5,  0.5],
                     [ 0.5,  0.5, -0.5]], float), 2, "body-centred (synonym for I)"),
    "F":  (np.array([[ 0.0,  0.5,  0.5],
                     [ 0.5,  0.0,  0.5],
                     [ 0.5,  0.5,  0.0]], float), 4, "face-centred"),
    "C":  (np.array([[ 0.5,  0.5,  0.0],
                     [-0.5,  0.5,  0.0],
                     [ 0.0,  0.0,  1.0]], float), 2, "C-face-centred"),
    "A":  (np.array([[ 0.5,  0.0, -0.5],
                     [ 0.0,  1.0,  0.0],
                     [ 0.5,  0.0,  0.5]], float), 2, "A-face-centred"),
    "B":  (np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0,  0.5,  0.5],
                     [ 0.0, -0.5,  0.5]], float), 2, "B-face-centred"),
    "R":  (np.array([[-1/3,  2/3,  2/3],
                     [ 2/3, -1/3,  2/3],
                     [ 2/3,  2/3, -1/3]], float), 3, "rhombohedral"),
    "I_BCT": (np.array([[ 0.5, -0.5,  0.5],
                        [ 0.5,  0.5,  0.5],
                        [-0.5, -0.5,  0.5]], float), 2, "BCT (ibrav=7)"),
}

# ---------------------------------------------------------------------------
# ibrav table:  (crystal_system, centering, unique_axis)
# ---------------------------------------------------------------------------
_IBRAV_TABLE: dict[tuple[str, str, str | None], int] = {
    ("cubic",       "P",  None):   1,
    ("cubic",       "F",  None):   2,
    ("cubic",       "I",  None):  -3,
    ("cubic",       "I1", None):   3,
    ("cubic",       "I2", None):  -3,
    ("hexagonal",   "P",  None):   4,
    ("trigonal",    "R",  None):   5,
    ("tetragonal",  "P",  None):   6,
    ("tetragonal",  "I",  None):   7,
    ("tetragonal",  "I_BCT", None): 7,
    ("orthorhombic","P",  None):   8,
    ("orthorhombic","C",  None):   9,
    ("orthorhombic","A",  None):  -9,
    ("orthorhombic","F",  None):  10,
    ("orthorhombic","I",  None):  11,
    ("orthorhombic","I1", None):  11,
    ("monoclinic",  "P",  "c"):   12,
    ("monoclinic",  "P",  "b"):  -12,
    ("monoclinic",  "C",  "c"):   13,
    ("monoclinic",  "C",  "b"):  -13,
    ("monoclinic",  "A",  "c"):   13,
    ("monoclinic",  "A",  "b"):  -13,
    ("monoclinic",  "I",  "c"):   13,
    ("monoclinic",  "I",  "b"):  -13,
    ("triclinic",   "P",  None):  14,
}


def _rounded(v: np.ndarray, dec: int = 10) -> tuple:
    return tuple(np.round(v, decimals=dec).flat)


class CellToIbrav:
    """
    Determine the QE ibrav index and lattice parameters from a set of
    direct-lattice basis vectors.

    Parameters
    ----------
    cell : array-like, shape (3, 3)
        Basis vectors as **rows** (matching QE's CELL_PARAMETERS format).
        If the matrix has vectors as **columns** (QE's internal ``at``
        Fortran array), set ``transpose=True``.
    tol : float
        Relative tolerance for equality comparisons of lengths and angles
        (default 1e-5).
    transpose : bool
        If True, transposes the input so that vectors are read as columns
        (QE internal convention).  Default False.
    """

    def __init__(self, cell: Sequence, tol: float = 1e-5, transpose: bool = False):
        self.cell = np.asarray(cell, dtype=float)
        if self.cell.shape != (3, 3):
            raise ValueError("cell must be a 3x3 array.")
        if transpose:
            self.cell = self.cell.T
        self.tol = tol
        self._compute_parameters()
        self._determine_ibrav()
        self._conventional_cell = self._reconstruct_conventional(self._centering)
        self._compute_conventional_parameters()
        if self._ibrav == 5:
            self._a = self._a_prim
            self._b = self._b_prim
            self._c = self._c_prim
            self._alpha = self._alpha_prim
            self._beta = self._beta_prim
            self._gamma = self._gamma_prim
            self._alpha_deg = self._alpha_deg_prim
            self._beta_deg = self._beta_deg_prim
            self._gamma_deg = self._gamma_deg_prim
            self._volume = self._volume_prim
        self._pcell = self.cell.copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def ibrav(self) -> int:
        return self._ibrav

    @property
    def crystal_system(self) -> str:
        return self._crystal_system

    @property
    def centering(self) -> str:
        return self._centering

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def c(self) -> float:
        return self._c

    @property
    def alpha(self) -> float:
        return self._alpha_deg

    @property
    def beta(self) -> float:
        return self._beta_deg

    @property
    def gamma(self) -> float:
        return self._gamma_deg

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def primitive_cell(self) -> np.ndarray:
        return self._pcell

    def parameters_dict(self) -> dict[str, float]:
        return {
            "a": self._a,
            "b": self._b,
            "c": self._c,
            "alpha": self._alpha_deg,
            "beta": self._beta_deg,
            "gamma": self._gamma_deg,
        }

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "CellToIbrav summary",
            "=" * 56,
            f"ibrav              : {self._ibrav}",
            f"Crystal system     : {self._crystal_system}",
            f"Centering          : {self._centering}",
            "",
            f"a = {self._a:.6f}",
            f"b = {self._b:.6f}",
            f"c = {self._c:.6f}",
            f"\u03b1 = {self._alpha_deg:.4f}\u00b0",
            f"\u03b2 = {self._beta_deg:.4f}\u00b0",
            f"\u03b3 = {self._gamma_deg:.4f}\u00b0",
            f"Volume = {self._volume:.6f}",
            "",
            f"Primitive cell:\n{self._pcell}",
            "=" * 56,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CellToIbrav(ibrav={self._ibrav}, "
            f"system={self._crystal_system}, "
            f"centering={self._centering})"
        )

    # ------------------------------------------------------------------
    # Internal: compute cell parameters
    # ------------------------------------------------------------------

    def _compute_parameters(self):
        G = self.cell @ self.cell.T
        self._a = float(np.sqrt(G[0, 0]))
        self._b = float(np.sqrt(G[1, 1]))
        self._c = float(np.sqrt(G[2, 2]))
        self._alpha = float(np.arccos(G[1, 2] / (self._b * self._c)))
        self._beta = float(np.arccos(G[0, 2] / (self._a * self._c)))
        self._gamma = float(np.arccos(G[0, 1] / (self._a * self._b)))
        self._alpha_deg = float(np.degrees(self._alpha))
        self._beta_deg = float(np.degrees(self._beta))
        self._gamma_deg = float(np.degrees(self._gamma))
        self._volume = float(abs(np.linalg.det(self.cell)))
        self._a_prim = self._a
        self._b_prim = self._b
        self._c_prim = self._c
        self._alpha_prim = self._alpha
        self._beta_prim = self._beta
        self._gamma_prim = self._gamma
        self._alpha_deg_prim = self._alpha_deg
        self._beta_deg_prim = self._beta_deg
        self._gamma_deg_prim = self._gamma_deg
        self._volume_prim = self._volume

    def _compute_conventional_parameters(self):
        G = self._conventional_cell @ self._conventional_cell.T
        self._a = float(np.sqrt(G[0, 0]))
        self._b = float(np.sqrt(G[1, 1]))
        self._c = float(np.sqrt(G[2, 2]))
        self._alpha = float(np.arccos(G[1, 2] / (self._b * self._c)))
        self._beta = float(np.arccos(G[0, 2] / (self._a * self._c)))
        self._gamma = float(np.arccos(G[0, 1] / (self._a * self._b)))
        self._alpha_deg = float(np.degrees(self._alpha))
        self._beta_deg = float(np.degrees(self._beta))
        self._gamma_deg = float(np.degrees(self._gamma))
        self._volume = float(abs(np.linalg.det(self._conventional_cell)))

    # ------------------------------------------------------------------
    # Internal: comparison helpers
    # ------------------------------------------------------------------

    def _is_close(self, x: float, y: float) -> bool:
        denom = max(abs(x), abs(y), 1e-10)
        return abs(x - y) / denom < self.tol

    def _is_right(self, angle_rad: float) -> bool:
        return abs(angle_rad - np.pi / 2) < self.tol * 10

    def _is_120(self, angle_rad: float) -> bool:
        return abs(angle_rad - 2 * np.pi / 3) < self.tol * 10

    def _is_60(self, angle_rad: float) -> bool:
        return abs(angle_rad - np.pi / 3) < self.tol * 10

    # ------------------------------------------------------------------
    # Internal: classify a metric tensor into a crystal system
    # ------------------------------------------------------------------

    def _classify_metric(self, G: np.ndarray) -> str:
        a = float(np.sqrt(G[0, 0]))
        b = float(np.sqrt(G[1, 1]))
        c = float(np.sqrt(G[2, 2]))
        al = float(np.arccos(G[1, 2] / (b * c)))
        be = float(np.arccos(G[0, 2] / (a * c)))
        ga = float(np.arccos(G[0, 1] / (a * b)))

        ab = self._is_close(a, b)
        bc = self._is_close(b, c)
        ac = self._is_close(a, c)
        ral = self._is_right(al)
        rbe = self._is_right(be)
        rga = self._is_right(ga)

        if ab and bc and ac and ral and rbe and rga:
            return "cubic"
        if ab and ral and rbe and (self._is_120(ga) or self._is_60(ga)):
            return "hexagonal"
        if ab and bc and ac:
            return "trigonal"
        if ab and not bc and ral and rbe and rga:
            return "tetragonal"
        if not ab and not bc and not ac and ral and rbe and rga:
            return "orthorhombic"
        if ral and rbe and not rga:
            return "monoclinic"
        if ral and rga and not rbe:
            return "monoclinic"
        return "triclinic"

    # ------------------------------------------------------------------
    # Internal: try to reconstruct a conventional cell via centering
    # ------------------------------------------------------------------

    def _is_axis_aligned(self, v: np.ndarray) -> bool:
        """True if *v* has at least two near-zero components (i.e. lies
        along a Cartesian axis)."""
        near_zero = sum(1 for comp in v if abs(comp) < self.tol * 10)
        return near_zero >= 2

    def _conventional_is_standard(self, system: str, A_conv: np.ndarray) -> bool:
        """
        Check whether *A_conv* (rows = vectors) looks like a standard
        conventional cell for the given *system*.

        Axial systems (cubic, tetragonal, orthorhombic, hexagonal)
        require all vectors to be axis-aligned.
        Monoclinic requires the unique-axis vector to be axis-aligned and
        the other two to have the zero component consistent with that axis.
        Trigonal and triclinic have no axis-alignment requirement.
        """
        if system in ("cubic", "tetragonal", "orthorhombic", "hexagonal"):
            return all(self._is_axis_aligned(v) for v in A_conv)
        if system == "monoclinic":
            G = A_conv @ A_conv.T
            a = float(np.sqrt(G[0, 0]))
            b = float(np.sqrt(G[1, 1]))
            c = float(np.sqrt(G[2, 2]))
            al = float(np.arccos(G[1, 2] / (b * c)))
            be = float(np.arccos(G[0, 2] / (a * c)))
            ga = float(np.arccos(G[0, 1] / (a * b)))
            ral = self._is_right(al)
            rbe = self._is_right(be)
            rga = self._is_right(ga)
            axis = 0 if not ral and rbe and rga else (
                    1 if not rbe and ral and rga else (
                    2 if not rga and ral and rbe else None))
            if axis is None:
                return True
            zero_comp = axis
            for idx, v in enumerate(A_conv):
                if idx == axis:
                    if not self._is_axis_aligned(v):
                        return False
                else:
                    if abs(v[zero_comp]) > self.tol * 100:
                        return False
            return True
        return True

    def _n_axis_aligned(self, A: np.ndarray) -> int:
        return sum(1 for v in A if self._is_axis_aligned(v))

    def _try_centering(self) -> tuple[str, str]:
        """
        Try to identify the crystal system and centering.

        Returns (crystal_system, centering_label).

        1. Classify the input metric directly.
        2. If high-symmetry with ≥2 right angles, accept P immediately.
        3. Otherwise try all non-P centering transformations and pick
           the one that yields the most symmetric conventional cell
           with a valid (standard) form.
        4. If the best differs from the direct system → adopt it.
        5. If the best gives the *same* system as direct → keep the
           one whose conventional cell has more axis-aligned vectors
           (this distinguishes e.g. true C-centred monoclinic from
           a simple P cell).
        """
        G_prim = self.cell @ self.cell.T
        sys_direct = self._classify_metric(G_prim)

        n_right = sum(
            1 for ang in (self._alpha, self._beta, self._gamma)
            if self._is_right(ang)
        )
        if sys_direct in ("cubic", "hexagonal", "tetragonal", "orthorhombic"
                         ) and n_right >= 2:
            return sys_direct, "P"

        order = ("triclinic", "monoclinic", "orthorhombic",
                 "tetragonal", "hexagonal", "trigonal", "cubic")

        best_sys: str | None = None
        best_cent: str | None = None
        best_score = -1

        for centering in ("F", "I", "I1", "I_BCT", "C", "A", "B", "R"):
            P, _, _ = _CENTERING_DATA[centering]
            try:
                P_inv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                continue
            G_conv = P_inv @ G_prim @ P_inv.T
            sys_conv = self._classify_metric(G_conv)
            score = order.index(sys_conv) if sys_conv in order else -1
            A_conv = P_inv @ self.cell

            if not self._conventional_is_standard(sys_conv, A_conv):
                continue

            if score > best_score:
                best_score = score
                best_sys = sys_conv
                best_cent = centering

        if best_cent is None or best_sys is None:
            return sys_direct, "P"

        sys_direct_score = order.index(sys_direct) if sys_direct in order else -1
        if best_sys != sys_direct and best_score >= sys_direct_score:
            return best_sys, best_cent

        sys_direct_n_ax = self._n_axis_aligned(self.cell)
        best_A = self._reconstruct_conventional(best_cent)
        best_n_ax = self._n_axis_aligned(best_A)
        if best_n_ax > sys_direct_n_ax:
            return best_sys, best_cent

        return sys_direct, "P"

    def _classify_system(self, cell: np.ndarray | None = None) -> str:
        G = (cell @ cell.T) if cell is not None else (self.cell @ self.cell.T)
        return self._classify_metric(G)

    # ------------------------------------------------------------------
    # Internal: determine ibrav from centering analysis
    # ------------------------------------------------------------------

    def _reconstruct_conventional(self, centering: str) -> np.ndarray:
        """Return the conventional cell vectors for a given *centering*."""
        if centering == "P" or centering not in _CENTERING_DATA:
            return self.cell.copy()
        P, _, _ = _CENTERING_DATA[centering]
        P_inv = np.linalg.inv(P)
        return P_inv @ self.cell

    def _determine_ibrav(self):
        system, centering = self._try_centering()

        if system == "trigonal":
            centering = "R"

        dist_offdiag: dict[str, float] = {}
        def _offdiag(cent: str) -> float:
            if cent not in dist_offdiag:
                A = self._reconstruct_conventional(cent)
                dist_offdiag[cent] = abs(A[0, 1]) + abs(A[0, 2]) + abs(A[1, 2])
            return dist_offdiag[cent]

        if system == "cubic" and centering in ("I", "I1"):
            if _offdiag("I1") < _offdiag("I"):
                centering = "I1"
            else:
                centering = "I"

        if system == "orthorhombic" and centering in ("C", "A", "B"):
            for tag, key_center in [("C", "C"), ("A", "A")]:
                P, _, _ = _CENTERING_DATA[tag]
                try:
                    P_inv = np.linalg.inv(P)
                except np.linalg.LinAlgError:
                    continue
                G_conv = P_inv.T @ (self.cell @ self.cell.T) @ P_inv
                a_est = float(np.sqrt(G_conv[0, 0]))
                b_est = float(np.sqrt(G_conv[1, 1]))
                if self._is_close(a_est, self._a) and self._is_close(b_est, self._b):
                    centering = key_center
                    break

        uniq = None
        if system == "monoclinic":
            A_conv = self._reconstruct_conventional(centering)
            Gc = A_conv @ A_conv.T
            ac = float(np.sqrt(Gc[0, 0]))
            bc = float(np.sqrt(Gc[1, 1]))
            cc = float(np.sqrt(Gc[2, 2]))
            alpha_c = float(np.arccos(Gc[1, 2] / (bc * cc)))
            beta_c = float(np.arccos(Gc[0, 2] / (ac * cc)))
            gamma_c = float(np.arccos(Gc[0, 1] / (ac * bc)))
            ral_c = self._is_right(alpha_c)
            rbe_c = self._is_right(beta_c)
            rga_c = self._is_right(gamma_c)
            if not rga_c and rbe_c:
                uniq = "c"
            elif not rbe_c and rga_c:
                uniq = "b"
            else:
                uniq = "c"

        key = (system, centering, uniq)
        if key in _IBRAV_TABLE:
            self._ibrav = _IBRAV_TABLE[key]
        else:
            self._ibrav = 0

        self._crystal_system = system
        self._centering = centering


# =========================================================================
# Convenience: detect ibrav directly from lattice parameters
# =========================================================================

def ibrav_from_parameters(
    a: float,
    b: float | None = None,
    c: float | None = None,
    alpha_deg: float | None = None,
    beta_deg: float | None = None,
    gamma_deg: float | None = None,
    tol: float = 1e-5,
) -> int:
    """
    Determine the QE ibrav index from lattice parameters alone.

    This is a heuristic based purely on the cell geometry (lengths and
    angles).  It cannot distinguish between P, I, F or C for the same
    crystal family — for that use :class:`CellToIbrav` with the full
    cell matrix.

    Returns the **primitive** (P) ibrav for the given geometry, i.e. 1
    for cubic, 4 for hexagonal, 5 for trigonal, 6 for tetragonal,
    8 for orthorhombic, ±12 for monoclinic, 14 for triclinic.
    """
    if b is None:
        b = a
    if c is None:
        c = a
    if alpha_deg is None:
        alpha_deg = 90.0
    if beta_deg is None:
        beta_deg = 90.0
    if gamma_deg is None:
        gamma_deg = 90.0

    al = np.radians(alpha_deg)
    be = np.radians(beta_deg)
    ga = np.radians(gamma_deg)
    right = np.pi / 2

    def close(x, y):
        return abs(x - y) / max(abs(x), abs(y), 1e-10) < tol

    def is_right(v):
        return abs(v - right) < tol * 10

    if close(a, b) and close(b, c) and is_right(al) and is_right(be) and is_right(ga):
        return 1

    if close(a, b) and not close(b, c) and is_right(al) and is_right(be) and abs(ga - 2 * np.pi / 3) < tol * 10:
        return 4

    if close(a, b) and close(b, c) and not is_right(al):
        return 5

    if close(a, b) and not close(b, c) and is_right(al) and is_right(be) and is_right(ga):
        return 6

    if not close(a, b) and not close(b, c) and not close(a, c) and is_right(al) and is_right(be) and is_right(ga):
        return 8

    if is_right(al) and is_right(be) and not is_right(ga):
        return 12

    if is_right(al) and is_right(ga) and not is_right(be):
        return -12

    return 14
