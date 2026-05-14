"""Body and shape model definitions."""

import os
import tempfile
import numpy as np
from typing import Optional, Tuple


class ShapeModel:
    """Base class for body shape models."""
    pass


class SphereShape(ShapeModel):
    """Uniform-density sphere."""

    def __init__(self, radius: float):
        """
        Parameters
        ----------
        radius : float
            Radius in meters.
        """
        self.radius = float(radius)

    def semi_axes(self) -> Tuple[float, float, float]:
        return (self.radius, self.radius, self.radius)

    def __repr__(self):
        return f"SphereShape(radius={self.radius} m)"


class EllipsoidShape(ShapeModel):
    """Uniform-density tri-axial ellipsoid."""

    def __init__(self, a: float, b: float, c: float):
        """
        Parameters
        ----------
        a : float
            Semi-major axis in meters.
        b : float
            Semi-intermediate axis in meters.
        c : float
            Semi-minor axis in meters.
        """
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def semi_axes(self) -> Tuple[float, float, float]:
        return (self.a, self.b, self.c)

    def __repr__(self):
        return f"EllipsoidShape(a={self.a}, b={self.b}, c={self.c} m)"


class PolyhedronShape(ShapeModel):
    """
    Triangular polyhedron shape model.

    Two construction paths:

    1. **CSV files** (GUBAS convention) — pass ``vertex_file`` and
       ``facet_file`` directly::

           shape = PolyhedronShape("verts.csv", "facets.csv")

       Expected formats (comma-delimited):
         - *vertex_file* : ``id, x_m, y_m, z_m``  (1-based id column,
           coordinates in **meters**)
         - *facet_file*  : ``v1, v2, v3``  (1-based vertex indices,
           triangles only)

    2. **Wavefront OBJ** — use the classmethod::

           shape = PolyhedronShape.from_obj("model.obj")

       Vertex coordinates must be in **meters**.  Only triangular faces
       (``f v1 v2 v3``) are supported.  Quad/n-gon faces raise an error.

    Attributes
    ----------
    vertex_file : str
        Path to the (possibly temporary) vertex CSV file.
    facet_file : str
        Path to the (possibly temporary) facet CSV file.
    """

    def __init__(self, vertex_file: str, facet_file: str):
        """
        Parameters
        ----------
        vertex_file : str
            Path to a comma-delimited CSV file with columns
            ``[id, x_m, y_m, z_m]`` (1-based id; coordinates in meters).
        facet_file : str
            Path to a comma-delimited CSV file with columns
            ``[v1, v2, v3]`` (1-based vertex indices; triangles only).
        """
        self.vertex_file = str(vertex_file)
        self.facet_file  = str(facet_file)
        self._tmp_files: list = []   # temp files created by from_obj(); none here
        _validate_csv_files(self.vertex_file, self.facet_file)

    @classmethod
    def from_obj(cls, path: str, scale: float = 1.0) -> 'PolyhedronShape':
        """
        Construct a PolyhedronShape by parsing a Wavefront OBJ file.

        Parameters
        ----------
        path : str
            Path to the ``.obj`` file.  Only triangular faces are supported.
        scale : float, optional
            Multiply all vertex coordinates by this factor before passing to
            the integrator.  The integrator expects coordinates in **meters**,
            so set ``scale=1000.0`` if the OBJ file uses kilometres.
            Default: 1.0 (coordinates already in metres).

        Returns
        -------
        PolyhedronShape

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the OBJ file is malformed, contains non-triangular faces,
            out-of-range face indices, or has fewer than 4 vertices/faces.
        """
        vertices, faces = _parse_obj(path)
        if scale != 1.0:
            vertices = vertices * scale
        _validate_mesh_arrays(vertices, faces, source=path)
        vpath, fpath = _write_gubas_csv(vertices, faces)

        obj = cls.__new__(cls)
        obj.vertex_file = vpath
        obj.facet_file  = fpath
        obj._tmp_files  = [vpath, fpath]
        return obj

    def __del__(self):
        """Remove any temporary CSV files created by from_obj()."""
        for p in getattr(self, '_tmp_files', []):
            try:
                os.unlink(p)
            except OSError:
                pass

    def __repr__(self):
        src = ' (from OBJ)' if self._tmp_files else ''
        return f"PolyhedronShape(vert='{self.vertex_file}', facet='{self.facet_file}'{src})"


# ── Polyhedron helper functions ───────────────────────────────────────────────

def _validate_csv_files(vertex_file: str, facet_file: str) -> None:
    """Validate GUBAS-format vertex and facet CSV files."""
    # ── existence ─────────────────────────────────────────────────────────────
    for path, label in [(vertex_file, 'vertex_file'), (facet_file, 'facet_file')]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PolyhedronShape: {label} not found: {path!r}")

    # ── vertex file ───────────────────────────────────────────────────────────
    try:
        vert = np.loadtxt(vertex_file, delimiter=',')
    except Exception as exc:
        raise ValueError(
            f"Cannot parse vertex_file {vertex_file!r} as CSV: {exc}"
        ) from exc

    vert = np.atleast_2d(vert)
    if vert.shape[1] != 4:
        raise ValueError(
            f"vertex_file must have exactly 4 columns [id, x, y, z], "
            f"got {vert.shape[1]} in {vertex_file!r}"
        )
    if not np.all(np.isfinite(vert[:, 1:])):
        raise ValueError(
            f"vertex_file contains NaN or Inf coordinate values: {vertex_file!r}"
        )
    n_verts = len(vert)
    if n_verts < 4:
        raise ValueError(
            f"vertex_file has only {n_verts} vertices; a closed polyhedron "
            f"needs at least 4"
        )

    # ── facet file ────────────────────────────────────────────────────────────
    try:
        facet = np.loadtxt(facet_file, delimiter=',')
    except Exception as exc:
        raise ValueError(
            f"Cannot parse facet_file {facet_file!r} as CSV: {exc}"
        ) from exc

    facet = np.atleast_2d(facet)
    if facet.shape[1] < 3:
        raise ValueError(
            f"facet_file must have at least 3 columns [v1, v2, v3], "
            f"got {facet.shape[1]} in {facet_file!r}"
        )
    n_faces = len(facet)
    if n_faces < 4:
        raise ValueError(
            f"facet_file has only {n_faces} faces; a closed polyhedron "
            f"needs at least 4"
        )
    idx = facet[:, :3].astype(int)
    if idx.min() < 1:
        raise ValueError(
            f"facet_file uses 1-based vertex indices but found index "
            f"{idx.min()!r}; indices must be >= 1"
        )
    if idx.max() > n_verts:
        raise ValueError(
            f"facet_file vertex index {idx.max()} exceeds the number of "
            f"vertices ({n_verts}); check that indices are 1-based"
        )


def _validate_mesh_arrays(vertices: np.ndarray, faces: np.ndarray,
                           source: str = '') -> None:
    """Validate in-memory vertex/face arrays (0-based face indices)."""
    tag = f" in {source!r}" if source else ''

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertex array must be shape (N, 3){tag}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"Vertex coordinates contain NaN or Inf{tag}")
    n_verts = len(vertices)
    if n_verts < 4:
        raise ValueError(
            f"Only {n_verts} vertices found{tag}; a closed polyhedron needs at least 4"
        )

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Face array must be shape (M, 3) — only triangular meshes are "
            f"supported{tag}"
        )
    n_faces = len(faces)
    if n_faces < 4:
        raise ValueError(
            f"Only {n_faces} faces found{tag}; a closed polyhedron needs at least 4"
        )

    lo, hi = faces.min(), faces.max()
    if lo < 0 or hi >= n_verts:
        bad = np.argwhere((faces < 0) | (faces >= n_verts))[0]
        raise ValueError(
            f"Face index out of range at face {bad[0]}, column {bad[1]}{tag}. "
            f"Expected 0-based indices in [0, {n_verts - 1}], "
            f"got {faces[bad[0], bad[1]]}"
        )


def _parse_obj(path: str):
    """
    Parse a Wavefront OBJ file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)  — coordinates in whatever units the OBJ uses
    faces    : ndarray, shape (M, 3)  — 0-based vertex indices
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OBJ file not found: {path!r}")

    vertices = []
    faces    = []

    with open(path, 'r') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            token = parts[0].lower()

            if token == 'v':
                if len(parts) < 4:
                    raise ValueError(
                        f"OBJ {path!r} line {lineno}: 'v' expects 3 coordinates, "
                        f"got {len(parts) - 1}: {line!r}"
                    )
                try:
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    raise ValueError(
                        f"OBJ {path!r} line {lineno}: cannot parse vertex "
                        f"coordinates: {line!r}"
                    )
                vertices.append(xyz)

            elif token == 'f':
                n_refs = len(parts) - 1
                if n_refs != 3:
                    raise ValueError(
                        f"OBJ {path!r} line {lineno}: only triangular faces are "
                        f"supported, but this face has {n_refs} vertices. "
                        f"Triangulate the mesh before importing."
                    )
                idx = []
                for vref in parts[1:]:
                    raw_i = vref.split('/')[0]   # strip /vt/vn suffixes
                    try:
                        i = int(raw_i)
                    except ValueError:
                        raise ValueError(
                            f"OBJ {path!r} line {lineno}: cannot parse face "
                            f"index {vref!r}"
                        )
                    if i == 0:
                        raise ValueError(
                            f"OBJ {path!r} line {lineno}: vertex index 0 is not "
                            f"valid in OBJ format (indices start at 1)"
                        )
                    # OBJ: positive = absolute 1-based, negative = relative to
                    # current vertex count
                    if i > 0:
                        idx.append(i - 1)
                    else:
                        idx.append(len(vertices) + i)   # negative relative index
                faces.append(idx)

    if not vertices:
        raise ValueError(f"No vertex ('v') lines found in OBJ file {path!r}")
    if not faces:
        raise ValueError(f"No face ('f') lines found in OBJ file {path!r}")

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def _write_gubas_csv(vertices: np.ndarray, faces: np.ndarray):
    """
    Write GUBAS-format CSV files from in-memory vertex/face arrays.

    Vertex file: ``[1-based_id, x_m, y_m, z_m]``
    Facet  file: ``[v1, v2, v3]``  (1-based; C++ does ``tet -= 1``)

    Returns
    -------
    vertex_path : str
    facet_path  : str
    """
    n_verts = len(vertices)

    # ── vertex CSV ────────────────────────────────────────────────────────────
    vfd, vpath = tempfile.mkstemp(suffix='_verts.csv', prefix='f2bp_')
    ffd, fpath = tempfile.mkstemp(suffix='_facets.csv', prefix='f2bp_')
    try:
        ids  = np.arange(1, n_verts + 1, dtype=float).reshape(-1, 1)
        vert_data = np.hstack([ids, vertices])          # [id, x, y, z]
        with os.fdopen(vfd, 'w') as vfh:
            np.savetxt(vfh, vert_data, delimiter=',', fmt='%.10g')

        face_data = faces + 1                           # 0-based → 1-based
        with os.fdopen(ffd, 'w') as ffh:
            np.savetxt(ffh, face_data, delimiter=',', fmt='%d')
    except Exception:
        try:
            os.unlink(vpath)
        except OSError:
            pass
        try:
            os.unlink(fpath)
        except OSError:
            pass
        raise

    return vpath, fpath


def _poly_com_and_inertia(vertices: np.ndarray, faces: np.ndarray):
    """
    Compute the centre of mass and inertia tensor for a uniform-density
    polyhedron using signed-tetrahedron decomposition.

    Density is implicitly 1; all results scale linearly with rho.

    Returns
    -------
    com : ndarray, shape (3,)
        Centre of mass in the same units as *vertices*.
    I_com : ndarray, shape (3, 3)
        Symmetric inertia tensor about the COM (rho = 1).
    """
    a = vertices[faces[:, 0]]   # (F, 3)
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]

    # Signed volume of each tet: a · (b × c)
    det = np.einsum('ij,ij->i', a, np.cross(b, c))   # (F,)
    total_vol = det.sum() / 6.0

    # COM: weighted centroid (4th vertex at origin → centroid = (a+b+c)/4)
    com = np.einsum('i,ij->j', det, a + b + c) / (24.0 * total_vol)

    # Second moments about origin via tetrahedral formula:
    #   ∫ x² dV = Σ det/60 * (ax² + bx² + cx² + ax·bx + ax·cx + bx·cx)
    #   ∫ xy dV = Σ det/120 * (2ax·ay + 2bx·by + 2cx·cy +
    #                           ax·by + ay·bx + ax·cy + ay·cx + bx·cy + by·cx)
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]

    def _sq(u, v, w):
        return float(np.dot(det, u**2 + v**2 + w**2 + u*v + u*w + v*w)) / 60.0

    def _cross(u1, u2, v1, v2, w1, w2):
        return float(np.dot(det, 2*u1*u2 + 2*v1*v2 + 2*w1*w2 +
                            u1*v2 + u2*v1 + u1*w2 + u2*w1 +
                            v1*w2 + v2*w1)) / 120.0

    int_x2 = _sq(ax, bx, cx)
    int_y2 = _sq(ay, by, cy)
    int_z2 = _sq(az, bz, cz)
    int_xy = _cross(ax, ay, bx, by, cx, cy)
    int_xz = _cross(ax, az, bx, bz, cx, cz)
    int_yz = _cross(ay, az, by, bz, cy, cz)

    # Inertia tensor about origin
    I_orig = np.array([
        [ int_y2 + int_z2, -int_xy,           -int_xz          ],
        [-int_xy,           int_x2 + int_z2,  -int_yz          ],
        [-int_xz,          -int_yz,            int_x2 + int_y2 ],
    ])

    # Parallel-axis theorem: I_com = I_orig - M*(|com|²·I₃ - com⊗com)
    M = total_vol   # rho = 1
    I_com = I_orig - M * (np.dot(com, com) * np.eye(3) - np.outer(com, com))

    return com, I_com


def _align_polyhedron(shape: 'PolyhedronShape'):
    """
    Check whether a polyhedral shape is centred on its COM and aligned with
    its principal axes of inertia.  If not, return a corrected shape.

    The corrected shape has vertices translated to the COM and rotated so that
    the principal axes align with the body-frame axes in ascending moment order
    (x = minimum moment / long axis, z = maximum moment / spin axis), consistent
    with the GUBAS ellipsoid convention.

    Returns
    -------
    shape : PolyhedronShape
        Original shape if already aligned, otherwise a new temporary
        PolyhedronShape whose CSV files contain the corrected vertices.
    msg : str or None
        Human-readable description of corrections applied, or None if none
        were needed.
    """
    vert_data = np.loadtxt(shape.vertex_file, delimiter=',')
    face_data = np.loadtxt(shape.facet_file,  delimiter=',').astype(int)
    vertices  = vert_data[:, 1:]        # drop 1-based id column (metres)
    faces     = face_data[:, :3] - 1    # 1-based → 0-based

    com, I_com = _poly_com_and_inertia(vertices, faces)

    R_char         = np.linalg.norm(vertices, axis=1).mean()
    com_offset     = float(np.linalg.norm(com))
    diag_mean      = float(np.abs(np.diag(I_com)).mean())
    max_offdiag_rel = float(np.abs([I_com[0, 1], I_com[0, 2], I_com[1, 2]]).max()) \
                      / (diag_mean + 1e-30)

    com_bad  = com_offset > 1e-3 * R_char
    axes_bad = max_offdiag_rel > 1e-4

    if not com_bad and not axes_bad:
        return shape, None

    msgs = []
    v = vertices - com
    if com_bad:
        msgs.append(f"translated to COM (offset was {com_offset:.4g} m)")

    # Eigendecomposition — ascending eigenvalue order: I_xx ≤ I_yy ≤ I_zz
    eigenvalues, eigenvectors = np.linalg.eigh(I_com)
    order       = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1   # enforce right-handed frame

    # Express vertices in the principal-axis frame: v_new = Qᵀ v
    v_aligned = (eigenvectors.T @ v.T).T

    if axes_bad:
        max_angle = float(np.degrees(
            np.arccos(np.clip(np.abs(np.diag(eigenvectors)), 0.0, 1.0))
        ).max())
        msgs.append(
            f"rotated to principal axes "
            f"(max axis deviation {max_angle:.2f}°, "
            f"off-diagonal/diagonal = {max_offdiag_rel:.2e})"
        )

    vpath, fpath = _write_gubas_csv(v_aligned, faces)
    new_shape             = PolyhedronShape.__new__(PolyhedronShape)
    new_shape.vertex_file = vpath
    new_shape.facet_file  = fpath
    new_shape._tmp_files  = [vpath, fpath]
    return new_shape, "; ".join(msgs)


class Body:
    """
    One body in a binary system.

    Example
    -------
    >>> b = Body("Dimorphos")
    >>> b.shape = EllipsoidShape(a=85, b=73, c=63)  # meters
    >>> b.density = 2400.0   # kg/m^3
    >>> b.inertia_order = 2
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "Body"
        self._shape: Optional[ShapeModel] = None
        self._density: Optional[float] = None  # kg/m^3
        self._inertia_order: int = 2

    # ── shape ─────────────────────────────────────────────────────────────────

    @property
    def shape(self) -> Optional[ShapeModel]:
        """Shape model (SphereShape, EllipsoidShape, or PolyhedronShape)."""
        return self._shape

    @shape.setter
    def shape(self, s: ShapeModel):
        if not isinstance(s, ShapeModel):
            raise TypeError(f"Expected a ShapeModel, got {type(s).__name__}")
        self._shape = s

    # ── density ───────────────────────────────────────────────────────────────

    @property
    def density(self) -> Optional[float]:
        """Bulk density in kg/m^3."""
        return self._density

    @density.setter
    def density(self, d: float):
        d = float(d)
        if d <= 0:
            raise ValueError("Density must be positive")
        self._density = d

    # ── inertia expansion order ───────────────────────────────────────────────

    @property
    def inertia_order(self) -> int:
        """
        Truncation order for the inertia integral expansion (must be even).
        Higher orders are more accurate but slower. Default: 2.
        """
        return self._inertia_order

    @inertia_order.setter
    def inertia_order(self, n: int):
        n = int(n)
        if n < 0 or n % 2 != 0:
            raise ValueError("inertia_order must be a non-negative even integer (0, 2, 4, …)")
        self._inertia_order = n

    # ── internal helpers (used by Simulation) ────────────────────────────────

    def _shape_flag(self) -> int:
        """Return the C++ shape flag: 0=sphere, 1=ellipsoid, 2=polyhedron."""
        if isinstance(self._shape, SphereShape):
            return 0
        elif isinstance(self._shape, EllipsoidShape):
            return 1
        elif isinstance(self._shape, PolyhedronShape):
            return 2
        raise ValueError(f"Unsupported shape type: {type(self._shape)}")

    def _semi_axes_km(self) -> Tuple[float, float, float]:
        """Semi-axes in km (C++ internal units). Returns (0,0,0) for polyhedra."""
        if isinstance(self._shape, (SphereShape, EllipsoidShape)):
            a, b, c = self._shape.semi_axes()
            return a / 1000.0, b / 1000.0, c / 1000.0
        return 0.0, 0.0, 0.0

    def _density_kg_km3(self) -> float:
        """Density in kg/km^3 (C++ internal units)."""
        # 1 kg/m^3 = 1e9 kg/km^3
        return self._density * 1.0e9

    def validate(self):
        """Raise ValueError if the body is not fully configured."""
        if self._shape is None:
            raise ValueError(f"Body '{self.name}': shape not set")
        if self._density is None:
            raise ValueError(f"Body '{self.name}': density not set")

    def __repr__(self):
        return (f"Body(name='{self.name}', shape={self._shape}, "
                f"density={self._density} kg/m³, inertia_order={self._inertia_order})")
