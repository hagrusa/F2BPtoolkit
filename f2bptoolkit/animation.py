"""Rendering and animation for F2BP simulation results."""

import os
import shutil
import subprocess
import tempfile
import textwrap
import numpy as np
from typing import Optional, Tuple


# ── mesh helpers ──────────────────────────────────────────────────────────────

class _EllipsoidMesh:
    """
    Pre-computed ellipsoid mesh in body-frame coordinates (meters).

    Stores both a structured UV grid (for matplotlib ``plot_surface``) and a
    flat vertex/face list (for VTK export and ``Poly3DCollection``).
    """

    def __init__(self, a: float, b: float, c: float,
                 n_lat: int = 24, n_lon: int = 48):
        phi   = np.linspace(0,        np.pi, n_lat + 1)
        theta = np.linspace(0, 2 * np.pi, n_lon + 1)
        P, T = np.meshgrid(phi, theta, indexing='ij')

        x0 = a * np.sin(P) * np.cos(T)
        y0 = b * np.sin(P) * np.sin(T)
        z0 = c * np.cos(P)

        self._grid_shape = P.shape                           # (n_lat+1, n_lon+1)
        self.vertices = np.stack(
            [x0.ravel(), y0.ravel(), z0.ravel()], axis=1)  # (M, 3)

        # Triangulated faces
        nl, nn = n_lat + 1, n_lon + 1
        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                p0 = i * nn + j;         p1 = i * nn + (j + 1)
                p2 = (i + 1) * nn + j;   p3 = (i + 1) * nn + (j + 1)
                faces.append([p0, p1, p3])
                faces.append([p0, p3, p2])
        self.faces = np.array(faces, dtype=int)             # (F, 3)

    def surface_grid(self, R: np.ndarray) -> Tuple:
        """Return (X, Y, Z) structured arrays rotated by R for ``plot_surface``."""
        v = (R @ self.vertices.T).T
        nl, nn = self._grid_shape
        return (v[:, 0].reshape(nl, nn),
                v[:, 1].reshape(nl, nn),
                v[:, 2].reshape(nl, nn))

    def rotated_verts(self, R: np.ndarray,
                      offset: Optional[np.ndarray] = None) -> np.ndarray:
        """Return (M, 3) vertices after rotation R and optional translation."""
        v = (R @ self.vertices.T).T
        return v if offset is None else v + offset


class _PolyhedronMesh:
    """Mesh loaded from GUBAS-format vertex/facet CSV files (meters)."""

    def __init__(self, vertex_file: str, facet_file: str):
        vert  = np.loadtxt(vertex_file, delimiter=',')
        facet = np.loadtxt(facet_file,  delimiter=',').astype(int)
        self.vertices = vert[:, 1:]        # drop 1-based id column; coords in m
        self.faces    = facet[:, :3] - 1  # 1-indexed → 0-indexed

    def rotated_verts(self, R: np.ndarray,
                      offset: Optional[np.ndarray] = None) -> np.ndarray:
        v = (R @ self.vertices.T).T
        return v if offset is None else v + offset


def _make_mesh(body):
    """Build a mesh object from a Body's shape model."""
    from .body import EllipsoidShape, SphereShape, PolyhedronShape
    shape = body.shape
    if shape is None:
        raise ValueError(f"Body '{body.name}' has no shape set.")
    if isinstance(shape, (SphereShape, EllipsoidShape)):
        a, b, c = shape.semi_axes()
        return _EllipsoidMesh(a, b, c)
    if isinstance(shape, PolyhedronShape):
        return _PolyhedronMesh(shape.vertex_file, shape.facet_file)
    raise ValueError(
        f"Cannot render shape type {type(shape).__name__}. "
        "Supported: SphereShape, EllipsoidShape, PolyhedronShape."
    )


# ── frame transforms ──────────────────────────────────────────────────────────

def _compute_N_to_O(A_to_N: np.ndarray,
                    r_A: np.ndarray,
                    v_A: np.ndarray) -> np.ndarray:
    """
    Compute the N→O rotation matrix (rows = orbital basis vectors in N).

    Orbital frame: x̂ = radial outward, ẑ = orbit angular momentum, ŷ = ẑ×x̂.
    """
    r_N = A_to_N @ r_A
    v_N = A_to_N @ v_A
    x_hat = r_N / np.linalg.norm(r_N)
    h     = np.cross(r_N, v_N)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return np.stack([x_hat, y_hat, z_hat])   # (3, 3)


def _body_rotations(A_to_N: np.ndarray, B_to_A: np.ndarray,
                    r_A: np.ndarray,    v_A: np.ndarray,
                    frame: str):
    """
    Return ``(prim_R, sec_R, sec_offset)`` for the requested display frame.

    *prim_R* and *sec_R* rotate body-frame column vectors into the display
    frame.  *sec_offset* is the secondary's centre position in that frame.
    """
    B_to_N    = A_to_N @ B_to_A
    sec_off_N = A_to_N @ r_A          # secondary centre in N

    if frame == 'inertial':
        return A_to_N, B_to_N, sec_off_N

    # Orbital frame
    N_to_O  = _compute_N_to_O(A_to_N, r_A, v_A)
    prim_R  = N_to_O @ A_to_N
    sec_R   = N_to_O @ B_to_N
    sec_off = N_to_O @ sec_off_N
    return prim_R, sec_R, sec_off


# ── VTK / ParaView export (no VTK dependency) ─────────────────────────────────

def _write_vtp(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Write a VTK XML PolyData (``.vtp``) file.

    No VTK installation required — the format is written as plain XML.
    """
    n_pts  = len(vertices)
    n_poly = len(faces)
    offsets = np.arange(3, 3 * n_poly + 1, 3, dtype=int)

    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        fh.write('  <PolyData>\n')
        fh.write(f'    <Piece NumberOfPoints="{n_pts}" NumberOfPolys="{n_poly}">\n')

        fh.write('      <Points>\n')
        fh.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        fh.write('         ')
        fh.write(' '.join(f'{v:.8g}' for v in vertices.ravel()))
        fh.write('\n        </DataArray>\n')
        fh.write('      </Points>\n')

        fh.write('      <Polys>\n')
        fh.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        fh.write('         ')
        fh.write(' '.join(str(i) for i in faces.ravel()))
        fh.write('\n        </DataArray>\n')
        fh.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        fh.write('         ')
        fh.write(' '.join(str(o) for o in offsets))
        fh.write('\n        </DataArray>\n')
        fh.write('      </Polys>\n')

        fh.write('    </Piece>\n')
        fh.write('  </PolyData>\n')
        fh.write('</VTKFile>\n')


def _write_pvd(path: str, timesteps, vtp_files) -> None:
    """Write a ParaView Data (``.pvd``) collection file."""
    pvd_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fh.write('  <Collection>\n')
        for t, vtp in zip(timesteps, vtp_files):
            rel = os.path.relpath(os.path.abspath(vtp), pvd_dir)
            fh.write(f'    <DataSet timestep="{t:.6g}" file="{rel}"/>\n')
        fh.write('  </Collection>\n')
        fh.write('</VTKFile>\n')


# ── pvpython helpers ──────────────────────────────────────────────────────────

def _find_pvpython(hint: Optional[str] = None) -> str:
    """Return the path to the pvpython executable."""
    if hint is not None:
        if os.path.isfile(hint) and os.access(hint, os.X_OK):
            return hint
        raise FileNotFoundError(f"pvpython not found at {hint!r}")

    candidates = [
        'pvpython',                                           # on PATH
        '/Applications/ParaView.app/Contents/bin/pvpython',  # macOS app bundle
        '/usr/bin/pvpython',
        '/usr/local/bin/pvpython',
        '/opt/ParaView/bin/pvpython',
    ]
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        "pvpython not found. Install ParaView and make sure pvpython is on your "
        "PATH, or pass pvpython='/path/to/pvpython'."
    )


def _write_pvpython_script(
    script_path: str,
    prim_pvd: str,
    sec_pvd: str,
    save_path: str,
    fps: float,
    image_resolution: Tuple[int, int],
    primary_color: Tuple[float, float, float],
    secondary_color: Tuple[float, float, float],
    background_color: Tuple[float, float, float],
    alpha: float,
    primary_name: str,
    secondary_name: str,
) -> None:
    """Write a pvpython script that loads the PVD time series and saves an MP4."""
    script = textwrap.dedent(f"""\
        from paraview.simple import *

        paraview.simple._DisableFirstRenderCameraReset()

        # ── Load time-series data ─────────────────────────────────────────────
        prim_reader = OpenDataFile({prim_pvd!r})
        RenameSource({primary_name!r}, prim_reader)
        sec_reader  = OpenDataFile({sec_pvd!r})
        RenameSource({secondary_name!r}, sec_reader)

        # ── Render view ───────────────────────────────────────────────────────
        view = GetActiveViewOrCreate('RenderView')
        view.ViewSize          = {list(image_resolution)!r}
        view.Background        = {list(background_color)!r}
        view.UseGradientBackground = 0

        # ── Primary body ──────────────────────────────────────────────────────
        prim_disp = Show(prim_reader, view)
        prim_disp.Representation = 'Surface'
        prim_disp.DiffuseColor   = {list(primary_color)!r}
        prim_disp.Specular        = 0.3
        prim_disp.SpecularPower   = 30.0
        prim_disp.Opacity         = {alpha!r}
        prim_disp.AmbientColor    = {list(primary_color)!r}

        # ── Secondary body ────────────────────────────────────────────────────
        sec_disp = Show(sec_reader, view)
        sec_disp.Representation = 'Surface'
        sec_disp.DiffuseColor   = {list(secondary_color)!r}
        sec_disp.Specular        = 0.3
        sec_disp.SpecularPower   = 30.0
        sec_disp.Opacity         = {alpha!r}
        sec_disp.AmbientColor    = {list(secondary_color)!r}

        # ── Camera ────────────────────────────────────────────────────────────
        Render()
        view.ResetCamera()
        Render()

        # ── Animation ─────────────────────────────────────────────────────────
        scene = GetAnimationScene()
        scene.UpdateAnimationUsingDataTimeSteps()

        # ── Save MP4 ──────────────────────────────────────────────────────────
        SaveAnimation(
            {save_path!r},
            view,
            FrameRate={fps!r},
            ImageResolution={list(image_resolution)!r},
        )
        print('Saved:', {save_path!r})
    """)
    with open(script_path, 'w') as fh:
        fh.write(script)


# ── AnimationInterface ────────────────────────────────────────────────────────

class AnimationInterface:
    """
    Rendering and animation methods for F2BP simulation results.

    Accessed as ``sim.animate`` after calling ``sim.integrate()``.

    Two renderers are available:

    * :meth:`matplotlib` — fast interactive 3-D animation using
      ``mpl_toolkits.mplot3d``.
    * :meth:`paraview` — exports a VTK time series (no VTK installation
      required) that can be opened in ParaView for high-quality rendering.
    """

    def __init__(self, results, bodies):
        self._r      = results
        self._bodies = bodies   # [primary, secondary]

    def _validate_frame(self, frame: str) -> None:
        if frame not in ('inertial', 'orbital'):
            raise ValueError(
                f"frame must be 'inertial' or 'orbital', got {frame!r}"
            )

    def _auto_stride(self, target: int = 300) -> int:
        """Stride that yields roughly *target* animation frames."""
        return max(1, self._r.n_steps // target)

    # ── matplotlib ────────────────────────────────────────────────────────────

    def matplotlib(
        self,
        frame: str = 'inertial',
        stride: Optional[int] = None,
        fps: float = 20.0,
        figsize: tuple = (8, 8),
        primary_color: str = 'royalblue',
        secondary_color: str = 'tomato',
        alpha: float = 0.80,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ):
        """
        Animate the simulation using matplotlib's 3-D engine.

        Ellipsoid bodies are drawn with ``plot_surface`` (structured grid,
        fast).  Polyhedron bodies are drawn with ``Poly3DCollection``.

        Parameters
        ----------
        frame : {'inertial', 'orbital'}
            Reference frame for the animation.
        stride : int, optional
            Step size through the timestep array.  Defaults to a value that
            gives ~300 animation frames.
        fps : float
            Frames per second for display and for saved files.
        figsize : tuple
            Matplotlib figure size ``(width, height)`` in inches.
        primary_color, secondary_color : str
            Face colours for the primary and secondary bodies.
        alpha : float
            Surface opacity in ``[0, 1]``.
        save_path : str, optional
            If given, save to this file instead of showing interactively.
            Format is inferred from the extension — ``.mp4`` requires
            ffmpeg, ``.gif`` requires Pillow.
        dpi : int
            Resolution for saved animations.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.animation import FuncAnimation

        self._validate_frame(frame)
        stride  = stride or self._auto_stride()
        r       = self._r
        indices = list(range(0, r.n_steps, stride))

        prim_mesh = _make_mesh(self._bodies[0])
        sec_mesh  = _make_mesh(self._bodies[1])

        # Fixed axis limits: max separation + largest body radius
        max_sep    = r.separation.max()
        max_r_body = max(
            np.linalg.norm(prim_mesh.vertices, axis=1).max(),
            np.linalg.norm(sec_mesh.vertices,  axis=1).max(),
        )
        lim = (max_sep + max_r_body) * 1.05

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        frame_label = ('Inertial Frame (N)' if frame == 'inertial'
                       else 'Orbital Frame (O)')
        ax.set_title(f'F2BP — {frame_label}')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(-lim, lim)

        time_text = ax.text2D(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=9, family='monospace',
        )
        # Legend proxies
        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color=primary_color,   label=self._bodies[0].name),
            mpatches.Patch(color=secondary_color, label=self._bodies[1].name),
        ], loc='upper right', fontsize=8)

        _surfaces = []   # surface/collection objects removed each frame

        def _draw_body(mesh, R, offset, color):
            if isinstance(mesh, _EllipsoidMesh):
                X, Y, Z = mesh.surface_grid(R)
                if offset is not None:
                    X, Y, Z = X + offset[0], Y + offset[1], Z + offset[2]
                surf = ax.plot_surface(
                    X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True,
                )
                return surf
            else:
                verts = mesh.rotated_verts(R, offset)
                tris  = verts[mesh.faces]      # (F, 3, 3) — fast numpy indexing
                col   = Poly3DCollection(
                    tris, facecolor=color, edgecolor='none', alpha=alpha,
                )
                ax.add_collection3d(col)
                return col

        def _update(idx):
            for obj in _surfaces:
                try:
                    obj.remove()
                except Exception:
                    pass
            _surfaces.clear()

            A_to_N = r.A_to_N[idx]
            B_to_A = r.B_to_A[idx]
            r_A    = r.position[idx]
            v_A    = r.velocity[idx]

            prim_R, sec_R, sec_off = _body_rotations(
                A_to_N, B_to_A, r_A, v_A, frame
            )
            _surfaces.append(_draw_body(prim_mesh, prim_R, None,    primary_color))
            _surfaces.append(_draw_body(sec_mesh,  sec_R,  sec_off, secondary_color))

            time_text.set_text(f't = {r.times[idx] / 86400.0:.4f} days')
            return _surfaces + [time_text]

        ani = FuncAnimation(
            fig, _update, frames=indices,
            blit=False, interval=1000.0 / fps,
        )

        if save_path:
            print(f"Saving animation to {save_path!r} ...")
            ani.save(save_path, fps=fps, dpi=dpi)
            plt.close(fig)
            print("Done.")
        else:
            plt.tight_layout()
            plt.show()

        return ani

    # ── ParaView MP4 rendering ────────────────────────────────────────────────

    def paraview(
        self,
        save_path: str,
        frame: str = 'inertial',
        stride: Optional[int] = None,
        fps: float = 20.0,
        image_resolution: Tuple[int, int] = (1920, 1080),
        primary_color: Tuple[float, float, float] = (0.27, 0.51, 0.71),
        secondary_color: Tuple[float, float, float] = (0.75, 0.23, 0.23),
        background_color: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        alpha: float = 1.0,
        keep_data: bool = False,
        data_dir: Optional[str] = None,
        pvpython: Optional[str] = None,
    ) -> str:
        """
        Render the simulation to an MP4 using ParaView's offscreen renderer.

        Writes VTK time-series files, generates a pvpython rendering script,
        and runs it via subprocess to produce the movie.  Requires a ParaView
        installation with ``pvpython`` on your PATH (or pass the path
        explicitly).

        Parameters
        ----------
        save_path : str
            Output MP4 file path (must end with ``.mp4``).
        frame : {'inertial', 'orbital'}
            Reference frame for the rendered geometry.
        stride : int, optional
            Step size through the timestep array.  Defaults to every step.
        fps : float
            Frames per second for the output video.
        image_resolution : (int, int)
            Width × height in pixels.
        primary_color : (R, G, B)
            Diffuse colour for the primary body, each component in ``[0, 1]``.
        secondary_color : (R, G, B)
            Diffuse colour for the secondary body.
        background_color : (R, G, B)
            Render view background colour.
        alpha : float
            Surface opacity in ``[0, 1]``.
        keep_data : bool
            If ``True``, keep the intermediate VTP/PVD files after rendering.
            If ``False`` (default) and *data_dir* was not provided, the
            temporary directory is deleted after the MP4 is written.
        data_dir : str, optional
            Directory for intermediate VTP/PVD files.  A temporary directory
            is used when not specified.
        pvpython : str, optional
            Explicit path to the ``pvpython`` executable.  Searched on PATH
            and in common installation locations when omitted.

        Returns
        -------
        str
            Absolute path to the saved MP4.

        Raises
        ------
        FileNotFoundError
            If ``pvpython`` cannot be found.
        RuntimeError
            If ``pvpython`` exits with a non-zero return code.
        """
        self._validate_frame(frame)
        stride    = stride or 1
        save_path = os.path.abspath(str(save_path))
        pvpy      = _find_pvpython(pvpython)

        # ── write VTP/PVD data ────────────────────────────────────────────────
        _tmp_dir   = None
        _own_dir   = data_dir is None
        if data_dir is None:
            _tmp_dir = tempfile.mkdtemp(prefix='f2bp_pv_')
            data_dir = _tmp_dir
        else:
            data_dir = os.path.abspath(str(data_dir))
            os.makedirs(data_dir, exist_ok=True)

        try:
            prim_dir = os.path.join(data_dir, 'primary')
            sec_dir  = os.path.join(data_dir, 'secondary')
            os.makedirs(prim_dir, exist_ok=True)
            os.makedirs(sec_dir,  exist_ok=True)

            r         = self._r
            prim_mesh = _make_mesh(self._bodies[0])
            sec_mesh  = _make_mesh(self._bodies[1])

            indices  = list(range(0, r.n_steps, stride))
            n_frames = len(indices)
            digits   = len(str(n_frames - 1))
            tick     = max(1, n_frames // 10)

            prim_vtps, sec_vtps, timesteps = [], [], []

            print(f"Writing {n_frames} frames → {data_dir!r} "
                  f"[frame='{frame}', stride={stride}]")

            for frame_num, idx in enumerate(indices):
                A_to_N_i = r.A_to_N[idx]
                B_to_A_i = r.B_to_A[idx]
                r_A      = r.position[idx]
                v_A      = r.velocity[idx]

                prim_R, sec_R, sec_off = _body_rotations(
                    A_to_N_i, B_to_A_i, r_A, v_A, frame
                )

                prim_verts = prim_mesh.rotated_verts(prim_R)
                sec_verts  = sec_mesh.rotated_verts(sec_R, sec_off)

                tag      = str(frame_num).zfill(digits)
                prim_vtp = os.path.join(prim_dir, f'primary_{tag}.vtp')
                sec_vtp  = os.path.join(sec_dir,  f'secondary_{tag}.vtp')

                _write_vtp(prim_vtp, prim_verts, prim_mesh.faces)
                _write_vtp(sec_vtp,  sec_verts,  sec_mesh.faces)

                prim_vtps.append(prim_vtp)
                sec_vtps.append(sec_vtp)
                timesteps.append(float(r.times[idx]))

                if (frame_num + 1) % tick == 0 or frame_num == n_frames - 1:
                    pct = 100 * (frame_num + 1) / n_frames
                    print(f"  {frame_num + 1:>{len(str(n_frames))}}/{n_frames}"
                          f"  ({pct:.0f}%)")

            prim_pvd = os.path.join(data_dir, 'primary.pvd')
            sec_pvd  = os.path.join(data_dir, 'secondary.pvd')
            _write_pvd(prim_pvd, timesteps, prim_vtps)
            _write_pvd(sec_pvd,  timesteps, sec_vtps)

            # ── write and run pvpython script ─────────────────────────────────
            script_path = os.path.join(data_dir, 'render.py')
            _write_pvpython_script(
                script_path    = script_path,
                prim_pvd       = prim_pvd,
                sec_pvd        = sec_pvd,
                save_path      = save_path,
                fps            = fps,
                image_resolution = image_resolution,
                primary_color  = primary_color,
                secondary_color= secondary_color,
                background_color= background_color,
                alpha          = alpha,
                primary_name   = self._bodies[0].name,
                secondary_name = self._bodies[1].name,
            )

            print(f"Launching pvpython: {pvpy}")
            proc = subprocess.run(
                [pvpy, script_path],
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                print(proc.stdout, end='')
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pvpython exited with code {proc.returncode}.\n"
                    f"stderr:\n{proc.stderr}"
                )

        finally:
            if _tmp_dir is not None and _own_dir and not keep_data:
                shutil.rmtree(_tmp_dir, ignore_errors=True)
            elif keep_data:
                print(f"Intermediate VTP/PVD files kept in: {data_dir!r}")

        print(f"Saved: {save_path!r}")
        return save_path
