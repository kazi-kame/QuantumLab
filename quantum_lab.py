import pygame
import sys
import threading
import numpy as np
import matplotlib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

WIDTH, HEIGHT = 1440, 720
SIDEBAR_W = 320
SIM_W = WIDTH - SIDEBAR_W
SIM_H = 520
BOTTOM_H = HEIGHT - SIM_H
FPS = 60

BG = (13, 13, 26)
PANEL = (18, 18, 31)
PANEL2 = (24, 24, 42)
BORDER = (45, 45, 80)
TEXT = (210, 210, 235)
TEXT_DIM = (110, 110, 150)
ACCENT = (80, 160, 255)
ACCENT2 = (80, 230, 160)
ACCENT3 = (255, 180, 60)
ACCENT4 = (220, 80, 80)
SCAR_COL = (255, 120, 40)
WHITE = (255, 255, 255)

pygame.font.init()
def mono(size, bold=False):
    for name in ["JetBrains Mono","Courier New","Lucida Console","monospace"]:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f: return f
        except Exception:
            pass
    return pygame.font.SysFont(None, size, bold=bold)

FONT_SM = mono(12)
FONT_MD = mono(14)
FONT_LG = mono(16, bold=True)
FONT_XL = mono(20, bold=True)

CMAPS = ["inferno","plasma","viridis","twilight","hot"]

LATEX_MAP = {
    r"\psi": "\u03c8",
    r"\hbar": "\u0127",
    r"\sigma": "\u03c3",
    r"\mu": "\u03bc",
    r"\pi": "\u03c0",
    r"\int": "\u222b",
    r"\sum": "\u03a3",
    r"\in": "\u2208",
    r"\to": "\u2192",
    r"\cycle": "\u27f3",
    r"\sqrt": "\u221a",
    r"^2": "\u00b2",
    r"^4": "\u2074",
    r"_0": "\u2080",
    r"_n": "\u2099",
    r"_y": "\u1d67", 
    r"~": "\u0303", 
    r"<": "\u27e8",
    r">": "\u27e9",
}

def tex(text):
    for k, v in LATEX_MAP.items():
        text = text.replace(k, v)
    return text

class Slider:
    def __init__(self, x, y, w, mn, mx, val, label, fmt=".2f", unit=""):
        self.rect = pygame.Rect(x, y, w, 4)
        self.min = mn
        self.max = mx
        self.value = val
        self.label = label
        self.fmt = fmt
        self.unit = unit
        self.dragging = False

    def draw(self, screen):
        pygame.draw.rect(screen, PANEL2, self.rect, border_radius=2)
        t = (self.value - self.min) / (self.max - self.min)
        fr = pygame.Rect(self.rect.x, self.rect.y, int(t * self.rect.width), 4)
        pygame.draw.rect(screen, ACCENT, fr, border_radius=2)
        hx = self.rect.x + int(t * self.rect.width)
        hy = self.rect.centery
        pygame.draw.circle(screen, ACCENT, (hx, hy), 7)
        pygame.draw.circle(screen, BG, (hx, hy), 3)
        val_str = format(self.value, self.fmt)
        label_s = FONT_SM.render(tex(self.label), True, TEXT_DIM)
        val_s = FONT_MD.render(f"{val_str}{tex(self.unit)}", True, TEXT)
        screen.blit(label_s, (self.rect.x, self.rect.y - 18))
        screen.blit(val_s, (self.rect.right - val_s.get_width(), self.rect.y - 18))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            hit = pygame.Rect(self.rect.x-8, self.rect.y-8,
                              self.rect.width+16, self.rect.height+16)
            if hit.collidepoint(event.pos):
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            rel = max(0, min(event.pos[0] - self.rect.x, self.rect.width))
            self.value = self.min + (rel / self.rect.width) * (self.max - self.min)
            return True
        return False

class Engine:
    def __init__(self):
        self.a = 0.6
        self.N = 60
        self.sigma_target = 200
        self.num_states = 60
        self.dt = 0.01

        self.vals = None
        self.vecs = None
        self.mask = None
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None

        self.coeffs = None
        self.psi0 = None
        self.psi_full = None
        self.time = 0.0

        self.eig_ipr = None
        self.revival_times = {}

        self.ipr_series = []
        self.auto_series = []

        self.is_solving = False

    def inside(self, X, Y):
        B = 1.0
        in_rect = (np.abs(X) <= self.a) & (np.abs(Y) <= B)
        in_left = (X + self.a)**2 + Y**2 <= B**2
        in_right = (X - self.a)**2 + Y**2 <= B**2
        return in_rect | in_left | in_right

    def build(self):
        B = 1.0
        nx = int(2*(self.a + B)*self.N) + 1
        ny = int(2*B*self.N) + 1
        x = np.linspace(-(self.a+B), (self.a+B), nx)
        y = np.linspace(-B, B, ny)
        dx = x[1]-x[0]; dy = y[1]-y[0]
        X, Y = np.meshgrid(x, y)
        mask = self.inside(X, Y)
        idx = np.full(mask.shape, -1, int)
        idx[mask] = np.arange(mask.sum())

        rows, cols, data = [], [], []
        rows_m, cols_m = np.where(mask)
        for r, c in zip(rows_m, cols_m):
            i = idx[r, c]
            rows.append(i); cols.append(i)
            data.append(-2/dx**2 - 2/dy**2)
            for dr, dc, h in [(0,1,dx),(0,-1,dx),(1,0,dy),(-1,0,dy)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                    j = idx[nr, nc]
                    rows.append(i); cols.append(j); data.append(1/h**2)
        L = csr_matrix((data,(rows,cols)), shape=(mask.sum(), mask.sum()))
        return L, mask, x, y, dx, dy

    def solve(self):
        self.is_solving = True
        try:
            L, mask, x, y, dx, dy = self.build()
            vals, vecs = eigsh(-L, k=self.num_states,
                               sigma=self.sigma_target, which='LM')
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]

            norms = np.sqrt(np.sum(vecs**2, axis=0) * dx * dy)
            vecs = vecs / norms[np.newaxis, :]

            eig_ipr = np.sum(vecs**4, axis=0) * dx * dy

            self._compute_revival_times(vals)

            self.vals = vals
            self.vecs = vecs
            self.mask = mask
            self.x = x
            self.y = y
            self.dx = dx
            self.dy = dy
            self.eig_ipr = eig_ipr

        except Exception as e:
            print(f"[solve error] {e}")
        self.is_solving = False

    def _compute_revival_times(self, vals):
        n = len(vals)
        n0 = n // 2
        rt = {}

        if n >= 3:
            dE1 = (vals[min(n0+1,n-1)] - vals[max(n0-1,0)]) / 2.0
            if abs(dE1) > 1e-10:
                rt['T_cl'] = 2*np.pi / abs(dE1)

        if n >= 5:
            dE2 = vals[min(n0+1,n-1)] - 2*vals[n0] + vals[max(n0-1,0)]
            if abs(dE2) > 1e-10:
                rt['T_rev'] = 2*np.pi / (abs(dE2)/2.0)
                for q in [2, 3, 4]:
                    rt[f'T_1/{q}'] = rt['T_rev'] / q

        if n >= 2:
            dE_mean = (vals[-1] - vals[0]) / (n - 1)
            if dE_mean > 1e-10:
                rt['T_H'] = 2*np.pi / dE_mean

        self.revival_times = rt

    def start(self):
        if not self.is_solving:
            threading.Thread(target=self.solve, daemon=True).start()

    def initialize_packet(self, x0, y0, px, py, width=0.2):
        if self.vecs is None:
            return
        mask = self.mask; x = self.x; y = self.y
        vecs = self.vecs; dx = self.dx; dy = self.dy

        X, Y = np.meshgrid(x, y)
        psi0 = (np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*width**2))
                * np.exp(1j*(px*X + py*Y)))
        psi0[~mask] = 0.0

        norm = np.sqrt(np.sum(np.abs(psi0[mask])**2) * dx * dy)
        if norm < 1e-14:
            return
        psi0 /= norm

        psi0_flat = psi0[mask]
        self.coeffs = np.conj(vecs).T @ psi0_flat * dx * dy

        self.psi0 = psi0_flat
        self.psi_full = None
        self.time = 0.0
        self.ipr_series.clear()
        self.auto_series.clear()

    def evolve(self):
        if self.coeffs is None:
            return None
        
        # Guard against race conditions during re-solve
        if self.mask is None or self.vecs is None:
            return None
            
        phases = np.exp(-1j * self.vals * self.time)
        try:
            psi_flat = self.vecs @ (self.coeffs * phases)
        except ValueError:
            # Dimension mismatch during update
            return None

        # Reconstruct 2D wavefunction
        # Check compatibility with current mask
        if psi_flat.shape[0] != np.sum(self.mask):
            return None

        psi = np.zeros(self.mask.shape, dtype=complex)
        psi[self.mask] = psi_flat
        self.psi_full = psi

        dens = np.abs(psi)**2

        self.ipr_series.append(
            float(np.sum(np.abs(psi_flat)**4) * self.dx * self.dy))

        # Check psi0 compatibility before dot product
        if self.psi0 is not None and self.psi0.shape == psi_flat.shape:
            auto_val = np.abs(np.dot(np.conj(self.psi0), psi_flat)
                            * self.dx * self.dy)**2
            self.auto_series.append(float(auto_val))

        self.time += self.dt
        return dens

    def husimi(self, psi_flat, grid_n=36):
        if psi_flat is None or self.vals is None:
            return None
        E_mean = float(np.mean(self.vals))
        sigma = 1.0 / max(np.sqrt(2.0*E_mean), 0.1)
        B = 1.0

        x0s = np.linspace(-(self.a+B)*0.75, (self.a+B)*0.75, grid_n)
        p0s = np.linspace(-np.sqrt(max(2*E_mean,1))*1.2,
                           np.sqrt(max(2*E_mean,1))*1.2, grid_n)

        Xg, Yg = np.meshgrid(self.x, self.y)
        xf = Xg[self.mask]
        yf = Yg[self.mask]

        inv2s2 = 1.0 / (4.0 * sigma**2)
        norm_c = (2*np.pi*sigma**2)**(-0.5)

        Q = np.zeros((grid_n, grid_n))

        for j, x0v in enumerate(x0s):
            dx_arr = xf - x0v
            gauss = norm_c * np.exp(-inv2s2 * (dx_arr**2 + yf**2))
            phases = np.exp(1j * p0s[:, np.newaxis] * dx_arr[np.newaxis, :])
            alpha = gauss[np.newaxis, :] * phases
            overlaps = (np.conj(alpha) @ psi_flat) * self.dx * self.dy
            Q[:, j] = np.abs(overlaps)**2 / np.pi

        return Q, x0s, p0s

    def wigner_marginal(self, psi_flat, n_p=128):
        if psi_flat is None or self.mask is None:
            return None

        psi_2d = np.zeros(self.mask.shape, dtype=complex)
        psi_2d[self.mask] = psi_flat
        psi_y = np.sum(psi_2d, axis=1) * self.dx

        ny = len(psi_y)
        Psi_k = np.fft.fft(psi_y, n=2*ny)
        C = np.fft.ifft(np.abs(Psi_k)**2).real * self.dy

        W_fft = np.fft.fftshift(np.fft.fft(C[:ny])).real / np.pi
        E_mean = float(np.mean(self.vals)) if self.vals is not None else 100.0
        p_max = np.sqrt(max(2*E_mean, 1.0)) * 1.5
        py_full = np.fft.fftshift(
            np.fft.fftfreq(ny, d=self.dy/(2*np.pi)))
        mask_k = np.abs(py_full) <= p_max
        return W_fft[mask_k], py_full[mask_k]

    def momentum_density(self):
        # Critical fix: Check for state consistency between thread updates
        if self.psi_full is None or self.mask is None:
            return None, None, None
            
        # If dimensions don't match (due to resolution change in solve thread), abort
        if self.psi_full.shape != self.mask.shape:
            return None, None, None

        psi_pad = self.psi_full.copy()
        # Safe to index now
        psi_pad[~self.mask] = 0.0
        
        fft2 = np.fft.fftshift(np.fft.fft2(psi_pad))
        kdens = np.abs(fft2)**2
        nx, ny = psi_pad.shape[1], psi_pad.shape[0]
        dkx = 2*np.pi / (nx * self.dx)
        dky = 2*np.pi / (ny * self.dy)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=self.dx/(2*np.pi)))
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=self.dy/(2*np.pi)))
        return kdens, kx, ky

def draw_text(screen, text_str, font, color, x, y, align="left"):
    surf = font.render(tex(text_str), True, color)
    if align == "right":
        x -= surf.get_width()
    elif align == "center":
        x -= surf.get_width()//2
    screen.blit(surf, (x, y))
    return surf.get_height()

def draw_panel(screen, rect, color=PANEL, border=BORDER, radius=6):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)

def mini_plot(screen, rect, series, color, max_val=None, label=""):
    draw_panel(screen, rect, PANEL2)
    if len(series) < 2:
        return
    data = series[-rect.width:]
    mv = max(data) if max_val is None else max_val
    if mv < 1e-14:
        return
    pts = []
    for i, v in enumerate(data):
        px = rect.x + int(i/len(data)*rect.width)
        py = rect.bottom - int((v/mv)*(rect.height-4)) - 2
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(screen, color, False, pts, 1)
    if label:
        draw_text(screen, label, FONT_SM, TEXT_DIM, rect.x+4, rect.y+3)

def colorbar(screen, cmap, rect, vmin=0, vmax=1, label=""):
    for i in range(rect.height):
        t = 1.0 - i/rect.height
        rgb = tuple(int(c*255) for c in cmap(t)[:3])
        pygame.draw.line(screen, rgb, (rect.x, rect.y+i), (rect.right, rect.y+i))
    pygame.draw.rect(screen, BORDER, rect, 1)
    draw_text(screen, f"{vmax:.0f}", FONT_SM, TEXT, rect.right+4, rect.y)
    draw_text(screen, f"{vmin:.0f}", FONT_SM, TEXT, rect.right+4, rect.bottom-12)
    if label:
        draw_text(screen, label, FONT_SM, TEXT_DIM, rect.right+4, rect.centery-6)

class QuantumLab:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Quantum Scar Laboratory v2")
        self.clock = pygame.time.Clock()
        self.engine = Engine()
        self.engine.start()

        self.playing = False
        self.tab = 0
        self.cmap_idx = 0
        self.cmap = matplotlib.colormaps[CMAPS[self.cmap_idx]]
        self.view_mode = 0
        self.ps_mode = 0
        self.selected_eig = -1
        self.static_dens = None
        self.husimi_cache = None
        self.wigner_cache = None
        self.ps_dirty = True

        sx = SIDEBAR_W + 20 if False else 16
        self.sliders = [
            Slider(16, 110, SIDEBAR_W-32, 0.0, 1.5, self.engine.a,
                   "Geometry a", unit=" [B]"),
            Slider(16, 175, SIDEBAR_W-32, 40, 120, self.engine.N,
                   "Resolution N", fmt=".0f"),
            Slider(16, 240, SIDEBAR_W-32, 50, 500, self.engine.sigma_target,
                   r"Energy \sigma", unit=r" [\hbar^2/2m]"),
            Slider(16, 305, SIDEBAR_W-32, 0.002, 0.05, self.engine.dt,
                   "Time step", unit=r" [\hbar/E]"),
        ]

        self._tabs = ["CONTROLS","STATES","ANALYSIS"]
        self._cmapbtn = pygame.Rect(16, 365, SIDEBAR_W-32, 28)
        self._solvebtn = pygame.Rect(16, 405, SIDEBAR_W-32, 28)
        self._playbtn = pygame.Rect(16, 445, SIDEBAR_W-32, 28)
        self._viewbtn = pygame.Rect(16, 490, SIDEBAR_W-32, 28)
        self._psbtn = pygame.Rect(16, 530, SIDEBAR_W-32, 28)

        self._drag = False
        self._drag_start = None

        self._ps_rect = pygame.Rect(SIDEBAR_W, SIM_H + BOTTOM_H//2,
                                    SIM_W//2, BOTTOM_H//2 - 4)

    def _next_cmap(self):
        self.cmap_idx = (self.cmap_idx + 1) % len(CMAPS)
        self.cmap = matplotlib.colormaps[CMAPS[self.cmap_idx]]

    def _draw_tabs(self):
        tw = SIDEBAR_W // 3
        for i, name in enumerate(self._tabs):
            r = pygame.Rect(i*tw, 0, tw, 38)
            col = ACCENT if i == self.tab else PANEL2
            pygame.draw.rect(self.screen, col, r)
            pygame.draw.rect(self.screen, BORDER, r, 1)
            draw_text(self.screen, name, FONT_SM,
                      BG if i==self.tab else TEXT_DIM,
                      r.centerx, r.centery-6, align="center")

    def _handle_tab_click(self, pos):
        tw = SIDEBAR_W // 3
        if pos[1] < 38 and pos[0] < SIDEBAR_W:
            self.tab = pos[0] // tw

    def _draw_controls(self):
        draw_text(self.screen, "PARAMETERS", FONT_LG, TEXT, 16, 50)
        for s in self.sliders:
            s.draw(self.screen)

        def btn(rect, label, active=False, col=None):
            c = col or (ACCENT if active else PANEL2)
            draw_panel(self.screen, rect, c, BORDER, radius=4)
            draw_text(self.screen, label, FONT_SM,
                      BG if active else TEXT,
                      rect.centerx, rect.centery-7, align="center")

        btn(self._cmapbtn, f"CMAP: {CMAPS[self.cmap_idx].upper()}")
        btn(self._solvebtn, "RE-SOLVE [R]",
            active=self.engine.is_solving, col=ACCENT4 if self.engine.is_solving else None)
        btn(self._playbtn, "PAUSE [SPACE]" if self.playing else "PLAY [SPACE]",
            active=self.playing)

        draw_text(self.screen, "VISUALISATION", FONT_LG, TEXT, 16, 478)
        view_labels = ["DENSITY","PHASE"]
        btn(self._viewbtn, f"VIEW: {view_labels[self.view_mode]}")
        ps_labels = ["PS: OFF","PS: HUSIMI","PS: WIGNER"]
        btn(self._psbtn, ps_labels[self.ps_mode])

        if self.engine.is_solving:
            draw_text(self.screen, r"\cycle solving...", FONT_SM, ACCENT3, 16, 572)
        elif self.engine.vals is not None:
            n = len(self.engine.vals)
            e0, e1 = self.engine.vals[0], self.engine.vals[-1]
            draw_text(self.screen, fr"{n} states E \in [{e0:.1f},{e1:.1f}]",
                      FONT_SM, ACCENT2, 16, 572)
            draw_text(self.screen, r"drag on sim \to launch packet",
                      FONT_SM, TEXT_DIM, 16, 590)
            draw_text(self.screen, "SPACE=play R=re-solve",
                      FONT_SM, TEXT_DIM, 16, 606)

    def _draw_states(self):
        if self.engine.vals is None:
            draw_text(self.screen, "no data — press R", FONT_MD, TEXT_DIM, 16, 60)
            return

        vals = self.engine.vals
        eig_ipr = self.engine.eig_ipr
        coeffs = self.engine.coeffs
        n = len(vals)

        panel_r = pygame.Rect(4, 44, SIDEBAR_W-8, HEIGHT-48)
        draw_panel(self.screen, panel_r, PANEL2)

        e_min, e_max = vals[0], vals[-1]
        e_range = max(e_max - e_min, 1e-6)

        ipr_mean = np.mean(eig_ipr)
        ipr_std = np.std(eig_ipr)
        scar_thresh = ipr_mean + 2*ipr_std

        if coeffs is not None:
            c2 = np.abs(coeffs)**2
            c2_max = max(c2.max(), 1e-14)
        else:
            c2 = None
            c2_max = 1.0

        ax_x = 50
        ax_top = panel_r.y + 10
        ax_bot = panel_r.bottom - 10
        ax_h = ax_bot - ax_top

        pygame.draw.line(self.screen, BORDER,
                         (ax_x, ax_top), (ax_x, ax_bot), 1)
        for tick in np.linspace(e_min, e_max, 5):
            ty = ax_bot - int((tick - e_min)/e_range * ax_h)
            pygame.draw.line(self.screen, BORDER, (ax_x-3, ty), (ax_x+3, ty), 1)
            draw_text(self.screen, f"{tick:.0f}", FONT_SM, TEXT_DIM,
                      ax_x-4, ty-5, align="right")

        draw_text(self.screen, r"E [\hbar^2/2m]", FONT_SM, TEXT_DIM,
                  ax_x-2, ax_top-14, align="right")

        stick_x0 = ax_x + 6
        stick_maxw = SIDEBAR_W - ax_x - 24
        mouse_pos = pygame.mouse.get_pos()

        for i in range(n):
            sy = ax_bot - int((vals[i] - e_min)/e_range * ax_h)
            is_scar = eig_ipr[i] > scar_thresh
            is_selected = (i == self.selected_eig)
            col = SCAR_COL if is_scar else ACCENT
            if is_selected:
                col = WHITE

            pygame.draw.line(self.screen, col,
                             (stick_x0, sy), (stick_x0+6, sy), 2)

            if c2 is not None:
                bw = int((c2[i]/c2_max) * stick_maxw)
                if bw > 0:
                    br = pygame.Rect(stick_x0+8, sy-2, bw, 4)
                    pygame.draw.rect(self.screen, (*col, 160), br,
                                     border_radius=1)

            hr = pygame.Rect(stick_x0, sy-5, stick_maxw, 10)
            if hr.collidepoint(mouse_pos):
                tip = (f"E={vals[i]:.2f} "
                       f"IPR={eig_ipr[i]:.4f}"
                       + (" SCAR" if is_scar else ""))
                draw_text(self.screen, tip, FONT_SM, WHITE,
                          4, HEIGHT-22)

        leg_y = panel_r.bottom - 42
        pygame.draw.line(self.screen, ACCENT, (6, leg_y), (18, leg_y), 2)
        draw_text(self.screen, "normal", FONT_SM, TEXT_DIM, 22, leg_y-6)
        pygame.draw.line(self.screen, SCAR_COL, (6, leg_y+14),(18, leg_y+14), 2)
        draw_text(self.screen, r"scar (IPR > \mu + 2\sigma)", FONT_SM, TEXT_DIM, 22, leg_y+8)
        if coeffs is not None:
            draw_text(self.screen, r"bar width = |c_n|^2", FONT_SM, TEXT_DIM, 6, leg_y+22)

        draw_text(self.screen, r"click stick \to view \psi_n",
                  FONT_SM, TEXT_DIM, 6, panel_r.bottom-10)

    def _states_click(self, pos):
        if self.engine.vals is None:
            return
        vals = self.engine.vals
        n = len(vals)
        e_min, e_max = vals[0], vals[-1]
        e_range = max(e_max - e_min, 1e-6)
        ax_x = 54
        ax_top = 54
        ax_bot = HEIGHT - 10
        ax_h = ax_bot - ax_top
        stick_x0 = ax_x + 6
        stick_maxw = SIDEBAR_W - ax_x - 24

        for i in range(n):
            sy = ax_bot - int((vals[i] - e_min)/e_range * ax_h)
            hr = pygame.Rect(stick_x0, sy-6, stick_maxw, 12)
            if hr.collidepoint(pos):
                if self.selected_eig == i:
                    self.selected_eig = -1
                    self.static_dens = None
                else:
                    self.selected_eig = i
                    mask = self.engine.mask
                    vecs = self.engine.vecs
                    psi_n = np.zeros(mask.shape)
                    psi_n[mask] = vecs[:, i]
                    self.static_dens = psi_n**2
                break

    def _draw_analysis(self):
        draw_text(self.screen, "ANALYSIS", FONT_LG, TEXT, 16, 50)
        if self.engine.vals is None:
            draw_text(self.screen, "solve first", FONT_SM, TEXT_DIM, 16, 80)
            return

        rt = self.engine.revival_times
        y = 72
        draw_text(self.screen, r"REVIVAL TIMESCALES [\hbar/E]",
                  FONT_SM, ACCENT, 16, y); y += 16
        for key, val in rt.items():
            draw_text(self.screen, f" {key:8s} = {val:.3f}",
                      FONT_SM, TEXT, 16, y); y += 14

        if self.engine.coeffs is not None:
            y += 8
            draw_text(self.screen, "EXPANSION NORM",
                      FONT_SM, ACCENT, 16, y); y += 16
            norm_check = float(np.sum(np.abs(self.engine.coeffs)**2))
            draw_text(self.screen, fr" \sum|c_n|^2 = {norm_check:.4f} (should be 1)",
                      FONT_SM, TEXT, 16, y); y += 14
            draw_text(self.screen, " [D03 Eq.8a]", FONT_SM, TEXT_DIM, 16, y)

    def _get_display_data(self, dens):
        if self.view_mode == 1 and self.engine.psi_full is not None:
            return dens, np.angle(self.engine.psi_full)
        return dens, None

    def _draw_sim(self, dens):
        if dens is None:
            pygame.draw.rect(self.screen, PANEL,
                             pygame.Rect(SIDEBAR_W, 0, SIM_W, SIM_H))
            draw_text(self.screen, "solving..." if self.engine.is_solving
                      else "drag to launch wavepacket",
                      FONT_MD, TEXT_DIM,
                      SIDEBAR_W + SIM_W//2, SIM_H//2 - 8, align="center")
            return

        dens_arr, phase = self._get_display_data(dens)

        if phase is not None:
            vmax = np.percentile(dens_arr[self.engine.mask], 98) or 1.0
            V = np.clip(dens_arr / vmax, 0, 1)
            H = (phase + np.pi) / (2*np.pi)
            Hi = (H * 6).astype(int) % 6
            f = H * 6 - np.floor(H * 6)
            q_ch = V * (1 - f)
            t_ch = V * f
            z = np.zeros_like(V)
            R = np.select([Hi==0,Hi==1,Hi==2,Hi==3,Hi==4,Hi==5],[V,q_ch,z,z,t_ch,V])
            G = np.select([Hi==0,Hi==1,Hi==2,Hi==3,Hi==4,Hi==5],[t_ch,V,V,q_ch,z,z])
            B = np.select([Hi==0,Hi==1,Hi==2,Hi==3,Hi==4,Hi==5],[z,z,t_ch,V,V,q_ch])
            rgb = (np.stack([R, G, B], axis=-1) * 255).astype(np.uint8)
            rgb[~self.engine.mask] = (0, 0, 0)
        else:
            vmax = np.percentile(dens_arr[self.engine.mask], 98) or 1.0
            dens_n = np.clip(dens_arr/vmax, 0, 1)
            rgb = (self.cmap(dens_n)[:,:,:3] * 255).astype(np.uint8)
            rgb[~self.engine.mask] = (0, 0, 0)

        surf = pygame.surfarray.make_surface(np.flipud(rgb))
        surf = pygame.transform.scale(surf, (SIM_W, SIM_H))
        self.screen.blit(surf, (SIDEBAR_W, 0))

        cb_rect = pygame.Rect(SIDEBAR_W + SIM_W + 4, 10, 10, SIM_H-20)
        if phase is None:
            colorbar(self.screen, self.cmap, cb_rect, 0, round(vmax,1), r"|\psi|^2")

        self._draw_domain_outline()

        if self._drag and self._drag_start:
            mp = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, ACCENT, self._drag_start, mp, 2)
            pygame.draw.circle(self.screen, ACCENT, self._drag_start, 4)

        mode_lbl = "PHASE" if self.view_mode == 1 else "DENSITY"
        draw_text(self.screen, mode_lbl, FONT_SM, TEXT_DIM,
                  SIDEBAR_W + 6, 4)
        draw_text(self.screen, fr"t = {self.engine.time:.2f} \hbar/E",
                  FONT_SM, TEXT_DIM, SIDEBAR_W + SIM_W - 6, 4, align="right")

    def _draw_domain_outline(self):
        if self.engine.mask is None:
            return
        mask = self.engine.mask
        B = 1.0; a = self.engine.a

        def w2s(wx, wy):
            sx_scale = SIM_W / (2*(a+B))
            sy_scale = SIM_H / (2*B)
            px = SIDEBAR_W + int((wx + a + B) * sx_scale)
            py = SIM_H - int((wy + B) * sy_scale)
            return (px, py)

        pts = []
        n_pts = 120
        for i in range(n_pts+1):
            theta = 2*np.pi * i / n_pts
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            if -np.pi/2 <= theta <= np.pi/2:
                wx = a + B*cos_t
                wy = B*sin_t
            elif np.pi/2 < theta <= np.pi:
                wx = -(a + B*abs(cos_t))
                wy = B*sin_t
            else:
                wx = a*np.sign(cos_t) + B*cos_t if abs(theta) > np.pi/2 else a + B*cos_t
                wy = B*sin_t
            pts.append(w2s(wx, wy))

        pts_outline = []
        for t in np.linspace(-np.pi/2, np.pi/2, 40):
            pts_outline.append(w2s(a + B*np.cos(t), B*np.sin(t)))
        pts_outline.append(w2s(a, B))
        pts_outline.append(w2s(-a, B))
        for t in np.linspace(np.pi/2, 3*np.pi/2, 40):
            pts_outline.append(w2s(-a + B*np.cos(t), B*np.sin(t)))
        pts_outline.append(w2s(-a, -B))
        pts_outline.append(w2s(a, -B))
        pts_outline.append(pts_outline[0])

        if len(pts_outline) > 1:
            pygame.draw.lines(self.screen, (0, 200, 220), False, pts_outline, 1)

    def _draw_bottom(self):
        bottom_rect = pygame.Rect(SIDEBAR_W, SIM_H, SIM_W, BOTTOM_H)
        draw_panel(self.screen, bottom_rect, BG, BORDER, radius=0)

        half_w = SIM_W // 2

        auto_rect = pygame.Rect(SIDEBAR_W+4, SIM_H+4, half_w-8, BOTTOM_H-8)
        draw_panel(self.screen, auto_rect, PANEL2)
        draw_text(self.screen, r"|<\psi_0|\psi(t)>|^2 AUTOCORRELATION",
                  FONT_SM, ACCENT, auto_rect.x+6, auto_rect.y+4)

        if len(self.engine.auto_series) >= 2:
            auto = self.engine.auto_series
            mv = max(auto) or 1.0
            pts = []
            n_show = auto_rect.width - 12
            seg = auto[-n_show:]
            for i, v in enumerate(seg):
                px = auto_rect.x + 6 + int(i/(max(len(seg)-1,1))*(n_show-1))
                py = auto_rect.bottom - 6 - int((v/mv)*(auto_rect.height-24))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, ACCENT, False, pts, 1)

            rt = self.engine.revival_times
            dt = self.engine.dt
            t_total = len(auto) * dt
            cols_rt = {
                'T_cl': ACCENT2,
                'T_H': ACCENT4,
                'T_rev': WHITE,
                'T_1/2': (*ACCENT3[:3],),
                'T_1/3': (*ACCENT3[:3],),
                'T_1/4': (*ACCENT3[:3],),
            }
            for key, tval in rt.items():
                if tval <= 0 or tval > t_total * 1.2:
                    continue
                frac = tval / t_total
                lx = auto_rect.x + 6 + int(frac * (n_show-1))
                if auto_rect.x+6 <= lx <= auto_rect.right-6:
                    col = cols_rt.get(key, TEXT_DIM)
                    pygame.draw.line(self.screen, col,
                                     (lx, auto_rect.y+18),
                                     (lx, auto_rect.bottom-6), 1)
                    draw_text(self.screen, key, FONT_SM, col,
                              lx+2, auto_rect.y+18)

            draw_text(self.screen, r"t [\hbar/E]", FONT_SM, TEXT_DIM,
                      auto_rect.right-50, auto_rect.bottom-14)
            draw_text(self.screen, "1.0", FONT_SM, TEXT_DIM,
                      auto_rect.x+6, auto_rect.y+18)

        ipr_rect = pygame.Rect(SIDEBAR_W + half_w + 4, SIM_H+4,
                               half_w-8, (BOTTOM_H)//2 - 6)
        mini_plot(self.screen, ipr_rect, self.engine.ipr_series,
                  ACCENT2, label=r"IPR(t) = \int|\psi|^4 dA")

        kdens, kx, ky = self.engine.momentum_density()
        kdens_rect = pygame.Rect(SIDEBAR_W + half_w + 4,
                                 SIM_H + BOTTOM_H//2 + 2,
                                 half_w-8, BOTTOM_H//2 - 6)
        draw_panel(self.screen, kdens_rect, PANEL2)
        draw_text(self.screen, r"|\psi~(k)|^2 MOMENTUM SPACE",
                  FONT_SM, ACCENT, kdens_rect.x+4, kdens_rect.y+4)

        if kdens is not None:
            vmax_k = np.percentile(kdens, 99.5) or 1.0
            kn = np.clip(kdens/vmax_k, 0, 1)
            rgb_k = (self.cmap(kn)[:,:,:3]*255).astype(np.uint8)
            ks = pygame.surfarray.make_surface(np.flipud(rgb_k))
            ks = pygame.transform.scale(ks, (kdens_rect.width-4,
                                             kdens_rect.height-20))
            self.screen.blit(ks, (kdens_rect.x+2, kdens_rect.y+18))

            if self.engine.vals is not None:
                E_mean = float(np.mean(self.engine.vals))
                k_ring = np.sqrt(E_mean)
                kx_range = kx[-1] - kx[0]
                ky_range = ky[-1] - ky[0]
                rw = kdens_rect.width - 4
                rh = kdens_rect.height - 20
                cx = kdens_rect.x + 2 + rw//2
                cy = kdens_rect.y + 18 + rh//2
                rx = int(k_ring/kx_range * rw) if kx_range > 0 else 0
                ry = int(k_ring/ky_range * rh) if ky_range > 0 else 0
                r_avg = (rx+ry)//2
                if 2 < r_avg < min(rw,rh)//2:
                    pygame.draw.circle(self.screen, (220,220,80),
                                       (cx, cy), r_avg, 1)
                    draw_text(self.screen, fr"|k|=\sqrtE={k_ring:.1f}",
                              FONT_SM, (220,220,80),
                              cx + r_avg + 2, cy - 6)

    def _draw_phase_space(self):
        if self.ps_mode == 0:
            return
        if self.engine.psi_full is None:
            return

        psi_flat = self.engine.psi_full[self.engine.mask]
        ps_rect = pygame.Rect(SIDEBAR_W, SIM_H,
                              SIM_W//3, BOTTOM_H-4)
        draw_panel(self.screen, ps_rect, PANEL2)

        if self.ps_mode == 1 and self.ps_dirty:
            result = self.engine.husimi(psi_flat, grid_n=32)
            if result:
                self.husimi_cache = result
            self.ps_dirty = False

        elif self.ps_mode == 2 and self.ps_dirty:
            result = self.engine.wigner_marginal(psi_flat)
            if result:
                self.wigner_cache = result
            self.ps_dirty = False

        if self.ps_mode == 1 and self.husimi_cache:
            Q, x0s, p0s = self.husimi_cache
            vmax_q = Q.max() or 1.0
            Qn = np.clip(Q/vmax_q, 0, 1)
            rgb_q = (self.cmap(Qn)[:,:,:3]*255).astype(np.uint8)
            qs = pygame.surfarray.make_surface(rgb_q)
            qs = pygame.transform.scale(qs, (ps_rect.width-4, ps_rect.height-20))
            self.screen.blit(qs, (ps_rect.x+2, ps_rect.y+20))
            draw_text(self.screen, "HUSIMI Q(x,p)",
                      FONT_SM, ACCENT, ps_rect.x+4, ps_rect.y+4)
            draw_text(self.screen, r"x \to", FONT_SM, TEXT_DIM,
                      ps_rect.right-30, ps_rect.bottom-14)
            draw_text(self.screen, "p", FONT_SM, TEXT_DIM,
                      ps_rect.x+4, ps_rect.y+22)

        elif self.ps_mode == 2 and self.wigner_cache:
            W, py_vals = self.wigner_cache
            draw_text(self.screen, r"WIGNER W(y=0, p_y) [marginal]",
                      FONT_SM, ACCENT, ps_rect.x+4, ps_rect.y+4)
            W_max = max(abs(W.max()), abs(W.min()), 1e-14)
            pts = []
            for i, wv in enumerate(W):
                px = ps_rect.x + 6 + int(i/len(W)*(ps_rect.width-12))
                py = ps_rect.centery - int((wv/W_max)*(ps_rect.height//2-10))
                pts.append((px, py))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, ACCENT, False, pts, 1)
            pygame.draw.line(self.screen, BORDER,
                             (ps_rect.x+6, ps_rect.centery),
                             (ps_rect.right-6, ps_rect.centery), 1)
            draw_text(self.screen, r"p_y \to", FONT_SM, TEXT_DIM,
                      ps_rect.right-36, ps_rect.bottom-14)
            draw_text(self.screen, "(neg values = quantum interference)",
                      FONT_SM, TEXT_DIM, ps_rect.x+4, ps_rect.bottom-14)

    def _draw_sidebar(self):
        pygame.draw.rect(self.screen, PANEL,
                         pygame.Rect(0, 0, SIDEBAR_W, HEIGHT))
        pygame.draw.line(self.screen, BORDER,
                         (SIDEBAR_W-1, 0), (SIDEBAR_W-1, HEIGHT), 1)
        self._draw_tabs()
        if self.tab == 0:
            self._draw_controls()
        elif self.tab == 1:
            self._draw_states()
        elif self.tab == 2:
            self._draw_analysis()

    def screen_to_world(self, pos):
        xpix, ypix = pos
        B = 1.0; a = self.engine.a
        sx = SIM_W / (2*(a+B))
        sy = SIM_H / (2*B)
        wx = (xpix - SIDEBAR_W) / sx - (a+B)
        wy = (SIM_H - ypix) / sy - B
        return wx, wy

    def run(self):
        ps_update_counter = 0

        while True:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_tab_click(event.pos)

                if self.tab == 0:
                    for s in self.sliders:
                        if s.handle(event):
                            self.engine.a = self.sliders[0].value
                            self.engine.N = int(self.sliders[1].value)
                            self.engine.sigma_target = self.sliders[2].value
                            self.engine.dt = self.sliders[3].value

                if event.type == pygame.MOUSEBUTTONDOWN and self.tab == 0:
                    if self._cmapbtn.collidepoint(event.pos):
                        self._next_cmap()
                    if self._solvebtn.collidepoint(event.pos):
                        self.engine.start()
                    if self._playbtn.collidepoint(event.pos):
                        self.playing = not self.playing
                    if self._viewbtn.collidepoint(event.pos):
                        self.view_mode = (self.view_mode + 1) % 2
                    if self._psbtn.collidepoint(event.pos):
                        self.ps_mode = (self.ps_mode + 1) % 3
                        self.ps_dirty = True

                if (event.type == pygame.MOUSEBUTTONDOWN
                        and self.tab == 1
                        and event.pos[0] < SIDEBAR_W):
                    self._states_click(event.pos)

                if (event.type == pygame.MOUSEBUTTONDOWN
                        and event.pos[0] > SIDEBAR_W
                        and event.pos[1] < SIM_H):
                    self._drag = True
                    self._drag_start = event.pos
                    self.selected_eig = -1
                    self.static_dens = None

                if event.type == pygame.MOUSEBUTTONUP and self._drag:
                    self._drag = False
                    if self.engine.vecs is not None:
                        end = event.pos
                        x0, y0 = self.screen_to_world(self._drag_start)
                        px = (end[0] - self._drag_start[0]) * 0.02
                        py = (end[1] - self._drag_start[1]) * 0.02
                        self.engine.initialize_packet(x0, y0, px, -py)
                        self.playing = True
                        self.ps_dirty = True

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    if event.key == pygame.K_r:
                        self.engine.start()
                    if event.key == pygame.K_v:
                        self.view_mode = (self.view_mode + 1) % 2
                    if event.key == pygame.K_p:
                        self.ps_mode = (self.ps_mode + 1) % 3
                        self.ps_dirty = True
                    if event.key == pygame.K_c:
                        self._next_cmap()

            self.screen.fill(BG)

            dens = None
            if self.selected_eig >= 0 and self.static_dens is not None:
                dens = self.static_dens
            elif not self.engine.is_solving and self.engine.vecs is not None:
                if self.playing:
                    dens = self.engine.evolve()
                    ps_update_counter += 1
                    if ps_update_counter % 8 == 0:
                        self.ps_dirty = True
                else:
                    dens = np.zeros(self.engine.mask.shape)

            self._draw_sim(dens)
            self._draw_bottom()
            if self.ps_mode > 0:
                self._draw_phase_space()
            self._draw_sidebar()

            pygame.display.flip()

if __name__ == "__main__":
    QuantumLab().run()