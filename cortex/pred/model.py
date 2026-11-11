"""Brains that predict their next senses. All share one encoder and one decoder, so what
differs between arms is only the core: how state is kept and how it moves.

A SHEET is a lattice of cells with C channels. Every cell is a small perceptron reading
its 3x3 neighbourhood, a compressed code of the senses, its own position and, with
`long` > 0, the channels of `long` far-away cells (fixed random wiring: outputs of one
cell become inputs of another, not only next door). With `types` > 1 cells come in kinds,
each kind with its own perceptron. Weights do not depend on the number of cells, so size
is free for the search: that is the whole bet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SENS, N_ACT, D_CODE, D_OUT, D_PROJ = 220, 3, 96, 96, 24


class Sheet(nn.Module):
    def __init__(self, side=16, ch=8, hidden=24, types=1, long=0, passes=2, seed=0):
        super().__init__()
        self.side, self.ch, self.passes = side, ch, passes
        g = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(D_CODE, D_PROJ)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side),
                                indexing="ij")
        self.register_buffer("pos", torch.stack([ys, xs, torch.sin(3.14159 * ys),
                                                 torch.sin(3.14159 * xs)])[None])
        kernel = torch.eye(9).reshape(9, 1, 3, 3).repeat(ch, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        n_in = 9 * ch + D_PROJ + 4 + long * ch
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Conv2d(n_in, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, ch, 1))
            for _ in range(types))
        for m in self.mlps:                       # a cell starts out doing nothing
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        kinds = torch.randint(types, (side, side), generator=g)
        self.register_buffer("masks", torch.stack([(kinds == k).float()
                                                   for k in range(types)])[:, None, None])
        self.long = long
        if long:
            self.register_buffer("src", torch.randint(side * side, (long, side * side),
                                                      generator=g))
        self.gate = nn.Conv2d(ch, 1, 1)
        self.value = nn.Conv2d(ch, D_OUT, 1)

    def init_state(self, batch, device=None):
        return torch.zeros(batch, self.ch, self.side, self.side, device=device)

    def forward(self, code, s):
        b = code.shape[0]
        c = self.proj(code)[:, :, None, None].expand(-1, -1, self.side, self.side)
        pos = self.pos.expand(b, -1, -1, -1)
        for _ in range(self.passes):
            nb = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="circular"), self.kernel, groups=self.ch)
            parts = [nb, c, pos]
            if self.long:
                far = s.flatten(2)[:, :, self.src]        # (b, ch, long, cells)
                parts.append(far.flatten(1, 2).reshape(b, self.ch * self.long,
                                                       self.side, self.side))
            x = torch.cat(parts, 1)
            d = sum(mask * m(x) for mask, m in zip(self.masks, self.mlps))
            s = (s + 0.3 * torch.tanh(d)).clamp(-4, 4)
        w = torch.softmax(self.gate(s).flatten(1), 1)
        out = (self.value(s).flatten(2) * w[:, None]).sum(2)
        return out, s


class Rec(nn.Module):
    """The control: a plain GRU, memory as one vector, wired all-to-all."""

    def __init__(self, hidden=16):
        super().__init__()
        self.proj = nn.Linear(D_CODE, D_PROJ)
        self.gru = nn.GRUCell(D_PROJ, hidden)
        self.value = nn.Linear(hidden, D_OUT)
        self.hidden = hidden

    def init_state(self, batch, device=None):
        return torch.zeros(batch, self.hidden, device=device)

    def forward(self, code, s):
        s = self.gru(self.proj(code), s)
        return self.value(s), s


class Now(nn.Module):
    """No memory at all: a perceptron over this moment's senses."""

    def __init__(self, hidden=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_CODE, hidden), nn.ReLU(), nn.Linear(hidden, D_OUT))

    def init_state(self, batch, device=None):
        return torch.zeros(batch, 1, device=device)

    def forward(self, code, s):
        return self.net(code), s


class Brain(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_SENS + N_ACT + 1, D_CODE), nn.Tanh())
        self.core = core
        self.dec = nn.Linear(D_OUT, N_SENS)

    def core_weights(self) -> int:
        return sum(p.numel() for p in self.core.parameters())

    def run(self, sens, act, eyes):
        """sens (b, t, 220) normalised, act (b, t, 3), eyes (b, t) 1 = open. Returns the
        prediction of frame t+1 at every t; with the eyes closed only the body is felt."""
        b, t = sens.shape[:2]
        s = self.core.init_state(b, sens.device)
        preds, reps = [], []
        for i in range(t):
            e = eyes[:, i, None]
            seen = torch.cat([sens[:, i, :212] * e, sens[:, i, 212:]], 1)
            out, s = self.core(self.enc(torch.cat([seen, act[:, i], e], 1)), s)
            preds.append(self.dec(out))
            reps.append(out)
        return torch.stack(preds, 1), torch.stack(reps, 1)


class WideRec(nn.Module):
    """A GRU that hears the whole code: the 24-number squeeze removed."""

    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRUCell(D_CODE, hidden)
        self.value = nn.Linear(hidden, D_OUT)
        self.hidden = hidden

    def init_state(self, batch, device=None):
        return torch.zeros(batch, self.hidden, device=device)

    def forward(self, code, s):
        s = self.gru(code, s)
        return self.value(s), s


class RetinaBrain(nn.Module):
    """A sheet wired to the senses cell by cell. A cell READS the rays through a learned
    address: key(where the cell is) against key(which way the ray looks), and each ray is
    predicted by READING the cells the same way. So input and output are connections
    between cells and rays, not one summary every cell hears alike, and the weights still
    do not depend on the number of cells or rays. Everything else stays: the 3x3, the
    position, the types, the far wiring; smell, touch, the body and the action are a code
    all cells hear."""

    N_RAYS, RAY_CH, D_RAY, D_KEY, D_RAY_OUT = 32, 6, 8, 8, 16

    def __init__(self, side=16, ch=8, hidden=24, types=1, long=0, passes=2, seed=0):
        super().__init__()
        self.side, self.ch, self.passes, self.long = side, ch, passes, long
        g = torch.Generator().manual_seed(seed)
        ang = torch.linspace(-1, 1, self.N_RAYS)
        self.register_buffer("ang", torch.stack([ang, torch.sin(1.57 * ang),
                                                 torch.cos(1.57 * ang)], 1))
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side),
                                indexing="ij")
        pos = torch.stack([ys, xs, torch.sin(3.14159 * ys), torch.sin(3.14159 * xs)])
        self.register_buffer("pos", pos[None])
        self.register_buffer("pos_flat", pos.flatten(1).T)
        self.register_buffer("kernel", torch.eye(9).reshape(9, 1, 3, 3).repeat(ch, 1, 1, 1))
        n_static = N_SENS - self.N_RAYS * self.RAY_CH
        self.static_in = nn.Sequential(nn.Linear(n_static + N_ACT + 1, D_PROJ), nn.Tanh())
        self.ray_in = nn.Linear(self.RAY_CH + 3, self.D_RAY)
        self.read_cell = nn.Linear(4, self.D_KEY)
        self.read_ray = nn.Linear(3, self.D_KEY)
        self.out_cell = nn.Linear(4, self.D_KEY)
        self.out_ray = nn.Linear(3, self.D_KEY)
        n_in = 9 * ch + D_PROJ + 4 + self.D_RAY + long * ch
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Conv2d(n_in, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, ch, 1))
            for _ in range(types))
        for m in self.mlps:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
        kinds = torch.randint(types, (side, side), generator=g)
        self.register_buffer("masks", torch.stack([(kinds == k).float()
                                                   for k in range(types)])[:, None, None])
        if long:
            self.register_buffer("src", torch.randint(side * side, (long, side * side),
                                                      generator=g))
        self.value = nn.Conv2d(ch, self.D_RAY_OUT, 1)
        self.ray_dec = nn.Linear(self.D_RAY_OUT, self.RAY_CH)
        self.gate = nn.Conv2d(ch, 1, 1)
        self.static_val = nn.Conv2d(ch, D_PROJ, 1)
        self.static_dec = nn.Linear(D_PROJ, n_static)

    def core_weights(self) -> int:
        return sum(p.numel() for p in self.mlps.parameters())

    def run(self, sens, act, eyes):
        b, t = sens.shape[:2]
        n, c_ray = self.N_RAYS, self.RAY_CH
        s = torch.zeros(b, self.ch, self.side, self.side)
        pos = self.pos.expand(b, -1, -1, -1)
        # who reads what: (cells, rays) and (rays, cells); addresses do not change in time
        read = torch.softmax(self.read_cell(self.pos_flat) @ self.read_ray(self.ang).T, 1)
        write = torch.softmax(self.out_ray(self.ang) @ self.out_cell(self.pos_flat).T, 1)
        preds = []
        ang = self.ang.expand(b, -1, -1)
        for i in range(t):
            e = eyes[:, i, None]
            rays = sens[:, i, :n * c_ray].reshape(b, n, c_ray) * e[:, None]
            r = self.ray_in(torch.cat([rays, ang], 2))                 # (b, rays, D_RAY)
            heard = (read @ r).transpose(1, 2).reshape(b, self.D_RAY, self.side, self.side)
            static = torch.cat([sens[:, i, n * c_ray:212] * e, sens[:, i, 212:], act[:, i], e], 1)
            code = self.static_in(static)[:, :, None, None].expand(-1, -1, self.side, self.side)
            for _ in range(self.passes):
                nb = F.conv2d(F.pad(s, (1, 1, 1, 1), mode="circular"), self.kernel, groups=self.ch)
                parts = [nb, code, pos, heard]
                if self.long:
                    far = s.flatten(2)[:, :, self.src]
                    parts.append(far.flatten(1, 2).reshape(b, self.ch * self.long,
                                                           self.side, self.side))
                x = torch.cat(parts, 1)
                d = sum(mask * m(x) for mask, m in zip(self.masks, self.mlps))
                s = (s + 0.3 * torch.tanh(d)).clamp(-4, 4)
            v = self.value(s).flatten(2).transpose(1, 2)               # (b, cells, D_RAY_OUT)
            ray_pred = self.ray_dec(write @ v).reshape(b, n * c_ray)
            w = torch.softmax(self.gate(s).flatten(1), 1)
            st = (self.static_val(s).flatten(2) * w[:, None]).sum(2)
            preds.append(torch.cat([ray_pred, self.static_dec(st)], 1))
        return torch.stack(preds, 1), None


class FlowBrain(nn.Module):
    """A ring of cells around the body whose MEMORY IS CARRIED BY THE BODY'S MOTION.

    Columns are directions relative to the body. A ray always lands in, and is predicted
    from, the columns it looks along: fixed wiring, as a retina is. A wall stays put in the
    world while the body turns, so what a column holds has to move to the next column. With
    `flow`, what the body does sets how much a cell takes from each of its nine neighbours,
    a multiplicative input the plain perceptron does not have, and the perceptron adds to
    the result. Nothing says which way or how far: only prediction teaches that.

    Passes grow with the columns (side // 8), so a pass moves the same number of cells at
    any size and the same weights run on a bigger ring: `load_state_dict` works across sizes.
    """

    N_RAYS, RAY_CH, D_RAY, D_RAY_OUT = 32, 6, 16, 16

    def __init__(self, side=16, rows=4, ch=8, hidden=24, flow=True, conserve=False, mirror=False,
                 once=False, stable=False, seed=0):
        super().__init__()
        self.mirror, self.stable = mirror, stable
        self.once = once             # the cell's own rule fires once a frame, whatever the size
        self.register_buffer("turn_back", torch.tensor([1.0, -1.0, 1.0]), persistent=False)
        self.side, self.rows, self.ch, self.flow = side, rows, ch, flow
        self.conserve = conserve     # the gain hears only the action and keeps what it carries
        self.register_buffer("from_col", torch.empty(0), persistent=False)   # (rays, columns)
        self.register_buffer("to_col", torch.empty(0), persistent=False)     # (columns, rays)
        self.set_side(side)
        ang = torch.linspace(-1, 1, self.N_RAYS)
        self.register_buffer("ang", torch.stack([ang, torch.sin(1.57 * ang),
                                                 torch.cos(1.57 * ang)], 1))
        self.register_buffer("row_pos", torch.linspace(-1, 1, rows)[None, None, :, None])
        self.register_buffer("around", torch.eye(9).reshape(9, 1, 3, 3).repeat(ch, 1, 1, 1))
        n_static = N_SENS - self.N_RAYS * self.RAY_CH
        self.static_in = nn.Sequential(nn.Linear(n_static + N_ACT + 1, D_PROJ), nn.Tanh())
        self.ray_in = nn.Linear(self.RAY_CH + 3, self.D_RAY)
        self.cell = nn.Sequential(nn.Conv2d(9 * ch + D_PROJ + 1 + self.D_RAY, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, ch, 1))
        nn.init.zeros_(self.cell[-1].weight)
        nn.init.zeros_(self.cell[-1].bias)
        self.gain = nn.Sequential(nn.Linear(N_ACT + 1, 16), nn.Tanh(), nn.Linear(16, 9 * ch))
        nn.init.zeros_(self.gain[-1].weight)            # at birth a cell just keeps what it has
        nn.init.zeros_(self.gain[-1].bias)
        keep = torch.zeros(9)
        keep[4] = 1.0
        self.register_buffer("keep", keep)
        self.value = nn.Conv2d(ch, self.D_RAY_OUT, 1)
        self.ray_dec = nn.Linear(self.D_RAY_OUT + 3, self.RAY_CH)
        self.gate = nn.Conv2d(ch, 1, 1)
        self.static_val = nn.Conv2d(ch, D_PROJ, 1)
        self.static_dec = nn.Linear(D_PROJ, n_static)

    def set_side(self, side: int) -> None:
        """Grow or shrink the ring: only the wiring changes, never a weight."""
        self.side, self.passes = side, max(1, side // 8)
        fov = 2.0943951023931953
        phi = -fov / 2 + (fov / 64) * (0.5 + 2 * torch.arange(self.N_RAYS).float())
        at = (phi / (2 * 3.14159265358979) * side) % side
        c0 = at.floor().long()
        frac = at - c0
        rays = torch.arange(self.N_RAYS)
        share = torch.zeros(self.N_RAYS, side)
        share[rays, c0 % side] += 1 - frac
        share[rays, (c0 + 1) % side] += frac
        self.from_col = share
        self.to_col = (share / share.sum(0, keepdim=True).clamp(min=1e-6)).T

    def core_weights(self) -> int:
        return sum(p.numel() for m in (self.cell, self.gain) for p in m.parameters())

    def neighbours(self, s):
        p = F.pad(F.pad(s, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
        return F.conv2d(p, self.around, groups=self.ch)

    def belief(self, s):
        """(b, ch, rows, columns) -> what each column expects a central ray to see, normalised."""
        v = self.value(s).mean(2).transpose(1, 2)
        mid = self.ang[self.N_RAYS // 2].expand(*v.shape[:2], -1)
        return self.ray_dec(torch.cat([v, mid], 2))

    def run(self, sens, act, eyes, keep=None):
        b, t = sens.shape[:2]
        n, c_ray, h, w = self.N_RAYS, self.RAY_CH, self.rows, self.side
        s = torch.zeros(b, self.ch, h, w, device=sens.device)
        row = self.row_pos.expand(b, 1, h, w)
        ang = self.ang.expand(b, -1, -1)
        none = torch.zeros(b, self.D_RAY, h, w, device=sens.device)
        preds = []
        for i in range(t):
            e = eyes[:, i, None]
            rays = sens[:, i, :n * c_ray].reshape(b, n, c_ray) * e[:, None]
            r = self.ray_in(torch.cat([rays, ang], 2))                  # (b, rays, D_RAY)
            heard = (self.to_col @ r).transpose(1, 2)[:, :, None].expand(-1, -1, h, -1)
            body = torch.cat([act[:, i], e], 1)
            static = torch.cat([sens[:, i, n * c_ray:212] * e, sens[:, i, 212:], body], 1)
            code = self.static_in(static)[:, :, None, None].expand(-1, -1, h, w)
            if self.conserve:
                one = torch.ones_like(e)
                g = self.gain(torch.cat([act[:, i], one], 1)).reshape(b, self.ch, 9)
                if self.mirror:      # turning left is turning right in a mirror, nothing more
                    other = self.gain(torch.cat([act[:, i] * self.turn_back, one], 1))
                    g = 0.5 * (g + other.reshape(b, self.ch, 3, 3).flip(3).reshape(b, self.ch, 9))
                if self.stable:      # all shares >= 0 and sum to 1: no frequency can ever grow
                    gain = torch.softmax(g + 4 * self.keep, 2)[:, :, :, None, None]
                else:
                    gain = (g - g.mean(2, keepdim=True) + self.keep)[:, :, :, None, None]
            else:
                g = self.gain(body).reshape(b, self.ch, 9)
                gain = (g + self.keep)[:, :, :, None, None]
            for p in range(self.passes):
                if p == 0 or not self.once:
                    x = torch.cat([self.neighbours(s), code, row, heard if p == 0 else none], 1)
                    s = s + 0.3 * torch.tanh(self.cell(x))
                if self.flow:
                    s = (self.neighbours(s).reshape(b, self.ch, 9, h, w) * gain).sum(2)
                s = s.clamp(-4, 4)
            if keep is not None:
                keep.append(s)
            v = self.value(s).mean(2).transpose(1, 2)                   # (b, columns, D_RAY_OUT)
            ray = self.ray_dec(torch.cat([self.from_col @ v, ang], 2)).reshape(b, n * c_ray)
            g = torch.softmax(self.gate(s).flatten(1), 1)
            st = (self.static_val(s).flatten(2) * g[:, None]).sum(2)
            preds.append(torch.cat([ray, self.static_dec(st)], 1))
        return torch.stack(preds, 1), None


class PolarBrain(nn.Module):
    """The flow ring grown into a 2-D sheet: columns are directions from the body, rows are
    distances, so walking can be expressed as well as turning.

    A step moves a nearby wall across many columns and a far one across few, so unlike the
    ring the share a cell takes from each neighbour depends on where the cell is as well as on
    what the body does. Nothing states the law: the net hears the motion and the cell's place
    and has to find it. Shares are non-negative, so no frequency can grow, and left is the
    mirror of right. Rays land on whole columns and each cell decides what that means at its
    distance. A ray is read through attention over the rows of its columns, which does not
    fade as the rows multiply. All weights are size-free: `set_side` changes only the wiring.
    """

    N_RAYS, RAY_CH, D_RAY, D_RAY_OUT = 32, 6, 16, 16
    SPEED = 213                  # the felt speed among the inner senses
    REACH, CELLS_PER_FRAME, RAD_PER_FRAME = 22.0, 2.1, 0.72   # the world's numbers, ablation only

    def __init__(self, side=16, ch=8, hidden=24, linear=False, velocity=False, physics=False,
                 hand_write=False, seed=0):
        super().__init__()
        self.hand_write = hand_write   # ablation: channels 0, 1 = wall, free, painted by hand
        self.register_buffer("norm_mean", torch.zeros(N_SENS))
        self.register_buffer("norm_std", torch.ones(N_SENS))
        self.ch, self.linear = ch, linear or velocity  # flow = motion x a learned map of the place,
        self.velocity = velocity     # that map is a speed: at speed 0 a cell keeps all it holds
        self.physics = physics       # an ablation: the true flow by hand, only writing and reading learn
        for name in ("from_col", "to_col", "row_pos", "place"):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self.set_side(side)
        ang = torch.linspace(-1, 1, self.N_RAYS)
        self.register_buffer("ang", torch.stack([ang, torch.sin(1.57 * ang),
                                                 torch.cos(1.57 * ang)], 1))
        self.register_buffer("around", torch.eye(9).reshape(9, 1, 3, 3).repeat(ch, 1, 1, 1))
        # thrust, turn, interact, felt speed; sin, cos of the column, distance, closeness
        self.register_buffer("mirrored", torch.tensor([1.0, -1, 1, 1, -1, 1, 1, 1])[None, :, None, None],
                             persistent=False)
        keep = torch.zeros(9)
        keep[4] = 1.0
        self.register_buffer("keep", keep[None, None, :, None, None])
        n_static = N_SENS - self.N_RAYS * self.RAY_CH
        self.static_in = nn.Sequential(nn.Linear(n_static + N_ACT + 1, D_PROJ), nn.Tanh())
        self.ray_in = nn.Linear(self.RAY_CH + 3, self.D_RAY)
        self.cell = nn.Sequential(nn.Conv2d(9 * ch + D_PROJ + 1 + self.D_RAY, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, ch, 1))
        nn.init.zeros_(self.cell[-1].weight)
        nn.init.zeros_(self.cell[-1].bias)
        if physics:
            self.gain = nn.Identity()
        elif self.linear:
            per_place = 2 if velocity else 9
            self.gain = nn.Sequential(nn.Conv2d(4, 16, 1), nn.Tanh(),
                                      nn.Conv2d(16, N_ACT * per_place * ch, 1))
            if not velocity:
                self.sharp = nn.Parameter(torch.full((1, ch, 1, 1, 1), 4.0))  # how firmly a cell keeps
            self.register_buffer("mirror_place", torch.tensor([-1.0, 1, 1, 1])[None, :, None, None],
                                 persistent=False)
            self.register_buffer("mirror_act", torch.tensor([1.0, -1, 1])[None, :, None, None, None, None],
                                 persistent=False)
            # in a mirror a speed along the rows stays, a speed along the columns changes sign
            self.register_buffer("mirror_speed", torch.tensor([1.0, -1])[None, None, None, :, None, None],
                                 persistent=False)
        else:
            self.gain = nn.Sequential(nn.Conv2d(8, 24, 1), nn.Tanh(), nn.Conv2d(24, 9 * ch, 1))
        if velocity and not physics:  # not exactly 0: the shares have a kink at speed 0, no gradient
            nn.init.normal_(self.gain[-1].weight, std=0.01)
        elif not physics:
            nn.init.zeros_(self.gain[-1].weight)
        if not physics:
            nn.init.zeros_(self.gain[-1].bias)
        self.value = nn.Conv2d(ch + 1, self.D_RAY_OUT, 1)
        self.attend = nn.Conv2d(ch + 1, 1, 1)
        self.ray_dec = nn.Linear(self.D_RAY_OUT + 3, self.RAY_CH)
        self.gate = nn.Conv2d(ch, 1, 1)
        self.static_val = nn.Conv2d(ch, D_PROJ, 1)
        self.static_dec = nn.Linear(D_PROJ, n_static)

    def set_side(self, side: int) -> None:
        self.side, self.rows, self.passes = side, max(4, side // 4), max(1, side // 8)
        fov = 2.0943951023931953
        phi = -fov / 2 + (fov / 64) * (0.5 + 2 * torch.arange(self.N_RAYS).float())
        at = (phi / (2 * 3.14159265358979) * side) % side
        c0 = at.floor().long()
        frac = at - c0
        rays = torch.arange(self.N_RAYS)
        share = torch.zeros(self.N_RAYS, side)
        share[rays, c0 % side] += 1 - frac
        share[rays, (c0 + 1) % side] += frac
        self.from_col = share
        self.to_col = (share / share.sum(0, keepdim=True).clamp(min=1e-6)).T
        r = (torch.arange(self.rows).float() + 0.5) / self.rows
        theta = torch.arange(side).float() / side * 2 * 3.14159265358979
        rr, tt = torch.meshgrid(r, theta, indexing="ij")
        self.row_pos = r[None, None, :, None]
        self.place = torch.stack([torch.sin(tt), torch.cos(tt), rr, 1 / (1 + 8 * rr)])[None]

    def core_weights(self) -> int:
        return sum(p.numel() for m in (self.cell, self.gain) for p in m.parameters())

    def neighbours(self, s):
        p = F.pad(F.pad(s, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate")
        return F.conv2d(p, self.around, groups=self.ch)

    def flow_maps(self):
        """Per action and place: logits of the shares, the mirrored twin already folded in."""
        h, w = self.rows, self.side
        if self.velocity:
            a = self.gain(self.place).reshape(1, N_ACT, self.ch, 2, h, w)
            twin = self.gain(self.place * self.mirror_place).reshape(1, N_ACT, self.ch, 2, h, w)
            return 0.5 * (a + twin * self.mirror_act * self.mirror_speed)
        a = self.gain(self.place).reshape(1, N_ACT, self.ch, 9, h, w)
        twin = self.gain(self.place * self.mirror_place).reshape(1, N_ACT, self.ch, 3, 3, h, w)
        twin = twin.flip(4).reshape(1, N_ACT, self.ch, 9, h, w)
        return 0.5 * (a + twin * self.mirror_act)

    def set_norm(self, mean, std) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean))
        self.norm_std.copy_(torch.as_tensor(std))

    @torch.no_grad()
    def paint(self, s, sens, e):
        """Write this frame's view by hand: every cell a ray crossed is free, its end a wall."""
        b = sens.shape[0]
        h, w = self.rows, self.side
        raw = sens * self.norm_std + self.norm_mean
        near = raw[:, 4::self.RAY_CH][:, :self.N_RAYS]
        solid = raw[:, 5::self.RAY_CH][:, :self.N_RAYS]
        fed = self.to_col.sum(1) > 0
        col_near = (self.to_col @ near[:, :, None])[:, :, 0]         # (b, columns)
        col_solid = (self.to_col @ solid[:, :, None])[:, :, 0]
        dist = (1 - col_near).clamp(0, 1)                            # as a share of the reach
        r = self.row_pos[0, 0, :, 0][None, :, None]                  # (1, rows, 1)
        looked = (e[:, 0, None, None] > 0.5) & fed[None, None, :]
        free = (r < dist[:, None]) & looked
        at = (dist * h).long().clamp(max=h - 1)
        hit = torch.zeros(b, h, w, dtype=torch.bool, device=s.device)
        hit.scatter_(1, at[:, None], (col_solid > 0.5)[:, None])
        hit = hit & looked
        out = s.clone()
        out[:, 0] = torch.where(free, torch.zeros_like(s[:, 0]), s[:, 0])
        out[:, 1] = torch.where(free, torch.ones_like(s[:, 1]), s[:, 1])
        out[:, 0] = torch.where(hit, torch.ones_like(s[:, 0]), out[:, 0])
        out[:, 1] = torch.where(hit, torch.zeros_like(s[:, 1]), out[:, 1])
        return out

    def shape_shift(self, dy, dx):
        """The nine shares of a shift of (dy, dx) cells; positive = taken from the next cell."""
        b, ch, h, w = dy.shape
        wy = torch.stack([F.relu(-dy), 1 - dy.abs(), F.relu(dy)], 2)
        wx = torch.stack([F.relu(-dx), 1 - dx.abs(), F.relu(dx)], 2)
        return (wy[:, :, :, None] * wx[:, :, None]).reshape(b, ch, 9, h, w)

    def true_flow(self, motion):
        """A point standing still in the world, for a body doing `motion` over one frame."""
        b, h, w = motion.shape[0], self.rows, self.side
        thrust, turn = motion[:, 0, None, None], motion[:, 1, None, None]
        sin, cos, r = self.place[:, 0], self.place[:, 1], self.place[:, 2]
        cells = 2 * 3.14159265358979 / w
        walk = thrust * self.CELLS_PER_FRAME
        dy = walk * cos * (h / self.REACH) / self.passes
        dx = (turn * self.RAD_PER_FRAME - walk * sin / (r * self.REACH)) / cells / self.passes
        return dy.clamp(-1, 1), dx.clamp(-1, 1)

    def shares(self, motion, maps=None):
        b, h, w = motion.shape[0], self.rows, self.side
        if self.physics:
            dy, dx = self.true_flow(motion)
            s = self.shape_shift(dy[:, None].expand(-1, self.ch, -1, -1), dx[:, None].expand(-1, self.ch, -1, -1))
            s = s.clone()                   # the last two channels stay fixed to the body
            s[:, -2:] = self.keep[0, 0, :, 0, 0][None, None, :, None, None]
            return s
        if self.linear:
            maps = self.flow_maps() if maps is None else maps
            g = (motion[:, :N_ACT, None, None, None, None] * maps).sum(1)
            if self.velocity:
                d = torch.tanh(g)
                return self.shape_shift(d[:, :, 0], d[:, :, 1])
            return torch.softmax(g + self.sharp * self.keep, 2)
        x = torch.cat([motion[:, :, None, None].expand(-1, -1, h, w), self.place.expand(b, -1, -1, -1)], 1)
        g = self.gain(x).reshape(b, self.ch, 9, h, w)
        other = self.gain(x * self.mirrored).reshape(b, self.ch, 3, 3, h, w).flip(3).reshape(b, self.ch, 9, h, w)
        return torch.softmax(0.5 * (g + other) + 4 * self.keep, 2)

    def run(self, sens, act, eyes, keep=None):
        b, t = sens.shape[:2]
        n, c_ray, h, w = self.N_RAYS, self.RAY_CH, self.rows, self.side
        s = torch.zeros(b, self.ch, h, w, device=sens.device)
        row = self.row_pos.expand(b, 1, h, w)
        ang = self.ang.expand(b, -1, -1)
        maps = self.flow_maps() if self.linear else None
        preds = []
        for i in range(t):
            e = eyes[:, i, None]
            rays = sens[:, i, :n * c_ray].reshape(b, n, c_ray) * e[:, None]
            r = self.ray_in(torch.cat([rays, ang], 2))
            heard = (self.to_col @ r).transpose(1, 2)[:, :, None].expand(-1, -1, h, -1)
            body = torch.cat([act[:, i], e], 1)
            static = torch.cat([sens[:, i, n * c_ray:212] * e, sens[:, i, 212:], body], 1)
            code = self.static_in(static)[:, :, None, None].expand(-1, -1, h, w)
            s_before = s
            s = s + 0.3 * torch.tanh(self.cell(torch.cat([self.neighbours(s), code, row, heard], 1)))
            if self.hand_write:
                keep_hand = self.paint(s_before, sens[:, i], e)[:, :2]
                s = torch.cat([keep_hand, s[:, 2:]], 1)
            shares = self.shares(torch.cat([act[:, i], sens[:, i, self.SPEED, None]], 1), maps)
            for _ in range(self.passes):
                s = (self.neighbours(s).reshape(b, self.ch, 9, h, w) * shares).sum(2)
            s = s.clamp(-4, 4)
            if keep is not None:
                keep.append(s)
            x = torch.cat([s, row], 1)
            v = (self.value(x) * torch.softmax(self.attend(x), 2)).sum(2).transpose(1, 2)
            ray = self.ray_dec(torch.cat([self.from_col @ v, ang], 2)).reshape(b, n * c_ray)
            g = torch.softmax(self.gate(s).flatten(1), 1)
            st = (self.static_val(s).flatten(2) * g[:, None]).sum(2)
            preds.append(torch.cat([ray, self.static_dec(st)], 1))
        return torch.stack(preds, 1), None


class NorthBrain(nn.Module):
    """A square of cells fixed to the compass, centred on the body. Steps move the content by WHOLE
    cells only, once the body has crossed into the next one: anything smoother blurs the memory
    (the measured reason smooth flow failed), so the jump is by hand here and the rest of the way
    is kept as the body's offset inside its cell. Turning moves nothing.

    What learns is writing and reading. A cell knows its direction relative to the body, so which
    ray looks its way changes as the body turns: live wiring, not a fixed address. A ray is read by
    marching along it, each cell may stop it, as walls do. All weights are size-free.
    """

    N_RAYS, RAY_CH, D_RAY, D_RAY_OUT = 32, 6, 16, 16
    REACH, CELLS_PER_FRAME, RAD_PER_FRAME = 22.0, 2.1, 0.72
    SIN_H, COS_H = 215, 216
    FOV = 2.0943951023931953

    def __init__(self, side=32, ch=8, hidden=24, render=False, seed=0):
        super().__init__()
        self.ch = ch
        # render: how near a ray ends and whether it hit come ONLY from where the cells stopped it,
        # as in a renderer, so to predict well the cells must become opaque where walls are
        self.render = render
        for name in ("off_y", "off_x", "dist", "theta", "t_k"):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self.set_side(side)
        self.register_buffer("norm_mean", torch.zeros(N_SENS))
        self.register_buffer("norm_std", torch.ones(N_SENS))
        ang = torch.linspace(-1, 1, self.N_RAYS)
        self.register_buffer("ang", torch.stack([ang, torch.sin(1.57 * ang),
                                                 torch.cos(1.57 * ang)], 1))
        self.register_buffer("around", torch.eye(9).reshape(9, 1, 3, 3).repeat(ch, 1, 1, 1))
        phi = -self.FOV / 2 + (self.FOV / 64) * (0.5 + 2 * torch.arange(self.N_RAYS).float())
        self.register_buffer("phi", phi, persistent=False)
        n_static = N_SENS - self.N_RAYS * self.RAY_CH
        self.static_in = nn.Sequential(nn.Linear(n_static + N_ACT + 1, D_PROJ), nn.Tanh())
        self.ray_in = nn.Linear(self.RAY_CH + 3, self.D_RAY)
        self.cell = nn.Sequential(nn.Conv2d(9 * ch + D_PROJ + 3 + self.D_RAY, hidden, 1), nn.ReLU(),
                                  nn.Conv2d(hidden, ch, 1))
        nn.init.zeros_(self.cell[-1].weight)
        nn.init.zeros_(self.cell[-1].bias)
        self.value = nn.Conv2d(ch + 1, self.D_RAY_OUT, 1)
        self.stop = nn.Conv2d(ch if render else ch + 1, 1, 1)
        nn.init.constant_(self.stop.bias, -2.0)   # a ray starts out going through
        self.beyond = nn.Parameter(torch.zeros(self.D_RAY_OUT))
        self.ray_dec = nn.Linear(self.D_RAY_OUT + 3, self.RAY_CH)
        self.gate = nn.Conv2d(ch, 1, 1)
        self.static_val = nn.Conv2d(ch, D_PROJ, 1)
        self.static_dec = nn.Linear(D_PROJ, n_static)

    def set_side(self, side: int) -> None:
        self.side = side
        self.cell_size = 2 * self.REACH / side
        off = (torch.arange(side).float() + 0.5) * self.cell_size - self.REACH
        self.off_y, self.off_x = torch.meshgrid(off, off, indexing="ij")
        self.dist = torch.hypot(self.off_y, self.off_x) / self.REACH
        self.theta = torch.atan2(self.off_y, self.off_x)
        k = max(16, side) if self.render else max(8, side // 2)
        self.t_k = (torch.arange(k).float() + 0.5) / k * self.REACH

    def set_norm(self, mean, std) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean))
        self.norm_std.copy_(torch.as_tensor(std))

    def core_weights(self) -> int:
        return sum(p.numel() for p in self.cell.parameters())

    def neighbours(self, s):
        return F.conv2d(F.pad(s, (1, 1, 1, 1)), self.around, groups=self.ch)

    def rays_to_cells(self, r, rel):
        """(b, rays, D) onto the sheet: each cell hears the ray looking its way, or nothing."""
        spacing = self.FOV / self.N_RAYS
        at = (rel - self.phi[0]) / spacing
        low = at.floor().clamp(0, self.N_RAYS - 2).long()
        f = (at - low.float()).clamp(0, 1)[:, None]
        b, h, w = rel.shape
        i0 = low.reshape(b, -1)[:, :, None].expand(-1, -1, self.D_RAY)
        v0 = r.gather(1, i0).transpose(1, 2).reshape(b, self.D_RAY, h, w)
        v1 = r.gather(1, i0 + 1).transpose(1, 2).reshape(b, self.D_RAY, h, w)
        looks = (rel.abs() <= self.FOV / 2)[:, None].float()
        return (v0 * (1 - f) + v1 * f) * looks

    def believed_solid(self, s, ry, rx, ay, ax):
        """The brain's own belief, per point: would a ray stop here? Points are (b, n) offsets from
        the body in world cells. Nobody ever showed it a map: this is only what predicting its
        senses made of `stop`."""
        grid = torch.stack([(rx + ax[:, None]) / self.REACH, (ry + ay[:, None]) / self.REACH], 2)[:, None]
        got = F.grid_sample(s, grid, align_corners=False, padding_mode="zeros")      # (b, ch, 1, n)
        if self.render:              # the chance a ray is stopped within one world cell
            return (1 - torch.exp(-F.softplus(self.stop(got))))[:, 0, 0]
        depth = (torch.hypot(ry, rx) / self.REACH)[:, None, None]
        return torch.sigmoid(self.stop(torch.cat([got, depth], 1)))[:, 0, 0]

    def read_rays(self, s, heading, ay, ax):
        """March each ray of the next frame across the sheet, stopping where cells stop it."""
        b = s.shape[0]
        a = heading[:, None] + self.phi[None]                  # (b, rays)
        t = self.t_k[None, None]                               # (1, 1, K)
        gy = (torch.sin(a)[:, :, None] * t + ay[:, None, None]) / self.REACH
        gx = (torch.cos(a)[:, :, None] * t + ax[:, None, None]) / self.REACH
        grid = torch.stack([gx, gy], 3)                        # grid_sample wants (x, y)
        got = F.grid_sample(s, grid, align_corners=False, padding_mode="zeros")   # (b, ch, rays, K)
        depth = (self.t_k / self.REACH)[None, None, None].expand(b, 1, self.N_RAYS, -1)
        x = torch.cat([got, depth], 1)
        if self.render:              # a density, so the same cell stops a ray alike at any size
            step = self.REACH / len(self.t_k)
            p = (1 - torch.exp(-F.softplus(self.stop(got)) * step))[:, 0]
        else:
            p = torch.sigmoid(self.stop(x))[:, 0]              # (b, rays, K)
        through = torch.cumprod(1 - p + 1e-6, 2)
        before = torch.cat([torch.ones_like(through[:, :, :1]), through[:, :, :-1]], 2)
        w = p * before                                         # the share of the ray stopped here
        v = (self.value(x).permute(0, 2, 3, 1) * w[:, :, :, None]).sum(2)
        v = v + self.beyond[None, None] * through[:, :, -1:]
        if not self.render:
            return v
        near = (w * (1 - self.t_k / self.REACH)[None, None]).sum(2)     # (b, rays), raw units
        hit = 1 - through[:, :, -1]
        return v, near, hit

    def write(self, s, sens_i, act_i, e):
        """What one frame's senses do to the sheet. Returns the sheet and the felt heading."""
        b = s.shape[0]
        n, c_ray, side = self.N_RAYS, self.RAY_CH, self.side
        ang = self.ang.expand(b, -1, -1)
        raw = sens_i * self.norm_std + self.norm_mean
        head = torch.atan2(raw[:, self.SIN_H], raw[:, self.COS_H])
        rays = sens_i[:, :n * c_ray].reshape(b, n, c_ray) * e[:, None]
        r = self.ray_in(torch.cat([rays, ang], 2))
        rel = (self.theta[None] - head[:, None, None] + 3.14159265358979) % (2 * 3.14159265358979) - 3.14159265358979
        heard = self.rays_to_cells(r, rel) * e[:, :, None, None]
        body = torch.cat([act_i, e], 1)
        static = torch.cat([sens_i[:, n * c_ray:212] * e, sens_i[:, 212:], body], 1)
        code = self.static_in(static)[:, :, None, None].expand(-1, -1, side, side)
        place = torch.stack([torch.sin(rel), torch.cos(rel)], 1)
        x = torch.cat([self.neighbours(s), code, self.dist.expand(b, 1, side, side), place, heard], 1)
        return (s + 0.3 * torch.tanh(self.cell(x))).clamp(-4, 4), head

    def jump(self, s, ay, ax, head, act_i):
        """The body moves; the sheet jumps when it has crossed into the next cell. Counted from the
        command and the felt heading only. Returns the sheet, the offset and the new heading."""
        b, side = s.shape[0], self.side
        turned = head + act_i[:, 1] * self.RAD_PER_FRAME
        mid = head + act_i[:, 1] * self.RAD_PER_FRAME / 2
        walk = act_i[:, 0].clamp(0, 1) * self.CELLS_PER_FRAME
        ay = ay + walk * torch.sin(mid)
        ax = ax + walk * torch.cos(mid)
        jy = torch.round(ay / self.cell_size).long()
        jx = torch.round(ax / self.cell_size).long()
        if (jy != 0).any() or (jx != 0).any():
            parts = []
            for k in range(b):
                p = s[k]
                if jy[k] or jx[k]:
                    p = torch.roll(p, (-int(jy[k]), -int(jx[k])), (1, 2))
                    keep_y = torch.ones(side, device=s.device)
                    keep_x = torch.ones(side, device=s.device)
                    if jy[k] > 0:
                        keep_y[-int(jy[k]):] = 0
                    elif jy[k] < 0:
                        keep_y[:-int(jy[k])] = 0
                    if jx[k] > 0:
                        keep_x[-int(jx[k]):] = 0
                    elif jx[k] < 0:
                        keep_x[:-int(jx[k])] = 0
                    p = p * keep_y[None, :, None] * keep_x[None, None, :]
                parts.append(p)
            s = torch.stack(parts)
            ay = ay - jy.float() * self.cell_size
            ax = ax - jx.float() * self.cell_size
        return s, ay, ax, turned

    def run(self, sens, act, eyes, keep=None):
        b, t = sens.shape[:2]
        n, c_ray, side = self.N_RAYS, self.RAY_CH, self.side
        s = torch.zeros(b, self.ch, side, side, device=sens.device)
        ay, ax = torch.zeros(b), torch.zeros(b)
        ang = self.ang.expand(b, -1, -1)
        preds = []
        for i in range(t):
            s, head = self.write(s, sens[:, i], act[:, i], eyes[:, i, None])
            if keep is not None:
                keep.append({"s": s, "ay": ay, "ax": ax, "head": head})
            s, ay, ax, turned = self.jump(s, ay, ax, head, act[:, i])
            v = self.read_rays(s, turned, ay, ax)
            if self.render:
                v, near, hit = v
            ray = self.ray_dec(torch.cat([v, ang], 2))             # (b, rays, RAY_CH)
            if self.render:          # the geometry of the view is the rendering, nothing else
                mean, std = self.norm_mean[:n * c_ray].reshape(n, c_ray), self.norm_std[:n * c_ray].reshape(n, c_ray)
                ray = torch.cat([ray[:, :, :4], ((near - mean[:, 4]) / std[:, 4])[:, :, None],
                                 ((hit - mean[:, 5]) / std[:, 5])[:, :, None]], 2)
            ray = ray.reshape(b, n * c_ray)
            g = torch.softmax(self.gate(s).flatten(1), 1)
            st = (self.static_val(s).flatten(2) * g[:, None]).sum(2)
            preds.append(torch.cat([ray, self.static_dec(st)], 1))
        return torch.stack(preds, 1), None


NORTH = {"north": dict(), "northr": dict(render=True)}
POLAR = {"polar": dict(), "polarl": dict(linear=True), "polarv": dict(velocity=True),
         "polarp": dict(physics=True), "polarh": dict(physics=True, hand_write=True)}
FLOW = {"flow": dict(flow=True), "ring": dict(flow=False),
        "flowc": dict(flow=True, conserve=True),
        "flowm": dict(flow=True, conserve=True, mirror=True),
        "flowo": dict(flow=True, conserve=True, mirror=True, once=True),
        "flows": dict(flow=True, conserve=True, mirror=True, once=True, stable=True)}

ARMS = {
    "now":       lambda seed, side: Now(48),
    "gru16":     lambda seed, side: Rec(16),
    "gru128":    lambda seed, side: Rec(128),
    "sheet":     lambda seed, side: Sheet(side, hidden=24, seed=seed),
    "types":     lambda seed, side: Sheet(side, hidden=6, types=4, seed=seed),
    "long":      lambda seed, side: Sheet(side, hidden=21, long=2, seed=seed),
    "typeslong": lambda seed, side: Sheet(side, hidden=5, types=4, long=2, seed=seed),
    "wide128":   lambda seed, side: WideRec(128),
    "wide512":   lambda seed, side: WideRec(512),
}
RETINA = {
    "rsheet":     dict(hidden=24),
    "rtypes":     dict(hidden=6, types=4),
    "rlong":      dict(hidden=21, long=2),
    "rtypeslong": dict(hidden=5, types=4, long=2),
}
PLAIN = ("now", "gru16", "gru128", "wide128", "wide512")


def build(arm: str, seed: int, side: int) -> nn.Module:
    if arm in NORTH:
        return NorthBrain(side, seed=seed, **NORTH[arm])
    if arm in POLAR:
        return PolarBrain(side, seed=seed, **POLAR[arm])
    if arm in FLOW:
        return FlowBrain(side, seed=seed, **FLOW[arm])
    if arm in RETINA:
        return RetinaBrain(side, seed=seed, **RETINA[arm])
    return Brain(ARMS[arm](seed, side))
