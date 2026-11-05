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
    if arm in RETINA:
        return RetinaBrain(side, seed=seed, **RETINA[arm])
    return Brain(ARMS[arm](seed, side))
