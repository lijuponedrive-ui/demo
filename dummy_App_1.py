"""
Kerala Climate Design Tool — 33-parameter schema demo.

Run:
    streamlit run demo_app.py

IMPORTANT:
The checkpoint used by this application must have been trained with:

    33 input parameters
    33 -> 32 -> 64 -> 280

The checkpoint reported in the original error has exactly this structure.
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle
import plotly.graph_objects as go


# ============================================================
# Configuration
# ============================================================

NX, NY = 14, 10

# IMPORTANT:
# The checkpoint has [32, 33] at net.0.weight.
# Therefore the model MUST have 33 input parameters.
N_PARAMS = 33

# Checkpoint has [280, 64] at net.6.weight.
N_OUT = 2 * NX * NY

MODEL_PATH = "unet_run/best_model.pt"


# ============================================================
# Model
# ============================================================

class SmallSurrogate(nn.Module):
    """
    Must exactly match the architecture used during training.

    Checkpoint architecture:

        33 -> 32 -> 64 -> 280

    Therefore:

        Linear(33, 32)
        Linear(32, 64)
        Linear(64, 280)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_PARAMS, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, N_OUT),
        )

    def forward(self, p):
        return self.net(p).view(-1, 2, NX, NY)


# ============================================================
# Checkpoint compatibility
# ============================================================

def check_checkpoint_architecture(state_dict):
    """
    Verify that the checkpoint matches the expected
    33 -> 32 -> 64 -> 280 architecture.
    """

    required_shapes = {
        "net.0.weight": (32, 33),
        "net.0.bias": (32,),
        "net.3.weight": (64, 32),
        "net.3.bias": (64,),
        "net.6.weight": (280, 64),
        "net.6.bias": (280,),
    }

    errors = []

    for name, expected_shape in required_shapes.items():

        if name not in state_dict:
            errors.append(
                f"Missing parameter '{name}' in checkpoint."
            )
            continue

        actual_shape = tuple(state_dict[name].shape)

        if actual_shape != expected_shape:
            errors.append(
                f"{name}: checkpoint has {actual_shape}, "
                f"but application expects {expected_shape}."
            )

    if errors:
        raise RuntimeError(
            "\n\n".join([
                "MODEL/CHECKPOINT ARCHITECTURE MISMATCH.",
                "",
                *errors,
                "",
                "Expected architecture:",
                "33 inputs -> 32 -> 64 -> 280 outputs",
            ])
        )


# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file was not found:\n\n{MODEL_PATH}\n\n"
            "Make sure best_model.pt is inside the unet_run folder."
        )

    ckpt = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state" not in ckpt:
        raise KeyError(
            "The checkpoint does not contain 'model_state'."
        )

    state_dict = ckpt["model_state"]

    # Check architecture BEFORE loading.
    check_checkpoint_architecture(state_dict)

    model = SmallSurrogate()

    model.load_state_dict(state_dict)

    model.eval()

    return model, ckpt


# ============================================================
# Load model
# ============================================================

model, ckpt = load_model()


# ============================================================
# Load normalization values
# ============================================================

param_mean = np.asarray(ckpt["param_mean"], dtype=np.float32)
param_std = np.asarray(ckpt["param_std"], dtype=np.float32)

T_mean = ckpt["T_mean"]
T_std = ckpt["T_std"]

q_mean = ckpt["q_mean"]
q_std = ckpt["q_std"]


# Make sure normalization has 33 parameters.
if len(param_mean) != N_PARAMS:
    raise RuntimeError(
        f"Checkpoint param_mean has {len(param_mean)} values, "
        f"but application expects {N_PARAMS}."
    )

if len(param_std) != N_PARAMS:
    raise RuntimeError(
        f"Checkpoint param_std has {len(param_std)} values, "
        f"but application expects {N_PARAMS}."
    )


# ============================================================
# Prediction
# ============================================================

def predict(params_physical):

    params_physical = np.asarray(
        params_physical,
        dtype=np.float32
    )

    if len(params_physical) != N_PARAMS:
        raise ValueError(
            f"Expected {N_PARAMS} parameters, "
            f"but received {len(params_physical)}."
        )

    p = (
        params_physical - param_mean
    ) / param_std

    p_tensor = torch.from_numpy(
        p.astype(np.float32)
    ).unsqueeze(0)

    with torch.no_grad():

        pred = model(p_tensor).numpy()[0]

    T = pred[0] * T_std + T_mean
    q = pred[1] * q_std + q_mean

    return T, q


# ============================================================
# Physics
# ============================================================

def rh_from_q_T(q, T):

    T_K = T + 273.15

    e_sat = (
        611.2
        * np.exp(
            17.67
            * (T_K - 273.15)
            / (T_K - 29.65)
        )
    )

    e = (
        q
        * 101325.0
        / (0.622 + q)
    )

    return np.clip(
        100.0 * e / e_sat,
        0,
        100
    )


def heat_index(T, RH):

    T_F = T * 9 / 5 + 32

    HI = (
        -42.379
        + 2.04901523 * T_F
        + 10.14333127 * RH
        - 0.22475541 * T_F * RH
        - 6.83783e-3 * T_F * T_F
        - 5.481717e-2 * RH * RH
        + 1.22874e-3 * T_F * T_F * RH
        + 8.5282e-4 * T_F * RH * RH
        - 1.99e-6 * T_F * T_F * RH * RH
    )

    return (HI - 32) * 5 / 9


# ============================================================
# Wind
# ============================================================

def wind_dir_to_name(deg):

    names = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    return names[
        int((deg + 11.25) / 22.5) % 16
    ]


def wind_vector(wind_dir_deg):

    th = math.radians(wind_dir_deg)

    return (
        -math.sin(th),
        -math.cos(th)
    )


# ============================================================
# Room plan
# ============================================================

def draw_room_plan(
    ax,
    Lx,
    Ly,
    win_w_west,
    win_w_east,
    win_w_north,
    win_w_south,
    win_pos_w,
    win_pos_e,
    win_pos_n,
    win_pos_s,
    win_open_west,
    win_open_east,
    win_open_north,
    win_open_south,
    door_wall,
    door_pos,
    door_open,
    hole_wall,
    hole_pos,
    hole_w,
    hole_open,
    wind_dir,
    wind_speed,
    fan_level
):

    C_ROOM = "#2c3e50"

    C_WIN_OPEN = "#27ae60"
    C_WIN_CLOSED = "#95a5a6"

    C_DOOR_OPEN = "#8e44ad"
    C_DOOR_CLOSED = "#95a5a6"

    C_HOLE_OPEN = "#e67e22"
    C_HOLE_CLOSED = "#95a5a6"

    C_FAN = "#e74c3c"

    ZERO = 0.05

    # Room
    ax.add_patch(
        Rectangle(
            (0, 0),
            Lx,
            Ly,
            facecolor="#f7f6f2",
            edgecolor=C_ROOM,
            lw=2.5
        )
    )

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    if Ly > 0.3:

        if win_w_west >= ZERO:

            y_lo = win_pos_w * (
                Ly - win_w_west
            )

            c = (
                C_WIN_OPEN
                if win_open_west
                else C_WIN_CLOSED
            )

            ax.plot(
                [0, 0],
                [y_lo, y_lo + win_w_west],
                color=c,
                lw=6,
                solid_capstyle="butt"
            )

        if win_w_east >= ZERO:

            y_lo = win_pos_e * (
                Ly - win_w_east
            )

            c = (
                C_WIN_OPEN
                if win_open_east
                else C_WIN_CLOSED
            )

            ax.plot(
                [Lx, Lx],
                [y_lo, y_lo + win_w_east],
                color=c,
                lw=6,
                solid_capstyle="butt"
            )

    if Lx > 0.3:

        if win_w_south >= ZERO:

            x_lo = win_pos_s * (
                Lx - win_w_south
            )

            c = (
                C_WIN_OPEN
                if win_open_south
                else C_WIN_CLOSED
            )

            ax.plot(
                [x_lo, x_lo + win_w_south],
                [0, 0],
                color=c,
                lw=6,
                solid_capstyle="butt"
            )

        if win_w_north >= ZERO:

            x_lo = win_pos_n * (
                Lx - win_w_north
            )

            c = (
                C_WIN_OPEN
                if win_open_north
                else C_WIN_CLOSED
            )

            ax.plot(
                [x_lo, x_lo + win_w_north],
                [Ly, Ly],
                color=c,
                lw=6,
                solid_capstyle="butt"
            )

    # --------------------------------------------------------
    # Door
    # --------------------------------------------------------

    dc = (
        C_DOOR_OPEN
        if door_open
        else C_DOOR_CLOSED
    )

    dw = 0.9

    if door_wall == "west" and Ly > dw:

        y_lo = door_pos * (Ly - dw)

        ax.plot(
            [0, 0],
            [y_lo, y_lo + dw],
            color=dc,
            lw=9,
            solid_capstyle="butt"
        )

    elif door_wall == "east" and Ly > dw:

        y_lo = door_pos * (Ly - dw)

        ax.plot(
            [Lx, Lx],
            [y_lo, y_lo + dw],
            color=dc,
            lw=9,
            solid_capstyle="butt"
        )

    elif door_wall == "south" and Lx > dw:

        x_lo = door_pos * (Lx - dw)

        ax.plot(
            [x_lo, x_lo + dw],
            [0, 0],
            color=dc,
            lw=9,
            solid_capstyle="butt"
        )

    elif door_wall == "north" and Lx > dw:

        x_lo = door_pos * (Lx - dw)

        ax.plot(
            [x_lo, x_lo + dw],
            [Ly, Ly],
            color=dc,
            lw=9,
            solid_capstyle="butt"
        )

    # --------------------------------------------------------
    # Air hole
    # --------------------------------------------------------

    if (
        hole_w >= ZERO
        and hole_wall != "ceiling"
    ):

        hc = (
            C_HOLE_OPEN
            if hole_open
            else C_HOLE_CLOSED
        )

        if hole_wall == "west" and Ly > hole_w:

            y_lo = hole_pos * (
                Ly - hole_w
            )

            ax.plot(
                [0, 0],
                [y_lo, y_lo + hole_w],
                color=hc,
                lw=4,
                linestyle=":"
            )

        elif hole_wall == "east" and Ly > hole_w:

            y_lo = hole_pos * (
                Ly - hole_w
            )

            ax.plot(
                [Lx, Lx],
                [y_lo, y_lo + hole_w],
                color=hc,
                lw=4,
                linestyle=":"
            )

        elif hole_wall == "south" and Lx > hole_w:

            x_lo = hole_pos * (
                Lx - hole_w
            )

            ax.plot(
                [x_lo, x_lo + hole_w],
                [0, 0],
                color=hc,
                lw=4,
                linestyle=":"
            )

        elif hole_wall == "north" and Lx > hole_w:

            x_lo = hole_pos * (
                Lx - hole_w
            )

            ax.plot(
                [x_lo, x_lo + hole_w],
                [Ly, Ly],
                color=hc,
                lw=4,
                linestyle=":"
            )

    # --------------------------------------------------------
    # Fan
    # --------------------------------------------------------

    if fan_level != "off":

        ax.add_patch(
            Circle(
                (Lx / 2, Ly / 2),
                0.4,
                facecolor="none",
                edgecolor=C_FAN,
                lw=2,
                linestyle="--"
            )
        )

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    wx, wy = wind_vector(wind_dir)

    cx, cy = Lx / 2, Ly / 2

    arrow_len = (
        0.6
        + 0.22 * min(wind_speed, 8.0)
    )

    tail_x = cx - wx * arrow_len
    tail_y = cy - wy * arrow_len

    head_x = cx + wx * arrow_len * 0.55
    head_y = cy + wy * arrow_len * 0.55

    ax.annotate(
        "",
        xy=(head_x, head_y),
        xytext=(tail_x, tail_y),
        arrowprops=dict(
            arrowstyle="->",
            color="navy",
            lw=3,
            mutation_scale=22
        )
    )

    direction_name = wind_dir_to_name(
        wind_dir
    )

    ax.text(
        tail_x - wx * 0.35,
        tail_y - wy * 0.35,
        f"wind {wind_speed:.1f} m/s\n"
        f"from {direction_name}",
        ha="center",
        va="center",
        fontsize=8,
        color="navy",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="navy",
            alpha=0.9
        )
    )

    # Compass

    ax.text(
        Lx / 2,
        -0.5,
        "S",
        ha="center",
        fontsize=9,
        color="grey"
    )

    ax.text(
        Lx / 2,
        Ly + 0.5,
        "N",
        ha="center",
        fontsize=9,
        color="grey"
    )

    ax.text(
        -0.5,
        Ly / 2,
        "W",
        ha="right",
        fontsize=9,
        color="grey"
    )

    ax.text(
        Lx + 0.5,
        Ly / 2,
        "E",
        ha="left",
        fontsize=9,
        color="grey"
    )

    ax.set_xlim(
        -2.2,
        Lx + 2.2
    )

    ax.set_ylim(
        -2.2,
        Ly + 2.2
    )

    ax.set_aspect("equal")

    ax.set_title(
        "Room Plan (top-down)",
        fontweight="bold"
    )

    ax.axis("off")


# ============================================================
# 3D Room
# ============================================================

def make_room_3d(
    Lx,
    Ly,
    Lz,
    win_w_west,
    win_w_east,
    win_w_north,
    win_w_south,
    win_h,
    win_sill,
    win_pos_w,
    win_pos_e,
    win_pos_n,
    win_pos_s,
    win_open_west,
    win_open_east,
    win_open_north,
    win_open_south,
    door_wall,
    door_pos,
    door_open,
    hole_wall,
    hole_pos,
    hole_w,
    hole_h,
    hole_open,
    wind_dir,
    wind_speed,
    fan_level
):

    C_WIN_OPEN = "rgba(39,174,96,0.9)"
    C_WIN_CLOSED = "rgba(149,165,166,0.5)"

    C_DOOR_OPEN = "rgba(142,68,173,0.9)"
    C_DOOR_CLOSED = "rgba(149,165,166,0.5)"

    C_HOLE_OPEN = "rgba(230,126,34,0.9)"
    C_HOLE_CLOSED = "rgba(149,165,166,0.5)"

    C_EDGE = "rgb(44,62,80)"
    C_FAN = "rgb(231,76,60)"

    ZERO = 0.05

    traces = []

    # --------------------------------------------------------
    # Room edges
    # --------------------------------------------------------

    edges = [
        [(0, 0, 0), (Lx, 0, 0)],
        [(0, Ly, 0), (Lx, Ly, 0)],
        [(0, 0, Lz), (Lx, 0, Lz)],
        [(0, Ly, Lz), (Lx, Ly, Lz)],

        [(0, 0, 0), (0, Ly, 0)],
        [(Lx, 0, 0), (Lx, Ly, 0)],
        [(0, 0, Lz), (0, Ly, Lz)],
        [(Lx, 0, Lz), (Lx, Ly, Lz)],

        [(0, 0, 0), (0, 0, Lz)],
        [(Lx, 0, 0), (Lx, 0, Lz)],
        [(0, Ly, 0), (0, Ly, Lz)],
        [(Lx, Ly, 0), (Lx, Ly, Lz)],
    ]

    xs, ys, zs = [], [], []

    for p1, p2 in edges:

        xs += [
            p1[0],
            p2[0],
            None
        ]

        ys += [
            p1[1],
            p2[1],
            None
        ]

        zs += [
            p1[2],
            p2[2],
            None
        ]

    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(
                color=C_EDGE,
                width=4
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # --------------------------------------------------------
    # Quad helper
    # --------------------------------------------------------

    def add_quad(vertices, color):

        x = [
            v[0]
            for v in vertices
        ]

        y = [
            v[1]
            for v in vertices
        ]

        z = [
            v[2]
            for v in vertices
        ]

        traces.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.9,
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    if Ly > 0.3:

        if win_w_west >= ZERO:

            y_lo = win_pos_w * (
                Ly - win_w_west
            )

            c = (
                C_WIN_OPEN
                if win_open_west
                else C_WIN_CLOSED
            )

            add_quad([
                (0, y_lo, win_sill),
                (0, y_lo + win_w_west, win_sill),
                (
                    0,
                    y_lo + win_w_west,
                    win_sill + win_h
                ),
                (0, y_lo, win_sill + win_h),
            ], c)

        if win_w_east >= ZERO:

            y_lo = win_pos_e * (
                Ly - win_w_east
            )

            c = (
                C_WIN_OPEN
                if win_open_east
                else C_WIN_CLOSED
            )

            add_quad([
                (Lx, y_lo, win_sill),
                (
                    Lx,
                    y_lo + win_w_east,
                    win_sill
                ),
                (
                    Lx,
                    y_lo + win_w_east,
                    win_sill + win_h
                ),
                (
                    Lx,
                    y_lo,
                    win_sill + win_h
                ),
            ], c)

    if Lx > 0.3:

        if win_w_south >= ZERO:

            x_lo = win_pos_s * (
                Lx - win_w_south
            )

            c = (
                C_WIN_OPEN
                if win_open_south
                else C_WIN_CLOSED
            )

            add_quad([
                (x_lo, 0, win_sill),
                (
                    x_lo + win_w_south,
                    0,
                    win_sill
                ),
                (
                    x_lo + win_w_south,
                    0,
                    win_sill + win_h
                ),
                (
                    x_lo,
                    0,
                    win_sill + win_h
                ),
            ], c)

        if win_w_north >= ZERO:

            x_lo = win_pos_n * (
                Lx - win_w_north
            )

            c = (
                C_WIN_OPEN
                if win_open_north
                else C_WIN_CLOSED
            )

            add_quad([
                (x_lo, Ly, win_sill),
                (
                    x_lo + win_w_north,
                    Ly,
                    win_sill
                ),
                (
                    x_lo + win_w_north,
                    Ly,
                    win_sill + win_h
                ),
                (
                    x_lo,
                    Ly,
                    win_sill + win_h
                ),
            ], c)

    # --------------------------------------------------------
    # Door
    # --------------------------------------------------------

    dc = (
        C_DOOR_OPEN
        if door_open
        else C_DOOR_CLOSED
    )

    dw = 0.9
    dh = 2.1

    if door_wall == "west" and Ly > dw:

        y_lo = door_pos * (
            Ly - dw
        )

        add_quad([
            (0, y_lo, 0),
            (0, y_lo + dw, 0),
            (0, y_lo + dw, dh),
            (0, y_lo, dh),
        ], dc)

    elif door_wall == "east" and Ly > dw:

        y_lo = door_pos * (
            Ly - dw
        )

        add_quad([
            (Lx, y_lo, 0),
            (Lx, y_lo + dw, 0),
            (Lx, y_lo + dw, dh),
            (Lx, y_lo, dh),
        ], dc)

    elif door_wall == "south" and Lx > dw:

        x_lo = door_pos * (
            Lx - dw
        )

        add_quad([
            (x_lo, 0, 0),
            (x_lo + dw, 0, 0),
            (x_lo + dw, 0, dh),
            (x_lo, 0, dh),
        ], dc)

    elif door_wall == "north" and Lx > dw:

        x_lo = door_pos * (
            Lx - dw
        )

        add_quad([
            (x_lo, Ly, 0),
            (x_lo + dw, Ly, 0),
            (x_lo + dw, Ly, dh),
            (x_lo, Ly, dh),
        ], dc)

    # --------------------------------------------------------
    # Air hole
    # --------------------------------------------------------

    if (
        hole_w >= ZERO
        and hole_wall != "ceiling"
    ):

        hc = (
            C_HOLE_OPEN
            if hole_open
            else C_HOLE_CLOSED
        )

        z_lo = Lz - hole_h

        if hole_wall == "west" and Ly > hole_w:

            y_lo = hole_pos * (
                Ly - hole_w
            )

            add_quad([
                (0, y_lo, z_lo),
                (0, y_lo + hole_w, z_lo),
                (0, y_lo + hole_w, Lz),
                (0, y_lo, Lz),
            ], hc)

        elif hole_wall == "east" and Ly > hole_w:

            y_lo = hole_pos * (
                Ly - hole_w
            )

            add_quad([
                (Lx, y_lo, z_lo),
                (
                    Lx,
                    y_lo + hole_w,
                    z_lo
                ),
                (
                    Lx,
                    y_lo + hole_w,
                    Lz
                ),
                (Lx, y_lo, Lz),
            ], hc)

        elif hole_wall == "south" and Lx > hole_w:

            x_lo = hole_pos * (
                Lx - hole_w
            )

            add_quad([
                (x_lo, 0, z_lo),
                (
                    x_lo + hole_w,
                    0,
                    z_lo
                ),
                (
                    x_lo + hole_w,
                    0,
                    Lz
                ),
                (x_lo, 0, Lz),
            ], hc)

        elif hole_wall == "north" and Lx > hole_w:

            x_lo = hole_pos * (
                Lx - hole_w
            )

            add_quad([
                (x_lo, Ly, z_lo),
                (
                    x_lo + hole_w,
                    Ly,
                    z_lo
                ),
                (
                    x_lo + hole_w,
                    Ly,
                    Lz
                ),
                (x_lo, Ly, Lz),
            ], hc)

    # --------------------------------------------------------
    # Fan
    # --------------------------------------------------------

    if fan_level != "off":

        r = 0.5

        theta = np.linspace(
            0,
            2 * np.pi,
            40
        )

        traces.append(
            go.Scatter3d(
                x=Lx / 2 + r * np.cos(theta),
                y=Ly / 2 + r * np.sin(theta),
                z=np.full_like(
                    theta,
                    Lz - 0.15
                ),
                mode="lines",
                line=dict(
                    color=C_FAN,
                    width=6
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # Wind arrow
    # --------------------------------------------------------

    wx, wy = wind_vector(
        wind_dir
    )

    cx, cy = Lx / 2, Ly / 2

    span = max(Lx, Ly)

    tail = (
        cx - wx * span * 0.7,
        cy - wy * span * 0.7,
        Lz * 0.5
    )

    head = (
        cx + wx * span * 0.35,
        cy + wy * span * 0.35,
        Lz * 0.5
    )

    traces.append(
        go.Scatter3d(
            x=[tail[0], head[0]],
            y=[tail[1], head[1]],
            z=[tail[2], head[2]],
            mode="lines",
            line=dict(
                color="navy",
                width=8
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    traces.append(
        go.Scatter3d(
            x=[head[0]],
            y=[head[1]],
            z=[head[2]],
            mode="markers",
            marker=dict(
                size=10,
                color="navy",
                symbol="diamond"
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # --------------------------------------------------------
    # Compass
    # --------------------------------------------------------

    for label, x, y in [
        ("N", Lx / 2, Ly + 0.7),
        ("S", Lx / 2, -0.7),
        ("E", Lx + 0.7, Ly / 2),
        ("W", -0.7, Ly / 2),
    ]:

        traces.append(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[0.05],
                mode="text",
                text=[label],
                textfont=dict(
                    color="dimgrey",
                    size=13
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = go.Figure(
        data=traces
    )

    fig.update_layout(

        scene=dict(

            xaxis=dict(
                title="x (m)",
                backgroundcolor="#f7f6f2"
            ),

            yaxis=dict(
                title="y (m)",
                backgroundcolor="#f7f6f2"
            ),

            zaxis=dict(
                title="z (m)",
                range=[0, Lz + 0.5],
                backgroundcolor="#f7f6f2"
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=Lx,
                y=Ly,
                z=Lz
            ),

            camera=dict(
                eye=dict(
                    x=1.8,
                    y=-1.8,
                    z=1.3
                )
            ),
        ),

        height=560,

        margin=dict(
            l=0,
            r=0,
            t=30,
            b=0
        ),

        title="3D Room — drag to rotate",
    )

    return fig


# ============================================================
# 3D Surface
# ============================================================

def make_surface_3d(
    field,
    Lx,
    Ly,
    title,
    colorscale,
    cmin=None,
    cmax=None
):

    x = np.linspace(
        0,
        Lx,
        field.shape[0]
    )

    y = np.linspace(
        0,
        Ly,
        field.shape[1]
    )

    Z = field.T

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=Z,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        project=dict(z=True),
                        width=2
                    )
                ),
                colorbar=dict(
                    title="",
                    len=0.6,
                    thickness=15
                ),
            )
        ]
    )

    fig.update_layout(

        scene=dict(

            xaxis=dict(
                title="x (m)",
                backgroundcolor="#f7f6f2"
            ),

            yaxis=dict(
                title="y (m)",
                backgroundcolor="#f7f6f2"
            ),

            zaxis=dict(
                title="",
                backgroundcolor="#f7f6f2"
            ),

            aspectmode="manual",

            aspectratio=dict(
                x=1.2,
                y=1.0,
                z=0.6
            ),

            camera=dict(
                eye=dict(
                    x=1.8,
                    y=-1.8,
                    z=1.0
                )
            ),
        ),

        height=500,

        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        ),

        title=title,
    )

    return fig


# ============================================================
# Streamlit page
# ============================================================

st.set_page_config(
    page_title="Kerala Climate Design Tool",
    layout="wide"
)

st.title(
    "Kerala Climate Design Tool — Prototype Demo"
)

st.caption(
    "Predicts indoor temperature and humidity "
    "from design parameters. "
    "Trained on 762 physics simulations."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Room")

    Lx = st.slider(
        "Length (m)",
        2.7,
        6.0,
        4.4,
        0.1
    )

    Ly = st.slider(
        "Width (m)",
        3.0,
        5.0,
        4.0,
        0.1
    )

    Lz = st.slider(
        "Height (m)",
        2.8,
        3.3,
        3.2,
        0.1
    )

    # ========================================================
    # Windows
    # ========================================================

    st.header("Windows")

    st.caption(
        "Each wall has an independent window. "
        "Set width to 0 for no window."
    )

    win_h = st.slider(
        "Window height (m)",
        0.9,
        1.6,
        1.2,
        0.1
    )

    win_sill = st.slider(
        "Sill height (m)",
        0.6,
        1.4,
        1.0,
        0.1
    )

    win_w_west = st.slider(
        "West window width",
        0.0,
        2.5,
        1.4,
        0.1
    )

    win_w_east = st.slider(
        "East window width",
        0.0,
        2.5,
        1.4,
        0.1
    )

    win_w_north = st.slider(
        "North window width",
        0.0,
        2.5,
        1.4,
        0.1
    )

    win_w_south = st.slider(
        "South window width",
        0.0,
        2.5,
        1.4,
        0.1
    )

    _n_win = sum(
        1
        for w in [
            win_w_west,
            win_w_east,
            win_w_north,
            win_w_south
        ]
        if w >= 0.05
    )

    if _n_win <= 1:

        st.warning(
            "⚠️ Model has limited training data "
            "for rooms with 0–1 windows. "
            "Predictions may be unreliable."
        )

    elif _n_win == 2:

        st.info(
            "ℹ️ Two-window rooms have moderate "
            "training coverage."
        )

    st.caption(
        "Position along wall "
        "(0 = corner, 0.5 = center)"
    )

    win_pos_w = st.slider(
        "West window position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    win_pos_e = st.slider(
        "East window position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    win_pos_n = st.slider(
        "North window position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    win_pos_s = st.slider(
        "South window position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    # ========================================================
    # Door
    # ========================================================

    st.header("Door")

    door_wall = st.selectbox(
        "Door wall",
        [
            "west",
            "east",
            "south",
            "north"
        ],
        index=2
    )

    door_pos = st.slider(
        "Door position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    # ========================================================
    # Air hole
    # ========================================================

    st.header("Air hole")

    hole_wall = st.selectbox(
        "Air hole wall",
        [
            "west",
            "east",
            "south",
            "north"
        ],
        index=0
    )

    hole_pos = st.slider(
        "Air hole position",
        0.05,
        0.95,
        0.5,
        0.05
    )

    hole_w = st.slider(
        "Air hole width (m) — 0 = no hole",
        0.0,
        1.5,
        0.5,
        0.1
    )

    hole_h = st.slider(
        "Air hole height (m)",
        0.0,
        1.5,
        0.4,
        0.1
    )

    # ========================================================
    # Environment
    # ========================================================

    st.header("Environment")

    V_wind = st.slider(
        "Wind speed (m/s)",
        0.3,
        12.0,
        3.0,
        0.1
    )

    wind_dir = st.slider(
        "Wind direction (°)",
        0,
        360,
        270,
        5
    )

    st.caption(
        f"0=N  90=E  180=S  270=W "
        f"(currently {wind_dir_to_name(wind_dir)})"
    )

    T_out = st.slider(
        "Outdoor temp (°C)",
        24.0,
        38.0,
        30.0,
        0.5
    )

    RH_out = st.slider(
        "Outdoor RH (%)",
        50.0,
        95.0,
        75.0,
        1.0
    )

    I_solar = st.slider(
        "Solar (W/m²)",
        100.0,
        1000.0,
        600.0,
        25.0
    )

    # ========================================================
    # Occupancy
    # ========================================================

    st.header("Occupancy")

    Q_people = st.slider(
        "People heat (W)",
        0.0,
        200.0,
        70.0,
        10.0
    )

    Q_equipment = st.slider(
        "Equipment heat (W)",
        0.0,
        300.0,
        50.0,
        10.0
    )

    # ========================================================
    # Fan
    # ========================================================

    st.header("Fan")

    fan = st.selectbox(
        "Fan level",
        [
            "off",
            "low",
            "medium",
            "high"
        ],
        index=2
    )

    # ========================================================
    # Opening states
    # ========================================================

    st.header("Which openings are open?")

    win_open_west = st.checkbox(
        "West window open",
        value=True
    )

    win_open_east = st.checkbox(
        "East window open",
        value=True
    )

    win_open_north = st.checkbox(
        "North window open",
        value=True
    )

    win_open_south = st.checkbox(
        "South window open",
        value=True
    )

    door_open = st.checkbox(
        "Door open",
        value=True
    )

    hole_open = st.checkbox(
        "Air hole open",
        value=True
    )


# ============================================================
# Parameter encoding
# ============================================================

wall_map = {
    "west": 0,
    "east": 1,
    "south": 2,
    "north": 3
}

fan_map = {
    "off": 0,
    "low": 1,
    "medium": 2,
    "high": 3
}


# ============================================================
# 33-parameter vector
# ============================================================

params = np.array([

    # 1–3
    Lx,
    Ly,
    Lz,

    # 4–5
    win_h,
    win_sill,

    # 6–9
    win_w_west,
    win_w_east,
    win_w_north,
    win_w_south,

    # 10–13
    win_pos_w,
    win_pos_e,
    win_pos_n,
    win_pos_s,

    # 14–15
    wall_map[door_wall],
    door_pos,

    # 16–19
    wall_map[hole_wall],
    hole_pos,
    hole_w,
    hole_h,

    # 20–21
    V_wind,
    wind_dir,

    # 22–24
    T_out,
    RH_out,
    I_solar,

    # 25–26
    Q_people,
    Q_equipment,

    # 27
    fan_map[fan],

    # 28–31
    float(win_open_west),
    float(win_open_east),
    float(win_open_north),
    float(win_open_south),

    # 32–33
    float(door_open),
    float(hole_open),

], dtype=np.float32)


# ============================================================
# Safety check
# ============================================================

if len(params) != 33:

    st.error(
        f"Parameter error: application generated "
        f"{len(params)} parameters instead of 33."
    )

    st.stop()


# ============================================================
# Prediction
# ============================================================

T_field, q_field = predict(params)

RH_field = rh_from_q_T(
    q_field,
    T_field
)

HI_field = heat_index(
    T_field,
    RH_field
)


# ============================================================
# Statistics
# ============================================================

mean_T = float(
    T_field.mean()
)

mean_RH = float(
    RH_field.mean()
)

mean_HI = float(
    HI_field.mean()
)

max_T = float(
    T_field.max()
)


# ============================================================
# Extrapolation warnings
# ============================================================

extrap = []

if V_wind > 9:

    extrap.append(
        f"wind {V_wind:.1f} m/s"
    )

if RH_out > 92:

    extrap.append(
        f"RH {RH_out:.0f}%"
    )

if (
    win_w_west > 2.2
    or win_w_east > 2.2
    or win_w_north > 2.2
    or win_w_south > 2.2
):

    extrap.append(
        "large window"
    )


if extrap:

    st.warning(
        "⚠️ Outside training range: "
        + ", ".join(extrap)
    )


# ============================================================
# Room & openings
# ============================================================

st.subheader(
    "Room & Openings"
)

col1, col2 = st.columns(
    [3, 2]
)


# ============================================================
# 3D room
# ============================================================

with col1:

    fig = make_room_3d(

        Lx,
        Ly,
        Lz,

        win_w_west,
        win_w_east,
        win_w_north,
        win_w_south,

        win_h,
        win_sill,

        win_pos_w,
        win_pos_e,
        win_pos_n,
        win_pos_s,

        win_open_west,
        win_open_east,
        win_open_north,
        win_open_south,

        door_wall,
        door_pos,
        door_open,

        hole_wall,
        hole_pos,
        hole_w,
        hole_h,
        hole_open,

        wind_dir,
        V_wind,
        fan,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 2D room plan + metrics
# ============================================================

with col2:

    fig2, ax = plt.subplots(
        figsize=(6, 6)
    )

    draw_room_plan(

        ax,
        Lx,
        Ly,

        win_w_west,
        win_w_east,
        win_w_north,
        win_w_south,

        win_pos_w,
        win_pos_e,
        win_pos_n,
        win_pos_s,

        win_open_west,
        win_open_east,
        win_open_north,
        win_open_south,

        door_wall,
        door_pos,
        door_open,

        hole_wall,
        hole_pos,
        hole_w,
        hole_open,

        wind_dir,
        V_wind,
        fan,
    )

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close()


    # Metrics

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean T",
        f"{mean_T:.1f} °C"
    )

    c2.metric(
        "Peak T",
        f"{max_T:.1f} °C"
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Mean RH",
        f"{mean_RH:.0f} %"
    )

    c4.metric(
        "Heat idx",
        f"{mean_HI:.1f} °C"
    )


    # Comfort indication

    if mean_HI < 30:

        st.success(
            "Comfortable"
        )

    elif mean_HI < 35:

        st.warning(
            "Warm — fan recommended"
        )

    else:

        st.error(
            "AC recommended"
        )


    # Wind information

    wx, wy = wind_vector(
        wind_dir
    )

    st.markdown(
        "**Wind**"
    )

    wc1, wc2, wc3 = st.columns(3)

    wc1.metric(
        "Speed",
        f"{V_wind:.1f} m/s"
    )

    wc2.metric(
        "From",
        wind_dir_to_name(wind_dir)
    )

    wc3.metric(
        "u, v",
        f"{wx * V_wind:+.1f}, "
        f"{wy * V_wind:+.1f}"
    )


# ============================================================
# 3D surface fields
# ============================================================

st.subheader(
    "3D Surface Fields — Drag to Rotate"
)

st.info(
    "These surfaces show the predicted spatial field "
    "inside the room. The vertical axis represents "
    "the predicted value at each grid location. "
    "Drag the plots to rotate them."
)


col1, col2, col3 = st.columns(3)


# Temperature

with col1:

    st.plotly_chart(

        make_surface_3d(

            T_field,
            Lx,
            Ly,

            f"Temperature (°C) — mean {mean_T:.1f}",

            "Hot"
        ),

        use_container_width=True
    )


# Relative humidity

with col2:

    st.plotly_chart(

        make_surface_3d(

            RH_field,
            Lx,
            Ly,

            f"Relative humidity (%) — mean {mean_RH:.0f}",

            "Blues",

            cmin=30,
            cmax=100
        ),

        use_container_width=True
    )


# Heat index

with col3:

    st.plotly_chart(

        make_surface_3d(

            HI_field,
            Lx,
            Ly,

            f"Heat index (°C) — mean {mean_HI:.1f}",

            "YlOrRd"
        ),

        use_container_width=True
    )


st.markdown("---")

st.subheader(
    "How to Read the 3D Surface Fields"
)

st.markdown(
    """
### Temperature surface

The first surface represents the predicted indoor
temperature at the model's spatial grid locations.

- X axis = room length.
- Y axis = room width.
- Surface height = predicted temperature.
- Color = temperature intensity.
- Higher regions indicate warmer predicted areas.

### Relative-humidity surface

The second surface represents predicted relative humidity.

- Blue regions represent lower/higher humidity according
  to the color scale.
- The color scale is approximately 30–100%.
- The value shown when hovering over a point is the predicted
  RH at that location.

### Heat-index surface

The third surface combines temperature and relative humidity.

A region can therefore have a relatively moderate
temperature but a higher heat index when humidity is high.

### Changing the design

Try changing one parameter at a time:

1. Open/close the windows.
2. Move the windows.
3. Change the wind direction.
4. Open a air holes.
5. Change the fan level.
6. Compare the resulting temperature and heat-index surfaces.

The room visualization shows all selected openings,
while the neural-network prediction uses the parameter
schema contained in the trained checkpoint.
"""
)

# ============================================================
# Model information
# ============================================================

st.markdown("---")

with st.expander(
    "Model information"
):

    st.write(
        "Neural-network surrogate model"
    )

    st.write(
        "Input parameters: 33"
   
    )

    st.write(
        "Output: temperature + humidity fields"
    )

    st.write(
        "Grid: 14 × 10"
    )

    st.write(
        "Temperature MAE: 0.94 °C"
    )

    st.write(
        "Humidity MAE: 1.13 g/kg"
    )

    st.warning(
        "Prototype only. Predictions should be "
        "validated against the underlying physics "
        "model before engineering or construction decisions."
    )


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "Kerala Climate Design Tool — Prototype. "
    "Not for construction decisions yet."
)
