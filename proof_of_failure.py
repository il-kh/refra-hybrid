#!/usr/bin/env python3
"""
Proof-of-Failure diagnostic plots for the Phnom Penh sheet pile survey.

Generates two publication-ready figures demonstrating why the seismic
refraction survey could not detect the rock/boulder layer at 7–19 m depth.

Plot 1 — Redpath (1973) 3-Layer Crossover Diagram
Plot 2 — Waveform Decay Waterfall (composite shot gather)

Usage:
    python proof_of_failure.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Site model parameters
# ---------------------------------------------------------------------------
V1, V2, V3 = 350.0, 870.0, 2000.0   # m/s
H1 = 3.0                              # Layer 1 thickness at array (m)
SPREAD = 23.0                          # Geophone spread length (m)
MAX_OFFSET = 53.0                      # Max source–receiver offset (m)
ROCK_DEPTHS_GEOTECH = (7, 10, 13, 16, 19)  # m below surface
DT = 0.0005                           # SEG-2 sample interval (s)
NPTS = 1024                            # samples per trace

# Colours
RED = "#D62728"
BLUE = "#1F77B4"
GREEN = "#2CA02C"
ORANGE = "#FF7F0E"
PURPLE = "#9467BD"

ROCK_COLOURS = {
    7:  "#2CA02C",
    10: "#FF7F0E",
    13: "#9467BD",
    16: "#17BECF",
    19: "#BCBD22",
}


# ===================================================================
# SEG-2 reader (minimal, for BTWZG-24 files)
# ===================================================================
def read_seg2(filepath: Path) -> list[dict]:
    """Return list of {data, src, rx} dicts from a SEG-2 file."""
    raw = filepath.read_bytes()
    endian = "<"
    n_traces = struct.unpack(endian + "H", raw[6:8])[0]
    offsets = [
        struct.unpack(endian + "I", raw[32 + i * 4 : 36 + i * 4])[0]
        for i in range(n_traces)
    ]
    traces: list[dict] = []
    for i in range(n_traces):
        t0 = offsets[i]
        td_size = struct.unpack(endian + "H", raw[t0 + 2 : t0 + 4])[0]
        n_samp = struct.unpack(endian + "I", raw[t0 + 8 : t0 + 12])[0]
        dfmt = struct.unpack("B", raw[t0 + 12 : t0 + 13])[0]
        td_str = raw[t0 + 32 : t0 + td_size]
        meta: dict[str, str] = {}
        for p in td_str.split(b"\x00"):
            s = p.decode("ascii", errors="replace").strip()
            if " " in s:
                k, v = s.split(" ", 1)
                meta[k] = v.strip()
        ds = t0 + td_size
        if dfmt == 2:
            samples = np.frombuffer(
                raw[ds : ds + n_samp * 4], dtype=endian + "i4"
            ).astype(float)
        elif dfmt == 4:
            samples = np.frombuffer(
                raw[ds : ds + n_samp * 4], dtype=endian + "f"
            ).copy()
        else:
            samples = np.zeros(n_samp)
        if len(samples) > NPTS:
            samples = samples[:NPTS]
        elif len(samples) < NPTS:
            samples = np.pad(samples, (0, NPTS - len(samples)))
        traces.append(
            {
                "data": samples,
                "src": float(meta.get("SOURCE_LOCATION", "0")),
                "rx": float(meta.get("RECEIVER_LOCATION", str(i))),
            }
        )
    return traces


# ===================================================================
# Plot 1 — Redpath 3-Layer Crossover Diagram
# ===================================================================
def make_crossover_plot() -> go.Figure:
    """T-X travel-time diagram showing V₃ never becomes first arrival
    within the 23 m survey spread for rock depths ≥ 10 m."""

    ic12 = np.arcsin(V1 / V2)
    ic13 = np.arcsin(V1 / V3)
    ic23 = np.arcsin(V2 / V3)

    def t_direct(x):
        return x / V1

    def t_refr2(x):
        return x / V2 + 2 * H1 * np.cos(ic12) / V1

    def t_refr3(x, h2):
        return x / V3 + 2 * H1 * np.cos(ic13) / V1 + 2 * h2 * np.cos(ic23) / V2

    def crossover_v2v3(h2):
        ti2 = 2 * H1 * np.cos(ic12) / V1
        ti3 = 2 * H1 * np.cos(ic13) / V1 + 2 * h2 * np.cos(ic23) / V2
        return (ti3 - ti2) / (1 / V2 - 1 / V3)

    # ---- Left subplot: T-X diagram for selected rock depths --------
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.10,
        subplot_titles=[
            "<b>Travel-Time Curves (T-X Diagram)</b>",
            "<b>V₃ Crossover Distance vs Rock Depth</b>",
        ],
    )

    x = np.linspace(0, 70, 500)

    # Direct wave V₁
    fig.add_trace(
        go.Scatter(
            x=x, y=t_direct(x) * 1000,
            mode="lines",
            line=dict(color="black", width=2.5),
            name=f"Direct wave V₁ = {V1:.0f} m/s",
        ),
        row=1, col=1,
    )
    # V₂ head wave
    fig.add_trace(
        go.Scatter(
            x=x, y=t_refr2(x) * 1000,
            mode="lines",
            line=dict(color=BLUE, width=2.5),
            name=f"V₂ refraction = {V2:.0f} m/s  (h₁ = {H1:.0f} m)",
        ),
        row=1, col=1,
    )

    # V₃ head wave for each rock depth
    for d_rock in ROCK_DEPTHS_GEOTECH:
        h2 = d_rock - H1
        xc = crossover_v2v3(h2)
        colour = ROCK_COLOURS[d_rock]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=t_refr3(x, h2) * 1000,
                mode="lines",
                line=dict(color=colour, width=2, dash="dot"),
                name=f"V₃ = {V3:.0f}  rock @ {d_rock} m  (x꜀ = {xc:.1f} m)",
            ),
            row=1, col=1,
        )
        # Crossover marker on V₂ line
        if xc < 70:
            fig.add_trace(
                go.Scatter(
                    x=[xc],
                    y=[t_refr2(xc) * 1000],
                    mode="markers+text",
                    marker=dict(size=10, color=colour, symbol="x-thin",
                                line=dict(width=2, color=colour)),
                    text=[f" {xc:.0f} m"],
                    textposition="top right",
                    textfont=dict(size=11, color=colour),
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # 23m spread band (RED)
    fig.add_vrect(
        x0=0, x1=SPREAD, fillcolor=RED, opacity=0.10,
        line=dict(width=2, color=RED), row=1, col=1,
    )
    fig.add_annotation(
        x=SPREAD / 2, y=5,
        text=f"<b>Survey Spread<br>{SPREAD:.0f} m</b>",
        showarrow=False,
        font=dict(size=13, color=RED),
        row=1, col=1,
    )
    # Max offset dashed line
    fig.add_vline(
        x=MAX_OFFSET, line=dict(color="grey", width=1.5, dash="dash"),
        annotation_text=f"Max offset {MAX_OFFSET:.0f} m",
        annotation_position="top left",
        annotation_font=dict(size=10, color="grey"),
        row=1, col=1,
    )

    # First-arrival composite (bold) – for rock@12 m (median case)
    h2_med = 12 - H1
    first_arr = np.minimum(np.minimum(t_direct(x), t_refr2(x)),
                           t_refr3(x, h2_med))
    fig.add_trace(
        go.Scatter(
            x=x, y=first_arr * 1000,
            mode="lines",
            line=dict(color="black", width=1, dash="longdash"),
            name="First arrival (rock @ 12 m)",
            opacity=0.4,
        ),
        row=1, col=1,
    )

    fig.update_xaxes(title_text="Source–Receiver Offset (m)",
                     range=[0, 65], row=1, col=1)
    fig.update_yaxes(title_text="Travel Time (ms)",
                     range=[0, 80], row=1, col=1)

    # ---- Right subplot: crossover distance vs rock depth -----------
    d_range = np.linspace(5, 22, 200)
    xc_curve = np.array([crossover_v2v3(d - H1) for d in d_range])

    fig.add_trace(
        go.Scatter(
            x=d_range, y=xc_curve,
            mode="lines",
            line=dict(color=BLUE, width=3),
            name="V₂→V₃ crossover distance",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 23m spread threshold (RED)
    fig.add_hline(
        y=SPREAD, line=dict(color=RED, width=3),
        row=1, col=2,
    )
    fig.add_annotation(
        x=18, y=SPREAD + 1.5,
        text=f"<b>Spread = {SPREAD:.0f} m</b>",
        showarrow=False,
        font=dict(size=13, color=RED),
        row=1, col=2,
    )

    # 53m max offset
    fig.add_hline(
        y=MAX_OFFSET, line=dict(color="grey", width=1.5, dash="dash"),
        row=1, col=2,
    )
    fig.add_annotation(
        x=18, y=MAX_OFFSET + 1.5,
        text=f"Max offset = {MAX_OFFSET:.0f} m",
        showarrow=False,
        font=dict(size=10, color="grey"),
        row=1, col=2,
    )

    # Geotech rock-depth band
    fig.add_vrect(
        x0=7, x1=19, fillcolor=ORANGE, opacity=0.12,
        line=dict(width=0),
        row=1, col=2,
    )
    fig.add_annotation(
        x=13, y=58,
        text="<b>Geotech rock range<br>7–19 m</b>",
        showarrow=False,
        font=dict(size=11, color=ORANGE),
        row=1, col=2,
    )

    # Shade the "unreachable" zone (xc > 23 m AND rock in 10-19 m)
    safe_mask = xc_curve > SPREAD
    if safe_mask.any():
        first_safe = d_range[safe_mask][0]
        fig.add_annotation(
            x=first_safe, y=SPREAD,
            text=f"  V₃ unreachable<br>  within spread<br>  for rock ≥ {first_safe:.0f} m",
            showarrow=True,
            arrowhead=2,
            arrowcolor=RED,
            ax=40, ay=-50,
            font=dict(size=11, color=RED),
            row=1, col=2,
        )

    # Mark specific geotech depths
    for d_rock in ROCK_DEPTHS_GEOTECH:
        h2 = d_rock - H1
        xc = crossover_v2v3(h2)
        colour = ROCK_COLOURS[d_rock]
        fig.add_trace(
            go.Scatter(
                x=[d_rock], y=[xc],
                mode="markers",
                marker=dict(size=10, color=colour, line=dict(width=1, color="black")),
                showlegend=False,
                hovertext=f"Rock @ {d_rock} m → x₃ = {xc:.1f} m",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Rock Depth (m)", range=[4, 22], row=1, col=2)
    fig.update_yaxes(title_text="Crossover Distance (m)", range=[0, 60], row=1, col=2)

    # ---- Layout ----------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(
                "<b>Plot 1 — Proof of Failure: Redpath (1973) 3-Layer Crossover Analysis</b><br>"
                "<sup>V₁ = 350 m/s (alluvium)  ·  V₂ = 870 m/s (stiff clay)  ·  "
                "V₃ = 2 000 m/s (rock/boulders)  ·  h₁ = 3 m  ·  "
                "Spread = 23 m  ·  Max offset = 53 m</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        width=1400,
        height=600,
        template="plotly_white",
        legend=dict(
            orientation="h", y=-0.20, x=0.5, xanchor="center",
            font=dict(size=10),
        ),
        margin=dict(t=110, b=120),
    )

    return fig


# ===================================================================
# Plot 2 — Waveform Decay Waterfall
# ===================================================================
def make_waterfall_plot() -> go.Figure:
    """Composite shot gather (near + far offset) showing energy
    attenuation as a function of source–receiver offset."""

    raw_dir = Path(__file__).parent / "raw_waveforms"

    # Find a representative transect (T450, shots 0 and 1)
    shot_near = None  # source at -0.5 m, offsets 0.5–23.5 m
    shot_far = None   # source at -30 m,  offsets 30–53 m
    for sg in sorted(raw_dir.glob("R0450-*.sg2")):
        parts = sg.stem.split("-")
        try:
            idx = int(parts[1])
        except (IndexError, ValueError):
            continue
        if idx == 1:
            shot_near = sg
        elif idx == 0:
            shot_far = sg

    if shot_near is None or shot_far is None:
        # Fallback: try any transect
        for prefix in ["R0530", "R0310", "R0690"]:
            candidates = sorted(raw_dir.glob(f"{prefix}-*.sg2"))
            for sg in candidates:
                idx = int(sg.stem.split("-")[1])
                if idx == 1 and shot_near is None:
                    shot_near = sg
                elif idx == 0 and shot_far is None:
                    shot_far = sg

    assert shot_near is not None and shot_far is not None, \
        f"Could not find shot files in {raw_dir}"

    # Read both shots
    tr_near = read_seg2(shot_near)
    tr_far = read_seg2(shot_far)

    # Build composite gather sorted by offset
    all_offsets: list[float] = []
    all_traces: list[np.ndarray] = []

    for tr in tr_near:
        off = abs(tr["rx"] - tr["src"])
        if off < 0.1:
            continue
        all_offsets.append(off)
        all_traces.append(tr["data"])

    for tr in tr_far:
        off = abs(tr["rx"] - tr["src"])
        if off < 0.1:
            continue
        all_offsets.append(off)
        all_traces.append(tr["data"])

    order = np.argsort(all_offsets)
    offsets = np.array(all_offsets)[order]
    traces = [all_traces[i] for i in order]

    # Global amplitude reference: max amplitude of nearest trace
    amp_ref = max(np.max(np.abs(tr)) for tr in traces[:3])
    t_axis = np.arange(NPTS) * DT * 1000  # ms

    fig = go.Figure()

    # ---- Waveform traces (true-amplitude, common scale) ----
    # We scale every trace by the SAME factor so relative amplitudes
    # are preserved — the visual shows decay directly.
    gain = 0.9 * (offsets[1] - offsets[0]) if len(offsets) > 1 else 1.0  # trace spacing
    gain_factor = gain / amp_ref

    # Also compute peak amplitude per trace for the envelope
    peak_amps = []

    for i, (off, tr) in enumerate(zip(offsets, traces)):
        scaled = tr * gain_factor
        peak_amps.append(np.max(np.abs(tr)))

        colour = "black" if off <= SPREAD else "#555555"
        opacity = 1.0 if off <= SPREAD else 0.65
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=scaled + off,
                mode="lines",
                line=dict(color=colour, width=0.6),
                opacity=opacity,
                showlegend=False,
                hovertext=f"Offset {off:.1f} m",
            )
        )

    peak_amps = np.array(peak_amps)

    # ---- Amplitude envelope overlay (right margin effect) ----------
    # Normalize peak amps to 0-1
    pa_norm = peak_amps / peak_amps.max()
    # Place envelope on the right side of the time window
    t_envelope = 280  # ms position
    env_scale = 50     # ms width for full amplitude
    env_x = t_envelope + pa_norm * env_scale

    fig.add_trace(
        go.Scatter(
            x=env_x, y=offsets,
            mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=4, color=RED),
            name="Peak amplitude envelope",
        )
    )

    # ---- Noise floor reference line ----
    # Estimate noise floor from the pre-trigger or late-record on far traces
    noise_levels = []
    for tr in traces[-5:]:
        late = tr[int(0.8 * NPTS):]  # last 20% of record
        noise_levels.append(np.std(late))
    noise_floor = np.mean(noise_levels)
    noise_norm = noise_floor / peak_amps.max()
    noise_x = t_envelope + noise_norm * env_scale

    fig.add_vline(
        x=noise_x,
        line=dict(color=ORANGE, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=noise_x, y=offsets.min() - 1,
        text="<b>Noise floor</b>",
        showarrow=False,
        font=dict(size=11, color=ORANGE),
        yanchor="bottom",
    )

    # ---- Decorations -----------------------------------------------
    # Red band: spread
    fig.add_hrect(
        y0=0, y1=SPREAD,
        fillcolor=GREEN, opacity=0.06,
        line=dict(width=0),
    )
    fig.add_annotation(
        x=10, y=SPREAD / 2,
        text=f"<b>Within 23 m spread</b>",
        showarrow=False,
        font=dict(size=11, color=GREEN),
    )

    # Gap annotation (no receiver coverage 24–30 m)
    fig.add_hrect(
        y0=SPREAD + 0.5, y1=29,
        fillcolor="grey", opacity=0.08,
        line=dict(width=0),
    )
    fig.add_annotation(
        x=120, y=26.5,
        text="<i>no receiver<br>coverage</i>",
        showarrow=False,
        font=dict(size=9, color="grey"),
    )

    # Far offset zone
    fig.add_annotation(
        x=10, y=42,
        text="<b>Far-offset shot<br>(30–53 m)</b>",
        showarrow=False,
        font=dict(size=11, color="#555555"),
    )

    # SNR annotation near 50 m
    snr_50 = peak_amps[offsets >= 48][0] / noise_floor if (offsets >= 48).any() else 1.0
    if (offsets >= 48).any():
        off_50 = offsets[offsets >= 48][0]
        fig.add_annotation(
            x=200, y=off_50,
            text=f"<b>SNR ≈ {snr_50:.1f}</b>",
            showarrow=True, arrowhead=2, arrowcolor=RED,
            ax=-60, ay=0,
            font=dict(size=12, color=RED),
        )

    # Main callout label
    fig.add_annotation(
        x=0.97, y=0.03,
        xref="paper", yref="paper",
        text=(
            "<b>Energy Attenuation —<br>"
            "Target Reflector Not Reached</b>"
        ),
        showarrow=False,
        font=dict(size=15, color=RED),
        align="right",
        xanchor="right", yanchor="bottom",
        bordercolor=RED,
        borderwidth=2,
        borderpad=8,
        bgcolor="rgba(255,255,255,0.85)",
    )

    # Moveout guide lines
    for v_guide, clr, dash, lbl in [
        (V1, "blue", "dot", f"V₁ = {V1:.0f}"),
        (V2, BLUE, "dashdot", f"V₂ = {V2:.0f}"),
    ]:
        t_guide = offsets / v_guide * 1000
        fig.add_trace(
            go.Scatter(
                x=t_guide, y=offsets,
                mode="lines",
                line=dict(color=clr, width=1.5, dash=dash),
                name=f"{lbl} m/s moveout",
                opacity=0.6,
            )
        )

    src_n = tr_near[0]["src"]
    src_f = tr_far[0]["src"]
    fig.update_layout(
        title=dict(
            text=(
                "<b>Plot 2 — Proof of Failure: Waveform Amplitude Decay</b><br>"
                f"<sup>Transect T450  ·  Near shot (src = {src_n:.1f} m)  +  "
                f"Far shot (src = {src_f:.1f} m)  ·  24 channels × 1 m  ·  "
                "Red curve = peak amplitude envelope</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Time (ms)",
            range=[0, 350],
        ),
        yaxis=dict(
            title="Source–Receiver Offset (m)",
            range=[offsets.max() + 2, -1],  # inverted so near=top
            autorange=False,
        ),
        width=1100,
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="h", y=-0.12, x=0.5, xanchor="center",
            font=dict(size=10),
        ),
        margin=dict(t=110, b=90),
    )

    return fig


# ===================================================================
# Main
# ===================================================================
def main():
    out = Path(__file__).parent

    # Plot 1
    print("Generating Plot 1 — Crossover diagram …")
    fig1 = make_crossover_plot()
    p1_html = out / "proof_1_crossover.html"
    p1_png = out / "proof_1_crossover.png"
    fig1.write_html(str(p1_html), include_plotlyjs="cdn")
    fig1.write_image(str(p1_png), scale=2)
    print(f"  → {p1_html}")
    print(f"  → {p1_png}")

    # Plot 2
    print("Generating Plot 2 — Waveform decay …")
    fig2 = make_waterfall_plot()
    p2_html = out / "proof_2_waveform_decay.html"
    p2_png = out / "proof_2_waveform_decay.png"
    fig2.write_html(str(p2_html), include_plotlyjs="cdn")
    fig2.write_image(str(p2_png), scale=2)
    print(f"  → {p2_html}")
    print(f"  → {p2_png}")

    print("\nDone. Attach the PNG files to your email.")


if __name__ == "__main__":
    main()
