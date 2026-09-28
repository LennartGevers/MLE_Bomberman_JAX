"""Draw game states as SVG and animate games in a notebook."""

from __future__ import annotations

import uuid
from collections.abc import Iterable

import numpy as np
import svgwrite

from bomberman.game.environment import FIELD, EnvironmentState
from bomberman.game.game import GameState
from bomberman.rollout import RolloutOutput

BG = "#191b1f"          # canvas
EMPTY = "#101114"       # empty tile (recessed)
WALL = "#2b2e33"        # wall tile (raised)
GRID = "#303338"        # hairline tile separator
LINE = "#7d838c"        # structural detail (crate brace, fuse)
BRIGHT = "#d6dae0"      # bright detail (agent ring, explosion, coin, bomb rim)
DIM = "#565b62"         # muted detail (dead agent)
SOFT_FILL = "#23262b"   # bomb disc
TEXT = "#aab0b8"


def render(state: GameState, *, cell: int = 32) -> svgwrite.Drawing:
    """Draw ``state`` as an SVG image with ``cell`` pixels per tile."""
    env: EnvironmentState = state.board
    field = np.asarray(env.field_map)
    cols, rows = field.shape
    width, height = cols * cell, rows * cell

    dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"))
    dwg.viewbox(0, 0, width, height)
    dwg.add(dwg.rect((0, 0), (width, height), fill=BG))

    for x in range(cols):
        for y in range(rows):
            _tile(dwg, x, y, int(field[x, y]), cell)

    _coins(dwg, env, cell)
    _explosions(dwg, env, cell)
    _bombs(dwg, env, cell)
    _agents(dwg, env, cell, np.asarray(state.alive))

    dwg.add(dwg.rect((0, 0), (width, height), fill="none", stroke=GRID, stroke_width=1))
    return dwg


def _center(x: int, y: int, cell: int) -> tuple[float, float]:
    return (x + 0.5) * cell, (y + 0.5) * cell


def _tile(dwg: svgwrite.Drawing, x: int, y: int, kind: int, cell: int) -> None:
    ox, oy = x * cell, y * cell
    fill = WALL if kind == FIELD.WALL else EMPTY
    dwg.add(dwg.rect((ox, oy), (cell, cell), fill=fill, stroke=GRID, stroke_width=0.5))

    if kind == FIELD.CRATE:
        m = cell * 0.17
        dwg.add(dwg.rect((ox + m, oy + m), (cell - 2 * m, cell - 2 * m),
                         fill="none", stroke=LINE, stroke_width=0.9))
        dwg.add(dwg.line((ox + m, oy + m), (ox + cell - m, oy + cell - m),
                         stroke=LINE, stroke_width=0.7))
        dwg.add(dwg.line((ox + cell - m, oy + m), (ox + m, oy + cell - m),
                         stroke=LINE, stroke_width=0.7))


def _coins(dwg: svgwrite.Drawing, env: EnvironmentState, cell: int) -> None:
    positions = np.asarray(env.coins.pos)
    collectable = np.asarray(env.coins.collectable(env.field_map))
    for (x, y), shown in zip(positions, collectable):
        if not shown:
            continue
        cx, cy = _center(int(x), int(y), cell)
        dwg.add(dwg.circle((cx, cy), r=cell * 0.12, fill="none",
                           stroke=BRIGHT, stroke_width=1))


def _star(cx: float, cy: float, r_out: float, r_in: float, spikes: int = 8) -> list[tuple[float, float]]:
    points = []
    for i in range(spikes * 2):
        angle = np.pi * i / spikes
        r = r_out if i % 2 == 0 else r_in
        points.append((cx + r * np.sin(angle), cy - r * np.cos(angle)))
    return points


def _explosions(dwg: svgwrite.Drawing, env: EnvironmentState, cell: int) -> None:
    active = np.asarray(env.explosions.active)
    blast = np.asarray(env.explosions.blast)
    drawn: set[tuple[int, int]] = set()
    for slot, on in enumerate(active):
        if not on:
            continue
        for x, y in blast[slot]:
            tile = (int(x), int(y))
            if tile in drawn:
                continue
            drawn.add(tile)
            cx, cy = _center(*tile, cell)
            dwg.add(dwg.polygon(_star(cx, cy, cell * 0.44, cell * 0.18),
                                fill="none", stroke=BRIGHT, stroke_width=0.9))


def _bombs(dwg: svgwrite.Drawing, env: EnvironmentState, cell: int) -> None:
    positions = np.asarray(env.bombs.pos)
    timers = np.asarray(env.bombs.timer)
    active = np.asarray(env.bombs.active)
    for (x, y), timer, on in zip(positions, timers, active):
        if not on:
            continue
        cx, cy = _center(int(x), int(y), cell)
        dwg.add(dwg.circle((cx, cy), r=cell * 0.23, fill=SOFT_FILL,
                           stroke=BRIGHT, stroke_width=1))
        dwg.add(dwg.line((cx + cell * 0.15, cy - cell * 0.15),
                         (cx + cell * 0.29, cy - cell * 0.31),
                         stroke=LINE, stroke_width=1))
        dwg.add(dwg.text(str(int(timer)), insert=(cx, cy + cell * 0.11),
                         text_anchor="middle", font_size=cell * 0.3,
                         font_family="ui-sans-serif, sans-serif", fill=TEXT))


def _agents(
    dwg: svgwrite.Drawing,
    env: EnvironmentState,
    cell: int,
    alive: np.ndarray,
) -> None:
    r = cell * 0.35
    for i, (x, y) in enumerate(np.asarray(env.agent_positions)):
        x, y = int(x), int(y)
        cx, cy = _center(x, y, cell)
        is_alive = bool(alive[i])
        stroke = BRIGHT if is_alive else DIM

        ring = dwg.circle((cx, cy), r=r, fill="none", stroke=stroke, stroke_width=1.2)
        if not is_alive:
            ring["stroke-dasharray"] = "2,2"
            d = r * 0.7
            dwg.add(dwg.line((cx - d, cy - d), (cx + d, cy + d), stroke=DIM, stroke_width=1))
            dwg.add(dwg.line((cx + d, cy - d), (cx - d, cy + d), stroke=DIM, stroke_width=1))
        dwg.add(ring)

        dwg.add(dwg.text(str(i), insert=(x * cell + cell * 0.15, y * cell + cell * 0.26),
                         text_anchor="middle", font_size=cell * 0.24,
                         font_family="ui-sans-serif, sans-serif",
                         fill=TEXT if is_alive else DIM))


def animate(
    states: Iterable[GameState] | RolloutOutput,
    *,
    env: int = 0,
    cell: int = 22,
    fps: float = 4.0,
):
    """Animate a list of game states or one game of a ``RolloutOutput``, with a play button and a slider."""
    from IPython.display import HTML

    if isinstance(states, RolloutOutput):
        states = states.frames(env)

    frames = [render(s, cell=cell).tostring() for s in states]
    if not frames:
        raise ValueError("no states to animate")

    uid = "bmb" + uuid.uuid4().hex[:8]
    last = len(frames) - 1
    interval = max(1, round(1000.0 / fps))
    divs = "".join(
        f'<div class="{uid}f" style="display:{"block" if i == 0 else "none"}">{svg}</div>'
        for i, svg in enumerate(frames)
    )
    return HTML(f"""
<div id="{uid}" style="display:inline-block">
  {divs}
  <div style="margin-top:6px;font:12px ui-sans-serif,system-ui;color:{TEXT}">
    <button id="{uid}b" style="width:2em">&#9208;</button>
    <input id="{uid}r" type="range" min="0" max="{last}" value="0"
           style="width:{max(180, (last + 1) * 4)}px;vertical-align:middle">
    <span id="{uid}l">0 / {last}</span>
  </div>
</div>
<script>
(function(){{
  var root=document.getElementById("{uid}");
  var f=root.getElementsByClassName("{uid}f");
  var r=document.getElementById("{uid}r"), l=document.getElementById("{uid}l"),
      b=document.getElementById("{uid}b");
  var i=0, playing=true;
  function show(k){{
    f[i].style.display="none";
    i=((k % f.length) + f.length) % f.length;
    f[i].style.display="block"; r.value=i; l.textContent=i+" / {last}";
  }}
  r.addEventListener("input", function(e){{ playing=false; b.innerHTML="&#9654;";
    show(parseInt(e.target.value,10)); }});
  b.addEventListener("click", function(){{ playing=!playing;
    b.innerHTML=playing?"&#9208;":"&#9654;"; }});
  setInterval(function(){{ if(playing) show(i+1); }}, {interval});
}})();
</script>
""")
