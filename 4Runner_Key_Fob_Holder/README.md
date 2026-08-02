# 4Runner Key Fob Holder

Drop-in insert for the coin pocket on the center console of a 2022 Toyota
4Runner (5th gen). Holds the smart key fob (HYQ14FBA/FBB) standing upright so
it's easy to grab — the fob sinks ~40mm into the slot and sticks out ~32mm.

- `4runner_key_fob_holder.scad` — parametric OpenSCAD source
- `4runner_key_fob_holder.stl` — rendered with the default dimensions below

## Before printing — verify two measurements

The coin pocket defaults are estimates (64 x 36 mm opening). Measure your
pocket's opening with calipers and update `pocket_length` / `pocket_width`
in the `.scad` if yours differs, then re-render:

```bash
openscad -o 4runner_key_fob_holder.stl 4runner_key_fob_holder.scad
```

Fit notes:

- The insert body is undersized by `fit_clearance` (0.4mm total) and relies on
  three vertical crush ribs per long face for a snug, rattle-free press fit.
- The top collar overhangs the pocket rim by 2.5mm, so slightly-short walls
  or a rough pocket edge stay hidden.
- The fob slot is 38.5 x 14.5mm; if your fob has a thick case/cover on it,
  bump `slot_width`.

## Print settings

- PETG, ABS, or ASA — **not PLA** (cars get hot enough to warp PLA)
- 0.2mm layers, 3 walls, ~15% infill, no supports
- Print slot-up, as oriented in the STL
