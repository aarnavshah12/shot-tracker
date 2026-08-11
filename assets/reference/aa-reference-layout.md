# AA demo reference layout (owner-supplied screenshots, 2026-08-11)

Two reference frames from the AA soccer demo (frame 75/155 mid-flight and
frame 155/155 resolved). This is the look M4 must match, translated to hoops.
Owner confirmed: "the UI should have stuff like this."

## Global

- Split screen: video left (~55%), right column of three stacked panels.
- Panels: cream/off-white background, thin dark separators.
- Headers: orange, monospace, ALL CAPS (e.g. `SHOT SPEED`, `GOAL?`).
- Big values: huge near-black monospace numerals (`75 KM/H`, `3.4 M`, `GOAL!!`).
- Small stat lines: gray monospace caps under the big value.
- Frame counter bottom-right on a dark strip: `FRAME 75 / 155`.

## Right column (top to bottom), hoops translation in brackets

1. `SHOT SPEED` — big number holds the shot's release speed (shown in KM/H
   in the reference) for the whole flight. Small lines:
   - `CURRENT SPEED: 19 KM/H` — updates every frame during flight.
   - `BALL PX: 24.0` — ball box size, debug flavor.
2. `BALL DISTANCE TO GOAL LINE` [BALL DISTANCE TO RIM] — big meters value
   counting down during flight (`7.2 M` → `3.4 M`). Small line: component
   offsets `X: -2.20M  Y: -3.41M  Z: 0.14M` [we are 2D: X/Y offsets from rim
   center via scale].
3. `GOAL?` [MAKE?] — during flight: big gray `NO` with small status line
   `CROSS DETECTOR: ACTIVE`. On resolve: big green `GOAL!!` [`MAKE!!`; red
   `MISS`] with small line `CROSS AT X: -2.33M  Z: 1.39M` [crossing offset
   from rim center].

## Bottom strip (overlays the video)

- Bottom-left inset: `Pose at Contact` [Pose at Release] — magenta/pink
  skeleton snapshot on cream, frozen at the contact/release frame.
- Bottom-center table: `Joint Angles | Left Side | Right Side` with rows
  `shoulder-to-hip` (8° / -8°), `hip-to-knee` (36° / 32°), `knee-to-ankle`
  (37° / 40°). Signed degrees, per-side columns.

## Video overlays

- Detection boxes with colored labels: green box + green label for the goal
  [rim], blue box + blue label for the ball.
- Shooter: pink/magenta COCO skeleton drawn on the player, plus a
  name/initials tag (`AA`, white on black) with a red triangle arrow above.
- Thick orange trajectory trail behind the ball during flight.
- Red target reticle drawn at the predicted/actual crossing point.

## Notes for M4

- Reference displays speed in KM/H; shots.jsonl stores m/s (plan 7). Decide
  display units with the owner at sign-off; converting at draw time is free.
- The plan's Panel 2 small line ("ball X/Y offset from rim center") matches
  the reference's component read-out; we have no Z in 2D.
- `CROSS DETECTOR: ACTIVE` maps to our DESCENDING phase (crossing check armed).
