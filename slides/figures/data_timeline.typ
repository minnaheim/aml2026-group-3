// Data alignment timeline
// Shows how monthly, quarterly, daily, and speech data are fused to a monthly feature matrix.
// data-timeline(step) reveals rows one at a time:
//   step 1 — time axis + monthly macro
//   step 2 — + quarterly GDP
//   step 3 — + daily financial
//   step 4 — + Fed speeches + rolling-window bracket
//   step 5 — + bottom "monthly feature matrix" badge
#import "@preview/cetz:0.5.0": canvas, draw

#let data-timeline(step: 5) = canvas(length: 1.2cm, {
  import draw: *

  // ── Palette ─────────────────────────────────────
  let c-m  = rgb("#3A6BB5")   // monthly  — blue
  let c-q  = rgb("#C96820")   // quarterly — orange
  let c-d  = rgb("#3E7228")   // daily    — green
  let c-s  = rgb("#AA1818")   // speeches — red
  let c-ax = rgb("#222222")
  let c-ar = rgb("#999999")   // merge arrows

  // ── Layout ──────────────────────────────────────
  let mw  = 2.4   // month width (cm)
  let rh  = 0.80  // row height — tall enough for comfortable single-line labels
  let gap = 0.50  // vertical gap between rows
  let lx  = -1.2  // left label anchor x
  let nm  = 5     // months shown
  let ls  = 0.60em

  // Row y-positions (bottom edge of band)
  let ya = 0.0
  let ym = 0.45                   // monthly macro
  let yq = ym + rh + gap          // quarterly GDP
  let yd = yq + rh + gap          // daily financial
  let ys = yd + rh + gap + 0.10   // speeches

  // ── Time axis ──────────────────────────────────
  line((-0.3, ya), (nm * mw + 0.6, ya),
       stroke: c-ax, mark: (end: "stealth", scale: 0.5))
  for i in range(nm + 1) {
    line((i * mw, ya - 0.07), (i * mw, ya + 0.07), stroke: c-ax)
  }
  let mlabels = ("Jan", "Feb", "Mar", "Apr", "May")
  for i in range(nm) {
    content((i * mw + mw / 2, ya - 0.15),
            text(size: ls)[#mlabels.at(i) 2020], anchor: "north")
  }

  // ── Monthly Macro (step 1+) ─────────────────────
  for i in range(nm) {
    let x0 = i * mw + 0.04
    let x1 = (i + 1) * mw - 0.04
    rect((x0, ym), (x1, ym + rh),
         fill: c-m.lighten(82%), stroke: (paint: c-m, thickness: 0.7pt))
  }
  content((lx, ym + rh / 2),
          text(size: ls)[*Monthly Macro*], anchor: "east")
  content((nm * mw + 0.15, ym + rh / 2),
          text(size: ls * 0.78, fill: c-m.darken(20%))[CPI · UNRATE · PAYEMS · INDPRO],
          anchor: "west")

  // ── Quarterly GDP (step 2+) ─────────────────────
  if step >= 2 {
    let qtrs = ((0, 3, "Q1 2020"), (3, 2, "Q2 2020"))
    for (qi, qn, ql) in qtrs {
      let x0 = qi * mw + 0.04
      let x1 = (qi + qn) * mw - 0.04
      rect((x0, yq), (x1, yq + rh),
           fill: c-q.lighten(82%), stroke: (paint: c-q, thickness: 0.7pt))
      content(((x0 + x1) / 2, yq + rh / 2),
              text(size: ls * 0.82, fill: c-q.darken(25%))[#ql],
              anchor: "center")
      // Dashed monthly dividers — show forward-fill boundaries
      for j in range(1, qn) {
        let xd = (qi + j) * mw
        line((xd, yq + 0.10), (xd, yq + rh - 0.10),
             stroke: (paint: c-q.lighten(10%), dash: "dashed", thickness: 0.5pt))
      }
    }
    content((lx, yq + rh / 2),
            text(size: ls)[*Quarterly GDP*], anchor: "east")
    content((nm * mw + 0.15, yq + rh / 2),
            text(size: ls * 0.78, fill: c-q.darken(20%))[fwd-fill · ÷3/month],
            anchor: "west")
  }

  // ── Daily Financial (step 3+) ───────────────────
  if step >= 3 {
    for i in range(nm) {
      let x0 = i * mw + 0.04
      let x1 = (i + 1) * mw - 0.04
      let w  = x1 - x0
      // ~22 trading-day ticks inside box
      for d in range(22) {
        let dx = x0 + (d + 0.5) * w / 22.0
        line((dx, yd + 0.11), (dx, yd + rh - 0.11),
             stroke: (paint: c-d.lighten(28%), thickness: 0.32pt))
      }
      rect((x0, yd), (x1, yd + rh), stroke: (paint: c-d, thickness: 0.7pt))
      // μ,σ label centred — single line, fits comfortably in rh=0.80
      content(((x0 + x1) / 2, yd + rh / 2),
              text(size: ls * 1.2, fill: c-d.darken(20%))[*$mu, sigma$*],
              anchor: "center")
    }
    content((lx, yd + rh / 2),
            text(size: ls)[*Daily Agg. Financial*], anchor: "east")
    content((nm * mw + 0.15, yd + rh / 2),
            text(size: ls * 0.78, fill: c-d.darken(20%))[GBP · YEN · FFR · T10Y3M],
            anchor: "west")
  }

  // ── FOMC Speeches + rolling-window bracket (step 4+) ──
  if step >= 4 {
    let s-fracs = (0.03, 0.10, 0.20, 0.30, 0.42, 0.50, 0.59, 0.68, 0.76, 0.87, 0.92, 0.97)
    let span = nm * mw

    rect((0.0, ys), (span, ys + rh), fill: c-s.lighten(94%), stroke: none)

    for f in s-fracs {
      let sx = f * span
      line((sx, ys + 0.06), (sx, ys + rh - 0.12),
           stroke: (paint: c-s.lighten(20%), thickness: 0.9pt))
    }
    // "…" hinting window extends further left
    content((-0.25, ys + rh / 2), text(size: 1.0em, fill: c-s)[…], anchor: "center")

    content((lx, ys + rh / 2),
            text(size: ls)[*Fed Speeches*], anchor: "east")
    content((nm * mw + 0.15, ys + rh / 2),
            text(size: ls * 0.78, fill: c-s.darken(10%))[daily but irregular],
            anchor: "west")

    // Rolling-window bracket (example: Apr 2020)
    let bky   = ys + rh + 0.24
    let w-end = 3.0 * mw    // Apr 2020 = start of month index 3
    let tk    = 0.13

    line((0.0, bky), (w-end, bky), stroke: (paint: c-s, thickness: 1pt))
    line((0.0,  bky - tk), (0.0,  bky), stroke: (paint: c-s, thickness: 1pt))
    line((w-end, bky - tk), (w-end, bky), stroke: (paint: c-s, thickness: 1pt))

    line((0.0, bky), (-0.6, bky),
         stroke: (paint: c-s, thickness: 1pt, dash: "dashed"),
         mark: (end: "stealth", scale: 0.45))

    content((w-end / 2, bky + 0.14),
            text(size: ls * 0.88, fill: c-s)[← 12-month rolling window (example: Apr 2020)],
            anchor: "south")
  }

  // ── Bottom badge (step 5+) ──────────────────────
  if step >= 5 {
    let span = nm * mw
    content((span / 2, ya - 0.55),
            box(fill: rgb("#2a2a68").lighten(90%),
                inset: (x: 0.55em, y: 0.28em), radius: 0.25em)[
              #text(size: ls * 0.88, fill: rgb("#2a2a68"))[→ *monthly feature matrix* · one row per calendar month]
            ], anchor: "north")
  }
})
