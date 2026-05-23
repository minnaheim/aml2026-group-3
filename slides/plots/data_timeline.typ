// Data alignment timeline
// Shows how monthly, quarterly, daily, and speech data are fused to a monthly feature matrix.
#import "@preview/cetz:0.5.0": canvas, draw

#let data-timeline = canvas(length: 1cm, {
  import draw: *

  // ── Palette ─────────────────────────────────────
  let c-m  = rgb("#3A6BB5")  // monthly  — blue
  let c-q  = rgb("#C96820")  // quarterly — orange
  let c-d  = rgb("#3E7228")  // daily    — green
  let c-s  = rgb("#AA1818")  // speeches — red
  let c-ax = rgb("#222222")
  let c-ar = rgb("#999999")  // merge arrows

  // ── Layout ──────────────────────────────────────
  let mw  = 2.4   // month width (cm)
  let rh  = 0.62  // row height (cm)
  let gap = 0.30  // gap between rows
  let lx  = -4.1  // left label anchor x
  let nm  = 5     // months shown
  let ls  = 0.58em

  // Row y-positions (bottom edge of band)
  let ya = 0.0
  let ym = 0.42
  let yq = ym + rh + gap
  let yd = yq + rh + gap
  let ys = yd + rh + gap + 0.10

  // ── Time axis ──────────────────────────────────
  line((-0.3, ya), (nm * mw + 0.6, ya),
       stroke: c-ax, mark: (end: "stealth", scale: 0.5))
  for i in range(nm + 1) {
    line((i * mw, ya - 0.06), (i * mw, ya + 0.06), stroke: c-ax)
  }
  let mlabels = ("Jan", "Feb", "Mar", "Apr", "May")
  for i in range(nm) {
    content((i * mw + mw / 2, ya - 0.13),
            text(size: ls)[#mlabels.at(i) 2020], anchor: "north")
  }

  // ── Monthly Macro ──────────────────────────────
  for i in range(nm) {
    let x0 = i * mw + 0.03
    let x1 = (i + 1) * mw - 0.03
    rect((x0, ym), (x1, ym + rh),
         fill: c-m.lighten(82%), stroke: (paint: c-m, thickness: 0.7pt))
    content(((x0 + x1) / 2, ym + rh / 2),
            text(size: ls * 0.82, fill: c-m.darken(25%))[CPI · UNRATE\ PAYEMS · INDPRO],
            anchor: "center")
  }
  content((lx, ym + rh / 2),
          align(right)[*Monthly Macro*\ #text(size: ls * 0.9)[monthly observations]],
          anchor: "east")

  // ── Quarterly GDP ──────────────────────────────
  // Q1 2020 spans Jan–Mar (indices 0–2); Q2 2020 spans Apr–May (indices 3–4, partial)
  let qtrs = ((0, 3, "Q1 2020"), (3, 2, "Q2 2020"))
  for (qi, qn, ql) in qtrs {
    let x0 = qi * mw + 0.03
    let x1 = (qi + qn) * mw - 0.03
    rect((x0, yq), (x1, yq + rh),
         fill: c-q.lighten(82%), stroke: (paint: c-q, thickness: 0.7pt))
    content(((x0 + x1) / 2, yq + rh / 2),
            text(size: ls * 0.82, fill: c-q.darken(25%))[#ql GDP],
            anchor: "center")
    // Dashed lines showing forward-fill boundaries within quarter
    for j in range(1, qn) {
      let xd = (qi + j) * mw
      line((xd, yq + 0.08), (xd, yq + rh - 0.08),
           stroke: (paint: c-q.lighten(10%), dash: "dashed", thickness: 0.5pt))
    }
  }
  content((lx, yq + rh / 2),
          align(right)[*Quarterly GDP*\ #text(size: ls * 0.9)[fwd-filled · ÷ 3 per month]],
          anchor: "east")

  // ── Daily Financial ────────────────────────────
  for i in range(nm) {
    let x0 = i * mw + 0.03
    let x1 = (i + 1) * mw - 0.03
    let w  = x1 - x0
    // ~22 trading-day ticks
    for d in range(22) {
      let dx = x0 + (d + 0.5) * w / 22.0
      line((dx, yd + 0.09), (dx, yd + rh - 0.09),
           stroke: (paint: c-d.lighten(30%), thickness: 0.32pt))
    }
    rect((x0, yd), (x1, yd + rh), stroke: (paint: c-d, thickness: 0.7pt))
    content(((x0 + x1) / 2, yd + rh / 2),
            text(size: ls * 0.82, fill: c-d.darken(20%))[$mu, sigma$],
            anchor: "center")
  }
  content((lx, yd + rh / 2),
          align(right)[*Daily Financial*\ #text(size: ls * 0.9)[GBP · YEN · FFR · T10Y3M\ → mean & std per month]],
          anchor: "east")

  // ── FOMC Speeches ──────────────────────────────
  // Irregular speech positions as fractions of total span [0, nm*mw]
  let s-fracs = (0.03, 0.10, 0.20, 0.30, 0.42, 0.50, 0.59, 0.68, 0.76, 0.87, 0.92, 0.97)
  let span = nm * mw

  // Light background band
  rect((0.0, ys), (span, ys + rh), fill: c-s.lighten(94%), stroke: none)

  for f in s-fracs {
    let sx = f * span
    line((sx, ys + 0.05), (sx, ys + rh - 0.10),
         stroke: (paint: c-s.lighten(25%), thickness: 0.9pt))
    circle((sx, ys + rh - 0.10), radius: 0.085, fill: c-s, stroke: none)
  }

  // "…" indicating speeches continue to the left (12-month window goes back further)
  content((-0.28, ys + rh / 2), text(size: 0.9em, fill: c-s)[…], anchor: "center")

  content((lx, ys + rh / 2),
          align(right)[*FOMC Speeches*\ #text(size: ls * 0.9)[irregular daily freq.\ 768-dim → PCA]],
          anchor: "east")

  // ── Rolling-window bracket (example: aggregating for Apr 2020) ─
  let bky    = ys + rh + 0.22
  let w-end  = 3.0 * mw   // Apr 2020 = left edge of month index 3
  let tk     = 0.12        // tick height

  // Main bracket from visible start to Apr
  line((0.0, bky), (w-end, bky), stroke: (paint: c-s, thickness: 1pt))
  line((0.0, bky - tk), (0.0, bky), stroke: (paint: c-s, thickness: 1pt))
  line((w-end, bky - tk), (w-end, bky), stroke: (paint: c-s, thickness: 1pt))

  // Dashed extension to the left (window reaches 12 months back, off-screen)
  line((0.0, bky), (-0.55, bky),
       stroke: (paint: c-s, thickness: 1pt, dash: "dashed"),
       mark: (end: "stealth", scale: 0.45))

  // Bracket label
  content((w-end / 2 - 0.3, bky + 0.13),
          text(size: ls * 0.90, fill: c-s)[← 12-month rolling window (example: Apr 2020)],
          anchor: "south")

  // ── Merge arrows at ~Feb (show alignment direction) ────────────
  // Drawn at the right third of Feb column so they don't overlap month labels
  let ax = 1.0 * mw + mw * 0.65

  // Speeches → Daily
  line((ax, ys), (ax, yd + rh + 0.04),
       stroke: (paint: c-ar, thickness: 0.42pt, dash: "dashed"),
       mark: (end: "stealth", scale: 0.38))
  // Daily → Quarterly
  line((ax, yd), (ax, yq + rh + 0.04),
       stroke: (paint: c-ar, thickness: 0.42pt, dash: "dashed"),
       mark: (end: "stealth", scale: 0.38))
  // Quarterly → Monthly
  line((ax, yq), (ax, ym + rh + 0.04),
       stroke: (paint: c-ar, thickness: 0.42pt, dash: "dashed"),
       mark: (end: "stealth", scale: 0.38))

  // ── Bottom label ───────────────────────────────
  content((span / 2, ya - 0.55),
          box(
            fill: rgb("#2a2a68").lighten(90%),
            inset: (x: 0.55em, y: 0.28em),
            radius: 0.25em,
          )[#text(size: ls * 0.90, fill: rgb("#2a2a68"))[→ *monthly feature matrix* · one row per month-year]],
          anchor: "north")
})

#data-timeline