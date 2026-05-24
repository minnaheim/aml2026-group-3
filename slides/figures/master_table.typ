// Master holdout results table.
// Rows: (h, model) — 3 horizons × 4 models = 12 rows.
// Columns: CPI · GDP · UNRATE, each with MAE and RMSE.
// Best MAE per (h, target) is bolded.
// Compile with: typst compile slides/slides.typ --root .

#let _parse(path) = {
  let raw = csv(path)
  let hdr = raw.first()
  let i(n) = hdr.position(x => x == n)
  let im = i("model")
  let it = i("target")
  let imae = i("MAE")
  let irmse = i("RMSE")
  raw
    .slice(1)
    .map(r => (
      model: r.at(im),
      target: r.at(it),
      mae: r.at(imae),
      rmse: r.at(irmse),
    ))
}

#let _get(rows, model, target) = {
  let hit = rows.filter(r => r.model == model and r.target == target)
  if hit.len() > 0 { hit.first() } else { (mae: "–", rmse: "–") }
}

#let _fmt(v) = {
  if v == "–" { "–" } else { str(calc.round(float(v), digits: 5)) }
}

#let master-table() = {
  let specs = (
    (
      h: 3,
      macro: _parse("/out/holdout/final_holdout/metrics_h3_macro.csv"),
      emb: _parse("/out/holdout/final_emb/metrics_h3_auto.csv"),
    ),
    (
      h: 6,
      macro: _parse("/out/holdout/final_holdout/metrics_h6_macro.csv"),
      emb: _parse("/out/holdout/final_emb/metrics_h6_auto.csv"),
    ),
    (
      h: 12,
      macro: _parse("/out/holdout/final_holdout/metrics_h12_macro.csv"),
      emb: _parse("/out/holdout/final_emb/metrics_h12_auto.csv"),
    ),
  )

  let targets = ("CPI", "GDP", "UNRATE")
  // (display label, data source key, csv model name)
  let model-defs = (
    ("AR", "macro", "AR"),
    ("ARIMA", "macro", "ARIMA"),
    ("TFT", "macro", "TFT"),
    ("TFT+Emb", "emb", "TFT"),
  )
  let n-models = model-defs.len() // 4

  // min MAE per (horizon-idx, target-idx) across all 4 models
  let mins = specs.map(sp => targets.map(t => model-defs
    .map(mdef => {
      let src = if mdef.at(1) == "macro" { sp.macro } else { sp.emb }
      let r = _get(src, mdef.at(2), t)
      if r.mae == "–" { 1e10 } else { float(r.mae) }
    })
    .fold(1e10, (a, b) => calc.min(a, b))))

  // 2 header rows + 3 × 4 data rows = 14 total
  let n-rows = 2 + specs.len() * n-models

  // Layout: h | Model | CPI_MAE | CPI_RMSE | [gap] | GDP_MAE | GDP_RMSE | [gap] | UNRATE_MAE | UNRATE_RMSE
  // 10 columns total; gap columns (idx 4, 7) are narrow spacers with no content or stroke.
  let gap = 0.9em

  set text(size: 0.72em)

  table(
    columns: (auto, auto, auto, auto, gap, auto, auto, gap, auto, auto),
    align: (center, left, right, right, center, right, right, center, right, right),
    // RMSE columns (3, 6, 9) get extra left padding to separate them from MAE
    inset: (x, y) => if x == 3 or x == 6 or x == 9 {
      (left: 1.4em, right: 0.55em, top: 0.45em, bottom: 0.45em)
    } else {
      (x: 0.55em, y: 0.45em)
    },
    stroke: (x, y) => {
      // suppress all stroke on the two spacer columns
      if x == 4 or x == 7 { return (top: none, bottom: none, left: none, right: none) }
      (
        top: if y == 0 or y == 2 { 1.5pt } else if y == 1 { 0.4pt } else if y > 2 and calc.rem(y - 2, n-models) == 0 {
          0.7pt
        } else { none },
        bottom: if y == n-rows - 1 { 1.5pt } else { none },
      )
    },

    // ── Header row 1: target group labels ──────────────
    [], [],
    table.cell(colspan: 2, align: center)[*CPI*],
    [],
    // gap
    table.cell(colspan: 2, align: center)[*GDP*],
    [],
    // gap
    table.cell(colspan: 2, align: center)[*UNRATE*],
    // ── Header row 2: metric labels ──────────────────
    [*$h$*], [*Model*],
    [*MAE* #sym.arrow.b], [*RMSE* #sym.arrow.b],
    [],
    // gap
    [*MAE* #sym.arrow.b], [*RMSE* #sym.arrow.b],
    [],
    // gap
    [*MAE* #sym.arrow.b], [*RMSE* #sym.arrow.b],

    // ── Data rows ─────────────────────────────────────
    ..specs
      .enumerate()
      .map(hi-sp => {
        let hi = hi-sp.first()
        let sp = hi-sp.last()
        model-defs
          .enumerate()
          .map(mi-mdef => {
            let mi = mi-mdef.first()
            let mdef = mi-mdef.last()
            let (label, srck, csvkey) = mdef
            let src = if srck == "macro" { sp.macro } else { sp.emb }
            let h-cell = if mi == 0 { [*#sp.h*] } else { [] }
            // build (mae, rmse) pairs per target, separated by gap cells
            let pairs = targets
              .enumerate()
              .map(ti-t => {
                let ti = ti-t.first()
                let t = ti-t.last()
                let r = _get(src, csvkey, t)
                let best = r.mae != "–" and calc.abs(float(r.mae) - mins.at(hi).at(ti)) < 1e-9
                (
                  if best { [*#_fmt(r.mae)*] } else { [#_fmt(r.mae)] },
                  if best { [*#_fmt(r.rmse)*] } else { [#_fmt(r.rmse)] },
                  // [#_fmt(r.rmse)],
                )
              })
            // interleave gap cells: CPI_pair + gap + GDP_pair + gap + UNRATE_pair
            let vals = pairs.at(0) + ([],) + pairs.at(1) + ([],) + pairs.at(2)
            (h-cell, [#label]) + vals
          })
          .flatten()
      })
      .flatten(),
  )
}
