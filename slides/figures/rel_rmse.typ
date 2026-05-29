#let _parse_relrmse(path) = {
  let raw = csv(path)
  let hdr = raw.first()
  let i(n) = hdr.position(x => x == n)
  let ih = i("horizon")
  let it = i("target")
  let imacro = i("RMSE_macro")
  let iemb = i("RMSE_emb")
  let irel = i("relRMSE")
  raw
    .slice(1)
    .map(r => (
      horizon: r.at(ih),
      target: r.at(it),
      rmse_macro: r.at(imacro),
      rmse_emb: r.at(iemb),
      relrmse: r.at(irel),
    ))
}

#let _get_rel(rows, horizon, target) = {
  let hit = rows.filter(r => r.horizon == str(horizon) and r.target == target)
  if hit.len() > 0 { hit.first() } else { (rmse_macro: "–", rmse_emb: "–", relrmse: "–") }
}

#let relrmse-table() = {
  let rows = _parse_relrmse("/out/holdout/relRMSE.csv")
  let horizons = (3, 6, 12)
  let targets = ("CPI", "GDP", "UNRATE")

  set text(size: 0.72em)

  align(center, table(
    columns: (auto, auto, auto, auto, 0.9em, auto, auto, auto, 0.9em, auto, auto, auto),
    align: (center, right, right, right, center, right, right, right, center, right, right, right),
    stroke: (x, y) => {
      if x == 4 or x == 8 { return (top: none, bottom: none, left: none, right: none) }
      (
        top: if y == 0 or y == 2 { 1.5pt } else if y == 1 { 0.4pt } else if y > 2 and calc.rem(y - 2, 3) == 0 { 0.7pt } else { none },
        bottom: if y == horizons.len() * 3 + 2 - 1 { 1.5pt } else { none },
      )
    },

    // header row 1
    [], 
    table.cell(colspan: 3, align: center)[*CPI*], [],
    table.cell(colspan: 3, align: center)[*GDP*], [],
    table.cell(colspan: 3, align: center)[*UNRATE*],

    // header row 2
    [*$h$*],
    [*RMSE#sub[TFT]* #sym.arrow.b], [*RMSE#sub[+Emb]* #sym.arrow.b], [*relRMSE* #sym.arrow.b], [],
    [*RMSE#sub[TFT]* #sym.arrow.b], [*RMSE#sub[+Emb]* #sym.arrow.b], [*relRMSE* #sym.arrow.b], [],
    [*RMSE#sub[TFT]* #sym.arrow.b], [*RMSE#sub[+Emb]* #sym.arrow.b], [*relRMSE* #sym.arrow.b],

    // data rows
    ..horizons.map(h => {
      let cells = targets.enumerate().map(ti-t => {
        let ti = ti-t.first()
        let t = ti-t.last()
        let r = _get_rel(rows, h, t)
        let fmt4(v) = {
          let s = str(calc.round(float(v), digits: 4))
          let parts = s.split(".")
          let dec = if parts.len() > 1 { parts.at(1) } else { "" }
          let padded = dec + "0" * (4 - dec.len())
          parts.at(0) + "." + padded
        }
        
        let rel = if r.relrmse == "–" { "–" } else { fmt4(r.relrmse) }
        let better = r.relrmse != "–" and float(r.relrmse) < 1.0
        (
          [#str(calc.round(float(r.rmse_macro), digits: 5))],
          [#str(calc.round(float(r.rmse_emb), digits: 5))],
          if better { [*#rel*] } else { [#rel] },
        )
      })
      let vals = cells.at(0) + ([],) + cells.at(1) + ([],) + cells.at(2)
      ([*#h*],) + vals
    }).flatten(),
    table.hline(stroke: 1.5pt),
  ))
}
