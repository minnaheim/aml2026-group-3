// Reusable holdout-metrics table.
// Usage (from slides.typ):
//   #import "figures/metrics_table.typ": metrics-table
//   #metrics-table(path: "../../out/holdout/default/metrics_h12_macro.csv")
//
// path is relative to THIS file (slides/figures/), so prefix with "../../".
// Shows: Model · Target · MAE · RMSE  — lowest-MAE row per target is bold.

#let metrics-table(path: "") = {
  let raw  = csv(path)
  let hdrs = raw.first()
  let rows = raw.slice(1)

  let idx(name) = hdrs.position(h => h == name)
  let i-model  = idx("model")
  let i-target = idx("target")
  let i-mae    = idx("MAE")
  let i-rmse   = idx("RMSE")

  // lowest MAE per target — determines which row gets bolded
  let min-mae(t) = rows
    .filter(r => r.at(i-target) == t)
    .map(r => float(r.at(i-mae)))
    .fold(1e10, (a, b) => calc.min(a, b))

  let n = rows.len()
  table(
    columns: 4,
    align: (left, left, right, right),
    inset: (x: 0.6em, y: 0.35em),
    // booktabs-style: thick top + bottom, thin mid rule after header
    stroke: (_, y) => (
      top:    if y == 0 { 1pt } else if y == 1 { 0.4pt } else { none },
      bottom: if y == n { 1pt } else { none },
    ),
    [*Model*], [*Target*], [*MAE* #sym.arrow.b], [*RMSE* #sym.arrow.b],
    ..rows.map(row => {
      let best = float(row.at(i-mae)) == min-mae(row.at(i-target))
      let cell(v) = if best { [*#v*] } else { [#v] }
      (
        cell(row.at(i-model)),
        cell(row.at(i-target)),
        cell(row.at(i-mae)),
        cell(row.at(i-rmse)),
      )
    }).flatten()
  )
}
