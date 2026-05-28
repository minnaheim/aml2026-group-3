// Variable-selection importance tables for the interpretation slide.
// All functions read from out/holdout/interpretation/analysis/ CSVs.
// Paths are relative to the project root (compile with --root <repo-root>).

#let _base = "/out/holdout/interpretation/analysis/"

// helper: parse CSV, drop header, round floats to 4 dp, return array of 5-tuples (all strings)
// columns in CSV: variable, CPI, GDP, UNRATE, mean_importance, min_importance
#let _load(path) = {
  let raw  = csv(_base + path)
  let rows = raw.slice(1)
  rows.map(r => (
    r.at(0),
    str(calc.round(float(r.at(1)), digits: 4)),
    str(calc.round(float(r.at(2)), digits: 4)),
    str(calc.round(float(r.at(3)), digits: 4)),
    str(calc.round(float(r.at(4)), digits: 4)),
  ))
}

// wrap content in a figure if caption given, otherwise return as-is
#let _maybe-figure(content, caption, height) = {
  let inner = if height != auto {
    block(height: height, content)
  } else { content }

  if caption != none {
    figure(inner, caption: caption)
  } else { inner }
}

// shared booktabs-style table builder; rows are already string 5-tuples
#let _imp-table(rows, n, ascending: false, caption: none, height: auto) = {
  let sorted = rows.sorted(key: r => float(r.at(4)))
  let ordered = if ascending { sorted } else { sorted.rev() }
  let data = ordered.slice(0, calc.min(n, ordered.len()))
  let total = data.len()

  let t = {
    set text(size: 0.9em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      inset: (x: 0.8em, y: 0.5em),
      stroke: (_, y) => (
        top:    if y == 0 { 2pt } else if y == 1 { 2pt } else { none },
        bottom: if y == total { 2pt } else { none },
      ),
      [*Variable*], [*CPI*], [*GDP*], [*UNRATE*], [*Mean*],
      ..data.map(r => (
        [#r.at(0)],
        [#r.at(1)],
        [#r.at(2)],
        [#r.at(3)],
        [*#r.at(4)*],
      )).flatten()
    )
  }
  _maybe-figure(t, caption, height)
}

#let encoder-table(n: 10, caption: none, height: auto) = {
  _imp-table(_load("importance_encoder_variables_h12_none.csv"), n,
             caption: caption, height: height)
}

#let decoder-table(n: 10, caption: none, height: auto) = {
  _imp-table(_load("importance_decoder_variables_h12_none.csv"), n,
             caption: caption, height: height)
}

#let static-table(n: 10, caption: none, height: auto) = {
  _imp-table(_load("importance_static_variables_h12_none.csv"), n,
             caption: caption, height: height)
}

// all encoder variables with mean importance < cutoff, sorted worst-first (ascending)
#let removal-table(cutoff: 0.01, caption: none, height: auto) = {
  let rows = _load("importance_encoder_variables_h12_none.csv")
  let candidates = rows
    .filter(r => float(r.at(4)) < cutoff)
    .sorted(key: r => float(r.at(4)))
  let total = candidates.len()

  let t = {
    set text(size: 0.9em)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      inset: (x: 0.8em, y: 0.5em),
      stroke: (_, y) => (
        top:    if y == 0 { 2pt } else if y == 1 { 2pt } else { none },
        bottom: if y == total { 2pt } else { none },
      ),
      [*Variable*], [*CPI*], [*GDP*], [*UNRATE*], [*Mean* #sym.arrow.t],
      ..candidates.map(r => (
        [#r.at(0)],
        [#r.at(1)],
        [#r.at(2)],
        [#r.at(3)],
        [#r.at(4)],
      )).flatten()
    )
  }
  _maybe-figure(t, caption, height)
}
