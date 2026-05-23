#import "@preview/cetz:0.5.0"

#let rnn-architecture = layout(size => {
  // approximate extents of the cetz drawing in canvas units
  let canvas-w = 16
  let canvas-h = 5
  let length = calc.min(size.width / canvas-w, size.height / canvas-h)
  cetz.canvas(length: length, {
    import cetz.draw: *

    set-style(stroke: 0.6pt)

    let cell-w = 1.4
    let cell-h = 1.1
    let dx = 2.4
    let dy = 1.8
    let radius = 0.18

    let input-fill = rgb("#bbe3b8")
    let hidden-fill = rgb("#a8cee8")
    let output-fill = rgb("#f4b8b8")

    // ---------- helpers --------------------------------------------------

    let cell(name, pos, label, color) = {
      rect(
        (rel: (-cell-w / 2, -cell-h / 2), to: pos),
        (rel: (cell-w / 2, cell-h / 2), to: pos),
        radius: radius,
        fill: color,
        stroke: 0.6pt + color.darken(40%),
        name: name,
      )
      if label != none {
        content(name + ".center", label)
      }
    }

    // a vertical input -> hidden -> output column at `pos`
    let column(id, pos, idx) = {
      cell("h-" + id, pos, none, hidden-fill)
      cell("u-" + id, (rel: (0, -dy), to: pos), $u_(#idx)$, input-fill)
      cell("x-" + id, (rel: (0, dy), to: pos), $x_(#idx)$, output-fill)
      line("u-" + id + ".north", "h-" + id + ".south", mark: (end: ">"))
      line("h-" + id + ".north", "x-" + id + ".south", mark: (end: ">"))
    }

    let arrow-label(p1, p2, label) = {
      content((p1, 50%, p2), label, anchor: "south", padding: 0.12)
    }

    // ---------- initial hidden state ------------------------------------

    cell("y0", (0, 0), $y_0$, input-fill)

    // ---------- first two timesteps -------------------------------------

    column("1", (rel: (dx, 0), to: "y0"), $1$)
    column("2", (rel: (dx, 0), to: "h-1"), $2$)

    line("y0.east", "h-1.west", mark: (end: ">"))
    line("h-1.east", "h-2.west", mark: (end: ">"))

    // ---------- left dots + arrow into step t ---------------------------

    content((rel: (dx, 0), to: "h-2"), $dots.c$, name: "dots-l")
    line("h-2.east", (rel: (-0.45, 0), to: "dots-l"), mark: (end: ">"))

    column("t", (rel: (dx, 0), to: "dots-l"), $t$)
    line((rel: (0.45, 0), to: "dots-l"), "h-t.west", mark: (end: ">"))

    // ---------- step t+1 ------------------------------------------------

    column("tp1", (rel: (dx, 0), to: "h-t"), $t + 1$)
    line("h-t.east", "h-tp1.west", mark: (end: ">"))

    // ---------- trailing dots ------------------------------------------

    content((rel: (dx, 0), to: "h-tp1"), $dots.c$, name: "dots-r")
    line("h-tp1.east", (rel: (-0.45, 0), to: "dots-r"), mark: (end: ">"))

    // ---------- arrow labels (y_(t-1), y_t, y_(t+1)) -------------------

    arrow-label((rel: (0.45, 0), to: "dots-l"), "h-t.west", $y_(t - 1)$)
    arrow-label("h-t.east", "h-tp1.west", $y_t$)
    arrow-label("h-tp1.east", (rel: (-0.45, 0), to: "dots-r"), $y_(t + 1)$)
  })
})

#rnn-architecture
