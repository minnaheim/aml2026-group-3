#import "@preview/polylux:0.4.0": *
#import "@preview/lovelace:0.3.1": *

#let heading-color = rgb("#2a2a68")
#let title-color = rgb("#08407E")

#let page-footer = context [
  #place(bottom + right)[
    #pad(bottom: 1.5em)[
      #text(size: 0.8em)[
        *#toolbox.slide-number* / #toolbox.last-slide-number
      ]
    ]
  ]
]

#let speaker-notes(notes) = context {
  let page-num = counter(page).display()
  let filename = "notes-slide-" + page-num + ".json"

  let note-data = (
    slide: page-num,
    notes: notes,
  )
  let json-string = json.encode(note-data)

  pdf.attach(
    filename,
    bytes(json-string),
    description: "Speaker notes for slide " + page-num,
    mime-type: "application/json",
  )
}

#let presentation(body) = {
  set page(paper: "presentation-16-9", footer: none, margin: 1.5cm)
  set text(font: "PT Sans", size: 1.6em)
  show heading.where(level: 1): set block(below: 1.2em)
  show heading.where(level: 1): set text(fill: heading-color, size: 1.2em)
  show figure.caption: set text(size: 0.8em)
  show footnote.entry: set text(size: 0.6em)
  body
}

#let title-slide(body) = slide[
  #block(
    height: 1fr,
    width: 100%,
    inset: (x: 0cm),
    align(horizon)[
      #grid(
        columns: (auto, 1fr, auto),
        align: (left, center, right),
        image("figures/uzh-logo.png", height: 1.5cm), [],
      )
    ],
  )

  #block(
    height: 13fr,
    width: 100%,
    {
      place(top + left)[
        #box(
          image("figures/uzh2.jpg", width: 100%, fit: "cover"),
          radius: 0.2em,
        )
      ]

      place(left + horizon, dy: 2.3cm, dx: -1.6em)[
        #block(
          fill: title-color,
          radius: 0.2em,
          width: 80%,
          inset: (x: 1.4em, y: 1.4em),
          outset: (y: 0cm),
          body,
        )
      ]
    },
  )
]

#let highlight(slide, color, content, next_slide: true) = {
  only(str(slide))[#text(fill: color, weight: "bold", content)]
  if next_slide {
    only("1-" + str(slide - 1) + "," + str(slide + 1) + "-")[#content]
  }
}

#let reveal(content, switch, color, from: "1") = {
  only(str(from) + "-" + str(switch - 1))[#content]
  only(str(switch) + "-")[#text(fill: color, weight: "bold", content)]
}

#let content-slides(body) = {
  set text(size: 1em)
  show heading.where(level: 1): set text(fill: title-color, size: 0.9em)
  show raw: set block(fill: silver.lighten(65%), width: 100%, inset: 1em)
  set page(footer: page-footer)
  body
}

#let math-template(doc) = {
  show math.equation: set text(font: "New Computer Modern Math")
  show math.equation.where(block: false): box

  set math.mat(delim: "[")
  set math.vec(delim: "[")

  set math.equation(numbering: "(1)")

  //// Make equation referencing only display the number.
  //show ref: it => {
  //  let el = it.element
  //  if el != none and el.func() == math.equation {
  //    // Override equation references.
  //    link(el.location(),numbering(
  //      el.numbering,
  //      ..counter(math.equation).at(el.location())
  //    ))
  //  } else {
  //    // Other references as usual.
  //    it
  //  }
  //}

  doc
}

#let avec(a) = math.bold(a)
#let vvec(a) = math.accent(math.bold(a), math.arrow)
#let nvec(a) = math.accent(avec(a), math.hat)

#let amat(a) = math.upright(math.bold(a))

#let xv = $avec(x)$
#let ii = $dotless.i$


#let angled(a) = math.lr($chevron.l #a chevron.r$)

#let inner(a, b) = angled($#a, #b$)
#let innerlines(a, b) = angled(math.vec(delim: none, a, b))

#let conj(u) = math.overline(u)
#let transp = math.tack.b
#let hert = math.upright(math.sans("H"))

#let clos(a) = math.overline(a)
#let openint(a, b) = $lr(\] #a, #b \[)$

#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)

#let mesh = $cal(M)$


#let wedge = math.and
#let wedgespace = math.scripts(math.inline(wedge.big))
#let hodge = math.class("unary", math.star)
#let sharp = "♯"
#let flat = "♭"

#let dif = math.class("unary", math.upright($d$))

#let grad = $avec("grad")$
#let curl = $avec("curl")$
#let div = $"div"$


#let Hvec = $avec(H)$
#let H0 = $limits(H)^circle.stroked.small$



#let restr(a) = $lr(#a|)$
#let trace = $"Tr"$

#let lin = $"Lin"$
#let alt = $"Alt"$

#let vol = "vol"

#let dom = "dom"
