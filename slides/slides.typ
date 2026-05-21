#import "template.typ": *

#show: presentation

#title-slide[
  #text(fill: white, size: 1.3em, weight: "bold")[Macro Forecasting with the TFT and Fed Speeches]

  // #text(fill: white, size: 0.8em)[
  //   Advanced Machine Learning FS2026
  // ]
  #text(fill: white, size: 0.8em, weight: "semibold")[
    #v(0.5em)
    Presented by Minna Heim, Chris Traill and Anna Zeitz
  ]


  #speaker-notes[
    My name is Minna Heim Bla bla
  ]
]

#slide[
  = Motivation
  - Macro forecasts matter for governments, financial markets, and *central banks*
  - The Federal Reserve (Fed) has a *dual mandate*: price stability + maximum employment
  - Accurate forecasts of inflation, unemployment, and GDP are crucial for interest rate decisions
  #pause
  $=>$ Can we do better than standard macro models?
]

#slide[
  = Research Question
  - The Fed may communicate *additional information* useful for forecasting:
      + *Superior information*: private data and internal models not available to the public
      + *Forward guidance*: signals about future policy to guide market expectations
  #pause
  $=>$ *Do Fed speeches contain information useful to forecast macroeconomic indicators?*
]

// #slide[
//   = Our Approach

//   - Statistical Benchmarks:  
//       - AR(p)
//       - ARIMA(p,d,q)

//   - Temporal Fusion Transformer (Macro Only)
//   - Temporal Fusion Transformer (with Fed Speeches)
// ]

#slide[
  = Data
  *3 types of Data:*
  // this is just an example :)
  // #only("2-") shows this only from slide 2 onwards
  #only("2-")[
   - Macro Data
        - Quarterly, Monthly, Daily
        - E.g.: CPI, UNRATE, GDP, Exchange Rates, etc. 
  ]
  #only("3-")[
    - Meta Data 
        - Macro Metadata (Popularity, Frequency, Units)
        - FOMC Dissent 
        - Days till next FOMC Meeting, etc.
  ]
  #only("4-")[
    - Embedded Fed Speeches
        - E.g. _"Remarks by Mr Roger W Ferguson Jr, Vice-Chairman of the Board of Governors of the US Federal Reserve System, before the National Economic Association in Boston on 7 January 2000."_
        - Embedded (via FOMC-RoBERTa, FinBERT and Ollama,)
        - Dimensionality Reduction (None, PCA, Factor Analysis)
        - Speech Aggregation (mean, exp. decay, (context)attention-based)

  ]
]

#slide[

  = Show Data...

  by plotting, e.g.
  
]


#slide[
  = Models
  - Statistical Benchmarks (AR, ARIMA)
  - Temporal Fusion Transformer @TFT

]

#slide[
  = Hyperparameter Tuning
  show the grid, best values, etc.
]

#slide[
  = Results
  - Incl. Tables, Figures
]

#slide[
  = Robustness Checks
  - Shuffling Embeddings
  - Using New unrelated Text (German Texts of Franz Kafka)
]

#slide[
  = Future Work
  - More Hyperparam Tuning
  - Try out SSMs for Forecasting
  - Add Vintages for Macro Data for higher Accuracy 
  - Scrape more speeches (to have a longer Train Horizon)
]


#slide[
  #bibliography(
    "bibliography.bib",
    style: "apa",
  )
]


// #slide[
//   #v(1fr)
//   #set text(size: 1.5em)
//   #align(center)[
//     = Appendix
//   ]
//   #v(1fr)
// ]
