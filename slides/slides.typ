#import "template.typ": *
#import "figures/data_timeline.typ": data-timeline
#import "figures/metrics_table.typ": metrics-table

#show: presentation
#set par(leading: 1.2em)

#title-slide[
  #text(fill: white, size: 1.3em, weight: "bold")[Macro Forecasting with the TFT and Fed Speeches]

  // #text(fill: white, size: 0.8em)[
  //   Advanced Machine Learning FS2026
  // ]
  #text(fill: white, size: 0.8em, weight: "semibold")[
    #v(0.5em)
    Presented by Minna Heim, Chris Traill and Anna Zeitz
  ]


  // #speaker-notes[
  //   My name is Minna Heim Bla bla
  // ]
]

#show: content-slides

#slide[
  = Motivation
  - Macro forecasts matter for
    - Governments
    - Financial markets
    - *Central banks*
  #only("2-")[
    - The Federal Reserve (Fed) has a *dual mandate*:
      + Price stability
      + Maximum employment
    - Accurate forecasts of inflation, unemployment, and GDP are crucial for interest rate decisions
  ]
  #only("3-")[
    $=>$ Can we do better than standard macro models?
  ]
]
// TODO: no mention of problem setting here, maybe included?
#slide[
  = Research Question
  - The Fed communicates with the public through official statements and speeches
  - It may communicate additional information:
    + *Superior information*: private data and internal models not available to the public
    + *Forward guidance*: signals about future policy to guide market expectations
  #only("2-")[
    #v(1em)
    $=>$ *Do Fed speeches contain information useful to forecast macroeconomic indicators?*
  ]
]

#slide[
  = Data Alignment
  // #data-timeline
    #align(center)[
      #v(2em)
    #box(width: 90%)[
      #import "figures/data_timeline.typ": 
      #data-timeline

    ]
  ]
]

#slide[
  #v(1fr)
  #align(center)[= Models]
  #v(1fr)
]



#slide[
  = Benchmarks

  #let hi-int = rgb("#C96820")  // integration: orange
  #let hi-ma  = rgb("#3A6BB5")  // MA:          blue

  #v(5em)
  #grid(
    columns: (15em, 1fr),
    column-gutter: 1.2em,
    row-gutter: 0.9em,
    align: (right + horizon, left + horizon),

    [AR(p)],[$display(y_t = c + sum_(i=1)^p phi_i y_(t-i) + epsilon_t)$],

  // show arima only below?
  [#v(2em)
  ARIMA(p, d, q)
  ],
  [
    #v(2em)
    $display(
    #text(fill: hi-int)[$Delta^d$] 
    y_t = c
    + sum_(i=1)^p phi_i thin 
    #text(fill: hi-int)[$Delta^d$]
      y_(t-i)
    + #text(fill: hi-ma)[$sum_(j=1)^q theta_j epsilon_(t-j)$]
    + epsilon_t
  )$
    ],
  )
    #v(2em)
    #text(size: 0.82em)[
      #box(fill: hi-int, width: 0.7em, height: 0.7em, radius: 0.1em)
      #h(0.2em) Integration: $d$-th difference of $y_t$
      #h(1.5em)
      #box(fill: hi-ma, width: 0.7em, height: 0.7em, radius: 0.1em)
      #h(0.2em) MA component: $q$ lags of residuals $epsilon_t$
    ]
]

#slide[
  = Temporal Fusion Transformer
  // Add Abbildung from TFT paper here maybe?; then explain it, also saying what variables we use for the static covariates etc; in appendix, list what the actual variables are?
  // if I use AR(IMA) equation, should i use TFT equations too?
    #place(center)[
    #v(1em)
    #image("figures/tft-arch.jpg", height: 90%)
  ]

]


// maybe as a nice übergang
#slide[
  #v(1fr)
  #align(center)[= Speech Embeddings]
  #v(1fr)
]

#slide[
  = Speech Embeddings
  Two models chosen for complementary perspectives:
  + *FinBERT*: BERT-based, pre-trained on financial statements @yang2020finbertpretrainedlanguagemodel
  + *FOMC-RoBERTa*: fine-tuned on FOMC data, hawkish/dovish classification @fomc_roberta_paper
  #only("2-")[
    #v(0.5em)
    *Processing*: chunk-and-average (512-token windows, 128-token overlap)
    - Assumption: information spread across entire speech, not just the beginning
    #v(0.5em)
    *Dimensionality reduction*: 768 $=>$ $X$ dims via Principal Component Analysis or Factor Analysis
    - $X$ is treated as hyperparameter
  ]
  // TODO: either we choose to do robustness here, or we remove this and put this to robustness
  // #only("3-")[
  // #v(0.5em)
  // *Robustness*: replace embeddings with Kafka texts $=>$ should add no predictive signal
  // ]
]

#slide[
  = Speech Alignment

  Speeches are held at *irregular* and *daily* frequency: how to align with monthly horizon?
  #v(0.5em)

  + *Mean aggregation*: unweighted mean over rolling window of $X$ months
  + *Exponential decay*: $w_t = exp(-lambda dot d_t)$ where $d_t$ = days before month $m$
    - More recent speeches receive higher weight
    - $lambda$ treated as fixed; speech window $X$ tuned as hyperparameter
  + *Attention-based*: key = embedding (after dimensionality reduction), query = mean macro state
    - Given current macro conditions, which past speeches are most informative?
    - Fitted on training data only $=>$ no leakage!
  #v(0.5em)
  All methods also track: voter speech ratio, chair speech ratio, gender share

]


#slide[
  // TODO: use hyperparam tuning grid, put it into main part of presentation
  = Hyperparameter Tuning
  Tune in two stages:
  + *Stage 1: Architecture (Macro-Only TFT)*
    - Bayesian search via Optuna (TPE sampler)
    - 2-fold CV, up to 50 trials
    - Includes: encoder length, hidden size, normalizer
  + *Stage 2: Speech Embedding Params*
    - Architecture fixed from Stage 1
    - Tune: aggregation, reduction, PCA dims, speech window
]



#slide[
  = Evaluation Protocol
  - *Walk-forward cross-validation*: expanding window, never use future data
  - *Metric*: MAE (mean absolute error), RMSE ????
  ADJUST HERE
  - *Multi-step forecasting*: predictions in non-overlapping 12-month windows
    - Context: training data + model's own previous predictions
    - Same assumption for AR, ARIMA and TFT $=>$ fair comparison
  #only("2-")[
    #v(0.5em)
    $=>$ All models evaluated on the same information set
  ]
]

#slide[
  #v(1fr)
  #align(center)[= Results]
  #v(1fr)
]

#slide[
  = Results
  - Incl. Tables, Figures
]

#slide[
  = Robustness Checks
  - Shuffling Embeddings
  - Using unrelated text (German Texts of Franz Kafka)
  - Use not full speech but only first 512 tokens of speech???
]

#slide[
  = Future Work
  - Hyperparameter tuning: switch to one stage tuning! Also: context-dependent attention
  - Try out SSMs for forecasting
    - Combine with TFT: replace LSTM with SSM for sequence encoder // better for long-run dependencies, also more targeted for time series data!
  - Add Vintages: use real-time macro data (as available at forecast time) for more realistic evaluation
  - Extend dataset: Working Paper @anna_snb_2026
    - 10,000+ speeches since 1914, continuously updated
]


#slide[
  #bibliography(
    "bibliography.bib",
    style: "apa",
  )
]


// save the last main slide number
#let main-slides = toolbox.last-slide-number

#set page(footer: context [
  #place(bottom + right)[
    #pad(bottom: 1.5em)[
      #text(size: 0.8em)[
        *#toolbox.slide-number* / #main-slides #text(fill: gray)[ | Appendix]
      ]
    ]
  ]
])

#slide[
  #v(1fr)
  #align(center)[= Appendix]
  #v(1fr)
]

// TODO: add slide on data visualisation (to explain why UNRATE vs. CPI, GDP so different in MAE!)

#slide[
  = Variable List: Macroeconomic Variables
  + *Target variables*
    - CPI, UNRATE, GDP (quarterly: forward-fill missing months) $=>$ AR and ARIMA only take targets
  + *Covariates*
    - Monthly variables: payroll (PAYEMS), industrial production (INDPRO), hours worked in manufacturing (AWHMAN), OECD composite leading indicator (USACLI)
    - Weekly / daily variables: exchange rates (GBP, YEN), effective federal funds rate (FFR), Fed balance sheet (WALCL)
      - Higher-frequency variables: take mean *and* std per month (volatility)
    - Lags at 1, 2, 6, 12 months for all target variables
  + *Preprocessing*: log-difference of CPI, GDP, PAYEMS, INDPRO, AWHMAN
]

#slide[
  = Inspect Data Frame:
  // titles of vars arent good yet... still need to fix this
  #place(center)[
    #v(2em)
    #image("figures/df_head.png", width: 100%)
  ]
]

#slide[
  = Inspect Target Vars
  // Gray bands = NBER recessions; CPI and GDP are log-differenced
  #place(center)[
    #v(0.5em)
    #image("figures/target_vars.png", width: 65%)
  ]
]

#slide[
  = Variable List
  The TFT takes four different types of variables:

  + *Static categoricals*: time-invariant, categorical
    - Identifier and frequency of each time series
  + *Static reals*: time-invariant, continuous
    - Number downloads time series from FRED, how long time series has existed
  + *Time-Varying Known Reals*: time-varying, known
    - Day of week, month, year; if day is holiday
    - FOMC Meeting dates
  + *Time-Varying Unknown Reals*: time-varying, unknown
    - Macroeconomic variables, speech embeddings, FOMC dissents

]

#slide[
  = TFT: Input Types
  // cite paper here?
  #place(center)[
    #v(1em)
    #image("figures/tft-inputs.png", height: 80%)
  ]
]


#slide[
  = Dimensionality Reduction

  FinBERT and FOMC-RoBERTa embeddings are of dimension $768$ per speech

  $=>$ Dimensionality reduction needed for TFT input

  We test two methods (number of components $X$ is hyperparameter, fit only on training data):
  + *PCA*: projects embeddings onto $X$ principal components
    - 20 components $approx$ 80% of variance explained
  + *Factor Analysis (FA)*: models embeddings as linear combination of $X$ latent factors
    - Unlike PCA: explicitly models noise, focuses on *shared* variance
    - Parameters estimated via EM algorithm

]


Based on your tuning code, here are the two appendix slides filled out:
typst#slide[
  = Hyperparameter Tuning: Stage 1 Search Space
  Architecture tuned on macro-only TFT (per target × horizon):

  #table(
    columns: (auto, auto, auto),
    [*Parameter*], [*Type*], [*Range / Values*],
    [Encoder length], [int], [12 – 48],
    [Hidden size], [categorical], [\{8, 16, 32, 64, 128, 256\}],
    [Hidden continuous size], [categorical], [\{2, 4, 8, 16\}],
    [Dropout], [float], [0.05 – 0.55],
    [Learning rate], [log-uniform], [$10^(-4)$ – 0.15],
    [Normalizer], [categorical], [\{encoder-none, group\}],
  )
  #v(0.5em)
  Fixed: `lstm_layers = 2`, `attention_head_size = 2`, `max_epochs = 50`, `patience = 15`
]

#slide[
  = Hyperparameter Tuning: Stage 2 Search Space
  Speech embedding params tuned (architecture fixed from Stage 1):

  #table(
    columns: (auto, auto, auto),
    [*Parameter*], [*Type*], [*Range / Values*],
    [Aggregation], [categorical], [\{mean, decay, attention\}],
    [Reduction], [categorical], [\{pca, fa\}],
    [PCA components ($X$)], [int], [5 – 30],
    [Speech window (months)], [int], [3 – 12],
  )
  #v(0.5em)
  Tuned separately for each of 9 combinations: \{CPI, GDP, UNRATE\} $times$ \{3, 6, 12\} months
]


#slide[
  = Hyperparameter Tuning: Stage 2 -- Optimal Hyperparameters
  #block[
    #set text(size: 14.5pt)
    #set par(leading: 0.3em)
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      table.header(
        [], table.cell(colspan: 3)[*FOMC-RoBERTa*], table.cell(colspan: 3)[*FinBERT*],
        [*Target*], [*h=3*], [*h=6*], [*h=12*], [*h=3*], [*h=6*], [*h=12*],
      ),
      [*CPI*],
      [Mean \ PCA, n = 21 \ W = 3 \ MAE = 0.001847],
      [Attention \ FA, n = 10 \ W = 8 \ MAE = 0.001849],
      [*Attention \ PCA, n = 12\ W = 9 \ MAE = 0.001842*],
      [*Attention \ FA, n = 20 \ W = 5 \ MAE = 0.001846*],
      [*Decay \ PCA, n = 23 \ W = 3 \ MAE = 0.001846*],
      [Mean \ PCA, n = 8 \ W = 7 \ MAE = 0.001847],

      [*GDP*],
      [*Mean \ PCA, n = 8 \ W = 7 \ MAE = 0.00154*],
      [Attention \ FA, n = 12 \ W = 8 \ MAE = 0.00171],
      [Attention \ FA, n = 5 \ W = 10 \ MAE = 0.00152],
      [Attention \ FA, n = 6 \ W = 12 \ MAE = 0.00170],
      [*Attention \ FA, n = 10 \ W = 8 \ MAE = 0.00157*],
      [*Decay \ PCA, n = 20 \ W = 12 \ MAE = 0.00151*],

      [*UNRATE*],
      [Decay \ FA, n = 12 \ W = 11 \ MAE = 1.31293],
      [Attention \ FA, n = 6 \ W = 12 \ MAE = 1.31501],
      [*Mean \ PCA, n = 8 \ W = 7 \ MAE = 1.11597*],
      [*Mean \ FA, n = 23 \ W = 5 \ MAE = 1.26152*],
      [*Mean \ PCA, n = 14 \ W = 4 \ MAE = 1.30896*],
      [Mean \ PCA, n = 25 \ W = 4 \ MAE = 1.20808],
    )]

  where W refers to the speech window size and n to the optimal number of components for the dimensionality reduction. The best result per target$times$horizon is shown in bold.

]

#slide[
  #v(1fr)
  #align(center)[= Results: In Sample]
  #v(1fr)
]

#slide[
  = Macro Only: h=12 
  #speaker-notes[
    ran with `python src/holdout-validation/e_main.py --tuned --wandb --horizon 12` 
    - h=12
    - macro only
    - HP tuned architecture:
    - targets: all
  ]
  #only("1")[
    #place(center)[
    #image("../out/holdout/default/predictions_vs_actuals_h12_macro.png")
  ]
  ]
  #only("2")[
    #v(3em)
    #place(center)[
    #metrics-table(path: "../../out/holdout/default/metrics_h12_macro.csv")
    ]
  ]

]
#slide[
  = with Embeddings: h=12 
  #speaker-notes[
    ran with `python src/holdout-validation/e_main.py --tuned --wandb --horizon 12` 
    - h=12
    - with embeddings
    - HP tuned architecture + embeddings:
    - targets: all
  ]
  #only("1")[
    plot
  ]
  #only("2")[
    metrics
  ]
]