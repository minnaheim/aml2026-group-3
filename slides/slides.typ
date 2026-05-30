#import "template.typ": *
#import "figures/data_timeline.typ": data-timeline
#import "figures/metrics_table.typ": metrics-table
#import "figures/master_table.typ": master-table
#import "figures/robustness_kafka.typ": robustness-kafka
#import "figures/rel_rmse.typ": relrmse-table
#import "figures/var_selection.typ": encoder-table, decoder-table, static-table, removal-table

#show: presentation
#set par(leading: 1.2em)

#title-slide[
  #text(fill: white, size: 1.3em, weight: "bold")[Macro Forecasting with the TFT and Fed Speeches]

  // #text(fill: white, size: 0.8em)[
  //   Advanced Machine Learning FS2026
  // ]
  #text(fill: white, size: 0.8em, weight: "semibold")[
    #v(0.5em)
     Minna Heim, Chris Traill and Anna Zeitz
  ]


  // #speaker-notes[
  //   My name is Minna Heim Bla bla
  // ]
]

#show: content-slides

#slide[
  = Motivation
   #only("1")[
  - Macro forecasts matter for
    - Governments
    - Financial markets
    - Central banks
   ]
  #only("2-")[
  - Macro forecasts matter for
    - Governments
    - Financial markets
    - *Central banks*
   ]
  #only("3-")[
    - The Federal Reserve (Fed) has a *dual mandate*:
      + Price stability
      + Maximum employment
  ]
  #only("4-")[
    - Accurate forecasts of inflation, unemployment, and GDP are crucial for interest rate decisions
  ]
  // #only("5-")[
  //   $=>$ *Can we do better than standard macro models?*
  // ]
]

// either this or the above
#slide[
  #v(1fr)
  #align(center)[= Can we do better than standard Macro Models?]
  #v(1fr)
]

#slide[
  = Research Question
  #only("1-")[
  - The Fed communicates with the public through official statements and speeches
  ]
  #only("2-")[
  - It may communicate additional information:
    + *Superior information*: private data and internal models not available to the public
    + *Forward guidance*: signals about future policy to guide market expectations
  ]
  #only("3-")[
    #v(1em)
    $=>$ *Do Fed speeches contain information useful to forecast macroeconomic indicators?*
  ]
]

#slide[
  = Data Alignment
#align(center)[
  #v(2em)
  #box(width: 90%)[
    #import "figures/data_timeline.typ": data-timeline
    #only("1")[#data-timeline(step: 1)]
    #only("2")[#data-timeline(step: 2)]
    #only("3")[#data-timeline(step: 3)]
    #only("4")[#data-timeline(step: 4)]
    #only("5-")[
      #figure(
        data-timeline(step: 5),
        caption: [Data alignment pipeline.],
      )
    ]
  ]
]
#only("5-")[
  Speeches are taken from #cite(<CampiglioDeyrisRomelliScalisi2023>, form: "prose") over period $1986-2023$.
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

  #set text(size: 1.2em)
  #v(1fr)
  #grid(
    columns: (10em, 1fr),
    column-gutter: 1.2em,
    row-gutter: 0.8em,
    align: (left + horizon, left + horizon),

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
  #v(1fr)
]

// more high res

#slide[
  = Temporal Fusion Transformer (TFT)
  // Add Abbildung from TFT paper here maybe?; then explain it, also saying what variables we use for the static covariates etc; in appendix, list what the actual variables are?
  // if I use AR(IMA) equation, should i use TFT equations too?
    #place(center)[
    #v(1fr)
    #figure(
    image("figures/tft-arch.jpg", height: 85%),
    caption: [Model Architecture of the TFT @TFT]
    )
    #v(1fr)
  ]
  #speaker-notes[
    based off Transformer, 
    not only past inputs, but static metadata, known future inputs (any additional information that helps forecasts)
    -> static enrichment?
    -> Masked Interpretable MHA?
    - Also great: gives out quantile loss, so basically, uncertainty bounds
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
  + *FinBERT*: BERT-based, pre-trained on financial communications @yang2020finbertpretrainedlanguagemodel
  + *FOMC-RoBERTa*: fine-tuned on FOMC data, hawkish/dovish classification @fomc_roberta_paper
  #only("2-")[
    #v(0.5em)
    *Processing*: chunk-and-average (512-token windows, 128-token overlap)
    - Assumption: information spread across entire speech, not just the beginning
    #v(0.5em)
    *Dimensionality reduction*: 768 $=>$ $X$ dims via Principal Component Analysis or Factor Analysis
    - $X$ is treated as hyperparameter
  ]
]

#slide[
  = Speech Alignment

  Speeches are held at *irregular* and *daily* frequency: how to align with monthly horizon?
  #v(0.5em)

  #only("2-")[
  + *Mean aggregation*: unweighted mean over rolling window of $X$ months
  ]
  #only("3-")[
  + *Exponential decay*: $w_t = exp(-lambda dot d_t)$ where $d_t$ = days before month $m$
    - More recent speeches receive higher weight
    - $lambda$ treated as fixed; speech window $X$ tuned as hyperparameter
  ]
  #only("4-")[
  + *Attention-based*: key = embedding (after dimensionality reduction), query = mean macro state
    - Given current macro conditions, which past speeches are most informative?
    - Fitted on training data only $=>$ no leakage!]
  #v(0.5em)
  #only("5-")[
  All methods also track: voter speech ratio, chair speech ratio, gender share]

]


#slide[
  = Hyperparameter Tuning
  Tune in two stages:
  #only("2-")[
  + *Stage 1: Architecture (Macro-Only TFT)*
    - Bayesian search via Optuna (TPE sampler)
    - 2-fold CV, up to 50 trials
    - Includes: encoder length, hidden size, normalizer
  ]
  #only("3-")[
  + *Stage 2: Speech Embedding Params*
    - Architecture fixed from Stage 1
    - Tune: aggregation, reduction, PCA dims, speech window
]
]



#slide[
  = Evaluation Protocol
  #speaker-notes[
  Point forecasts (MAE and RMSE) use the median quantile (q=0.5) from the quantile loss output.
]

  - *Walk-forward cross-validation*: expanding window, never use future data
  #only("2-")[
  - *Metric*: MAE (mean absolute error), RMSE, and relative RMSE]
  #only("3-")[
  - *Multi-step forecasting*: predictions in non-overlapping 12-month windows
    - Context: training data + model's own previous predictions
    - Same assumption for AR, ARIMA and TFT
  ]
  #only("4-")[
    #v(0.5em)
    $=>$ All models evaluated on the same information set
  ]
]

#slide[
  #v(1fr)
  #align(center)[= Results
   == Holdout Evaluation]
  #v(1fr)
]

#slide[
  = Holdout Metrics: All Models & Horizons
  #speaker-notes[
    AR / ARIMA: univariate baselines.
    TFT: macro-only, tuned architecture.
    TFT+Emb: best per-target embedding from HP tuning.
    Bold = lowest value per metric and (h, target).
  ]
  #v(0.4em)
#figure(
  master-table(),
  caption: [Main Results. Bold = lowest value per metric and (horizon, target).],
  kind: table,
)
]

#slide[
  = Answering Our Research Question
#figure(
    relrmse-table(),
  caption: [Relative RMSE. Bold if TFT with Embeddings better than TFT, per (horizon, target).],
  kind: table,
)
]



#slide[
  // vereinheitlichen die legends??
  = Holdout Predictions ($h$=12)
  #speaker-notes[
    step 1: macro-only TFT vs baselines (final_holdout run)
    step 2: TFT with best per-target embedding (final_emb run)
  ]
  #only("1")[
    #place(center)[
      #v(1fr)
      #figure(
        image("/out/holdout/final_holdout/predictions_vs_actuals_h12_macro.png", height: 88%),
        caption: [Holdout Predictions $h$=12, Macro Only]
      )
      #v(1fr)
    ]
  ]

  #only("2")[
    #place(center)[
       #v(1fr)
      #figure(
      image("/out/holdout/final_emb/predictions_vs_actuals_h12_auto.png", height: 88%),
      caption: [Holdout Predictions $h$=12, Including Embeddings])
       #v(1fr)
    ]
  ]
]


#slide[
  = Robustness Checks
  - Using unrelated text (German Texts of Franz Kafka)
  - Use only first 512 tokens of speech
  $=>$ *Mixed results*: e.g., comparable / better than TFT + Emb. at CPI, h = 3 but worse at CPI, h = 6
  $=>$ More details: Table 9 in Appendix
]

#slide[
  = Future Work
  - Hyperparameter tuning: switch to one stage tuning
    - Context-dependent attention
    - Only FOMC speeches
  - Explore New Methods:
    - Try out SSMs for forecasting
    - Combine with TFT: replace LSTM with SSM for sequence encoder // better for long-run dependencies, also more targeted for time series data!
  - Add vintages: use real-time macro data (as available at forecast time) for more realistic evaluation
  - Extend dataset: Working Paper @anna_snb_2026
    - 10,000+ speeches since 1914, continuously updated
]

#slide[
  = Conclusion

  - AR and ARIMA dominate CPI and UNRATE
  - TFT great for GDP forecasts
  *$=>$ Speeches improve TFT forecasts at medium horizon length*
  #only("2-")[
  - But: not necessarily due to speech content but due to more stable TFT 
  ]
  #only("3-")[
    *$=>$ Conclusion: we struggle to beat statistical benchmarks*
  ]
]

#slide[
  #set text(size: 0.8em)
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
  = Data Frame:
  // titles of vars arent good yet... still need to fix this
  #place(center)[
    #v(2em)
    #figure(
      image("figures/df_head.png", width: 100%),
      caption: [Head of the main data frame])
  ]
]

#slide[
  = Target Vars
  // Gray bands = NBER recessions; CPI and GDP are log-differenced
  #place(center)[
    #v(0.5em)
    #figure(
    image("figures/target_vars.png", width: 65%),
    caption: [Time series of the target variables])
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
    #figure(
    image("figures/tft-inputs.png", height: 80%),
    caption: [Temporal Fusion Transformer: Input Types])
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

#slide[
  = Hyperparameter Tuning: Stage 1 Search Space
  Architecture tuned on macro-only TFT (per target × horizon):
 #figure(
  table(
    columns: (auto, auto, auto, auto),
    [*Parameter*], [*Type*], [*Range / Values*],[*Nr. of Trials*],
    [Encoder length], [int], [12 – 48],[20],
    [Hidden size], [categorical], [\{8, 16, 32, 64, 128, 256\}],[20],
    [Hidden continuous size], [categorical], [\{2, 4, 8, 16\}], [20],
    [Dropout], [float], [0.05 – 0.55],[20],
    [Learning rate], [log-uniform], [$10^(-4)$ – 0.15],[20],
    [Normalizer], [categorical], [\{encoder-none, group\}],[20],
  ),
  caption: [Stage 1: Hyperparameter search space and number of trials.],
  kind: table,
)
  #v(0.5em)
  Fixed: 
  `lstm_layers = 2`, `attention_head_size = 2`, 
  
  `max_epochs = 50`, `patience = 15`, `batch_size = 128`
]

#slide[
  = Hyperparameter Tuning: Stage 2 Search Space
  Speech embedding params tuned (architecture fixed from Stage 1):

#figure(
  table(
    columns: (auto, auto, auto),
    [*Parameter*], [*Type*], [*Range / Values*],
    [Aggregation], [categorical], [\{mean, decay, attention\}],
    [Reduction], [categorical], [\{pca, fa\}],
    [PCA components ($X$)], [int], [5 – 30],
    [Speech window (months)], [int], [3 – 12],
  ),
  caption: [Stage 2: Hyperparameter search space and number of trials.],
  kind: table,
)
  #v(0.5em)
  Tuned separately for each of 9 combinations: \{CPI, GDP, UNRATE\} $times$ \{3, 6, 12\} months
]


#slide[
  = Hyperparameter Tuning: Stage 2 -- Optimal Hyperparameters
  #block[
    #set text(size: 14pt)
    #set par(leading: 0.3em)
    #figure(
      table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      table.header(
        [], table.cell(colspan: 3)[*FOMC-RoBERTa*], table.cell(colspan: 3)[*FinBERT*],
        [*Target*], [*$h$=3*], [*$h$=6*], [*$h$=12*], [*$h$=3*], [*$h$=6*], [*$h$=12*],
      ),
      [*CPI*],
      [Mean \ PCA, $n$ = 21 \ $W$ = 3 \ MAE = 0.001847],
      [Attention \ FA, $n$ = 10 \ $W$ = 8 \ MAE = 0.001849],
      [*Attention \ PCA, $n$ = 12\ $W$ = 9 \ MAE = 0.001842*],
      [*Attention \ FA, $n$ = 20 \ $W$ = 5 \ MAE = 0.001846*],
      [*Decay \ PCA, $n$ = 23 \ $W$ = 3 \ MAE = 0.001846*],
      [Mean \ PCA, $n$ = 8 \ $W$ = 7 \ MAE = 0.001847],

      [*GDP*],
      [*Mean \ PCA, $n$ = 8 \ $W$ = 7 \ MAE = 0.00154*],
      [Attention \ FA, $n$ = 12 \ $W$ = 8 \ MAE = 0.00171],
      [Attention \ FA, $n$ = 5 \ $W$ = 10 \ MAE = 0.00152],
      [Attention \ FA, $n$ = 6 \ $W$ = 12 \ MAE = 0.00170],
      [*Attention \ FA, $n$ = 10 \ $W$ = 8 \ MAE = 0.00157*],
      [*Decay \ PCA, $n$ = 20 \ $W$ = 12 \ MAE = 0.00151*],

      [*UNRATE*],
      [Decay \ FA, $n$ = 12 \ $W$ = 11 \ MAE = 1.31293],
      [Attention \ FA, $n$ = 6 \ $W$ = 12 \ MAE = 1.31501],
      [*Mean \ PCA, $n$ = 8 \ $W$ = 7 \ MAE = 1.11597*],
      [*Mean \ FA, $n$ = 23 \ $W$ = 5 \ MAE = 1.26152*],
      [*Mean \ PCA, $n$ = 14 \ $W$ = 4 \ MAE = 1.30896*],
      [Mean \ PCA, $n$ = 25 \ $W$ = 4 \ MAE = 1.20808],
    ),
  caption: [Stage 2: Optimal hyperparameters per (target, horizon).],
  kind: table,
)]

  where $$W$$ refers to the speech window size and $$n$$ to the optimal number of components for the dimensionality reduction. The best result per target$times$horizon is shown in bold, and $h$ stands for forecast horizon.

]


#slide[
  // vereinheitlichen die legends??
  = Holdout Predictions ($h$=6)
  #speaker-notes[
    step 1: macro-only TFT vs baselines (final_holdout run)
    step 2: TFT with best per-target embedding (final_emb run)
  ]
  #only("1")[
    #place(center)[
      #v(1fr)
      #figure(
        image("/out/holdout/final_holdout/predictions_vs_actuals_h6_macro.png", height: 88%),
        caption: [Holdout Predictions $h$=6, Macro Only]
      )
      #v(1fr)
    ]
  ]

  #only("2")[
    #place(center)[
       #v(1fr)
      #figure(
      image("/out/holdout/final_emb/predictions_vs_actuals_h6_auto.png", height: 88%),
      caption: [Holdout Predictions $h$=6, Including Embeddings])
       #v(1fr)
    ]
  ]
]

#slide[
  // vereinheitlichen die legends??
  = Holdout Predictions ($h$=3)
  #speaker-notes[
    step 1: macro-only TFT vs baselines (final_holdout run)
    step 2: TFT with best per-target embedding (final_emb run)
  ]
  #only("1")[
    #place(center)[
      #v(1fr)
      #figure(
        image("/out/holdout/final_holdout/predictions_vs_actuals_h3_macro.png", height: 88%),
        caption: [Holdout Predictions $h$=3, Macro Only]
      )
      #v(1fr)
    ]
  ]

  #only("2")[
    #place(center)[
       #v(1fr)
      #figure(
      image("/out/holdout/final_emb/predictions_vs_actuals_h3_auto.png", height: 88%),
      caption: [Holdout Predictions $h$=3, Including Embeddings])
       #v(1fr)
    ]
  ]
]

#slide[

  = Variable Selection
  #only(1)[
    // top 10 encoder (time-varying unknown reals)
    #place(center)[
    #v(2em)
    #encoder-table(n: 10, caption: [Top 10 encoder variables, $h=12$], )
  ]
    
  ]
  #only(2)[
    // top 10 decoder (time-varying known reals)
    #place(center)[
    #v(2em)
    #decoder-table(n: 10, caption: [Top 10 decoder variables, $h=12$], )
  ]

  ]
  #only(3)[
    // top 10 decoder (time-varying known reals)
    #place(center)[
    #v(2em)
    #static-table(n: 10, caption: [Top 10 static variables, $h=12$], )
  ]

  ]
  // TODO: maybe not add this, jsut confusing that CPI is here.
  // #only(4)[
  //   // top 10 decoder (time-varying known reals)
  //   #place(center)[
  //   #v(2em)
  //   #removal-table(cutoff: 0.005, caption: [Potential Removal Candidates, $h=12$], )
  // ]
  ]
  #slide[
  = Holdout Metrics: Robustness
  #set text(size: 0.8em)
#figure(
  robustness-kafka(),
  caption: [Robustness: Kafka and 512-token embeddings. Bold = lowest value per metric and (horizon, target).],
  kind: table,
)
]