#import "template.typ": *

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


  #speaker-notes[
    My name is Minna Heim Bla bla
  ]
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

// #slide[
//   = Our Approach

//   - Statistical Benchmarks:  
//       - AR(p)
//       - ARIMA(p,d,q)

//   - Temporal Fusion Transformer (Macro Only)
//   - Temporal Fusion Transformer (with Fed Speeches)
// ]

//#slide[
//  = Data
//  *3 types of Data:*
// this is just an example :)
  // #only("2-") shows this only from slide 2 onwards
//  #only("2-")[
//   - Macro Data
//        - Quarterly, Monthly, Daily
//        - E.g.: CPI, UNRATE, GDP, Exchange Rates, etc. 
//  ]
//  #only("3-")[
//    - Meta Data 
//        - Macro Metadata (Popularity, Frequency, Units)
//        - FOMC Dissent 
//        - Days till next FOMC Meeting, etc.
//  ]
//  #only("4-")[
//    - Embedded Fed Speeches
//        - E.g. _"Remarks by Mr Roger W Ferguson Jr, Vice-Chairman of the Board of Governors of the US Federal Reserve System, before the National Economic Association in Boston on 7 January 2000."_
//        - Embedded (via FOMC-RoBERTa, FinBERT and Ollama,)
//        - Dimensionality Reduction (None, PCA, Factor Analysis)
//        - Speech Aggregation (mean, exp. decay, (context)attention-based)
//
//  ]
//]

#slide[

  = Show Data...

  by plotting, e.g.

CPI, UNRATE, GDP plots (stationarity/non-stationarity) => maybe just one plot with the log differenced?

Mention log-differencing for GDP/CPI

Timeline (Zeitstrahl) could live here or be its own slide
  
]

#slide[

  = Show Data 2


Timeline (Zeitstrahl) could live here or be its own slide
  
]

#slide[
  = Models
  - Statistical Benchmarks (AR, ARIMA)
  - Temporal Fusion Transformer @TFT

]

#slide[
  = Temporal Fusion Transformers
  Add Abbildung from TFT paper here maybe?; then explain it, also saying what variables we use for the static covariates etc; in appendix, list what the actual variables are?

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
  #only("3-")[
  #v(0.5em)
  *Robustness*: replace embeddings with Kafka texts $=>$ should add no predictive signal
  ]
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
  = Hyperparameter Tuning
  Tune in two stages:
  + *Stage 1: Architecture (Macro-Only TFT)*
    - Bayesian search via Optuna (TPE sampler)
    - 2-fold CV, up to 50 trials
    - Includes: encoder length, hidden size, normalizer
    // MOVE THIS TO APPENDIX
    // - Search space:
      // - Encoder length: 12–48
      // - Hidden size: \{8, 16, 32, 64, 128, 256\}
      // - Dropout: 0.05–0.55
      // - Learning rate: $10^(-4)$–$0.15$ (log scale)
      // - Normalizer: \{encoder-none, group\}
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
  Tuned separately for each of 9 combinations: \{CPI, GDP, UNRATE\} × \{3, 6, 12\} months
]
