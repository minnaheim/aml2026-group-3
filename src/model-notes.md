## Notes on TFT Model to use:


**To Dos**: 
- remove lagged variables if trying to predict variable of interest!! 
- run all for new variation -> long form data, what happens?


### Keeping Score of what I did:

1. Added the 4 monthly macro variables to the tft (good)
2. Added metadata from fredR package of the target variable to the model (better)
3. Added the metadata of all macro variables to the model (worse!)
4. Scaled data (to compare model performance), changed training dataset
    a. pivoted to long 
    b. changed `series_id` from macro to actual var name -> constant var no info
    c. included metadata of each variable aka one: `meta_unit` (changes w/ var), and one `meta_popularity`



**Extensions for TFT**:
<!-- done: - to get more static covariates (because apparently model performs best with max. metadata), scrape information from [FRED website](https://fred.stlouisfed.org/series/PAYEMS)? -->
- Add weights to meetings, depending on when the meetings occur w.r.t the FOMC meeting 
- include the blackout period



**Final Choice**: [Temporal Fusion Transformer in Pytorch](https://pytorch-forecasting.readthedocs.io/en/v1.0.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html)

Examples:

- [Pytorch Example Demand Forecasting](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html)
- [Kaggle Demand Planning Implementation](https://www.kaggle.com/code/tomwarrens/temporal-fusion-transformer-in-pytorch)



<!-- I have found multiple implementations of the TFT paper on huggingface & catalyzex:

**Catalyzex**:

- https://www.catalyzex.com/paper/temporal-fusion-transformers-for/code

**Original Repo on Github**:

- https://github.com/google-research/google-research/tree/master/tft


**Paper on Hugging Face**:

- https://huggingface.co/papers/1912.09363

**Models belonging to this TFT paper**:

- https://huggingface.co/keras-io/structured-data-classification-grn-vsn
- https://huggingface.co/lwaekfjlk/Time-Series-Library

**Insights from study done with TFTs**:
-> he TFT is based off a
neural language processing mode -->