## Notes on TFT Model to use:


**To Dos**: 

- now: 3 monthly series to predict one (removed others due to length, and shortened the 3 to overlap)
- add different y, predict on all 3 vars
- create training loop, which tries to predict each macro variable based on the others




**Extensions for TFT**:
- to get more static covariates (because apparently model performs best with max. metadata), scrape information from [FRED website](https://fred.stlouisfed.org/series/PAYEMS)?
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