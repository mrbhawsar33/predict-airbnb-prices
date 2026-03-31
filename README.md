# predict-airbnb-prices
'StayWise'(a fictional vacation rental platform) wants to predict the optimal nightly price for new listings based on factors such as location, amenities, and reviews.
All listing and booking data are stored in AWS S3, and the data science workflow uses MLflow for experiment tracking and model management.

# Project structure
```
predict-airbnb-prices
├── data/              
├── notebooks/      
│   ├── eda.ipynb    
├── src/                # All Python source code (.py files)
│   ├── ingest.py       
│   ├── preprocess.py   
│   └── train.py        
├── mlruns/             # for MLflow
├── requirements.txt    
└── main.py             
```