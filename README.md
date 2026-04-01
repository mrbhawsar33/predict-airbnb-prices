# predict-airbnb-prices
An end-to-end Machine Learning pipeline to predict nightly listing prices using XGBoost, Random Forest, and Ridge Regression, tracked via MLflow.

# Project Architecture
* Ingestion: Pulls raw 2019 NYC Airbnb data from AWS S3.

* Preprocessing: Handles log-transformation of price, target encoding for 221 neighborhoods, and robust scaling.

* Experiment Tracking: Uses MLflow to log parameters, metrics (RMSE, MAE, R², MAPE), and model artifacts.

# Project structure
```
predict-airbnb-prices
├── notebooks/      
│   ├── eda.ipynb    
├── screenshots/        # Contains screenshots of important project checkpoints
│   ├── mlflow/     
├── src/                # All Python source code (.py files)
│   ├── ingest.py       
│   ├── preprocess.py   
│   └── train.py      
└── main.py             # start point of experiment pipeline  
├── mlruns/             # generated for MLflow
├── requirements.txt    
└── README.md
└── setup.py     

```

# How to Run
## 1. Setup Environment (external depenedencies + ):

```
pip install -r requirements.txt
pip install -e .
```

## 2. Execute Pipeline:
```
python main.py
```

## 3. View Dashboard:
```
mlflow ui
```