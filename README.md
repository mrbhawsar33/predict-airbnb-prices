# predict-airbnb-prices
As a part of the Data Science team at StayWise, a global vacation rental platform. The company aims to enhance its pricing system to help hosts set competitive rates. All listing and booking data are stored in AWS S3, and the data science workflow uses MLflow for experiment tracking and model management.


predict-airbnb-prices
├── data/               # Local folder for small samples or metadata
├── notebooks/      
│   ├── eda.ipynb    
├── src/                # All your Python source code (.py files)
│   ├── ingest.py       # Task 1: S3 access logic
│   ├── preprocess.py   # Task 2: Cleaning logic
│   └── train.py        # Task 3-4: Modeling and MLflow
├── mlruns/             # Task 5: Created automatically by MLflow
├── requirements.txt    # List of libraries (s3fs, pandas, mlflow, etc.)
└── main.py             # The "entry point" to run the whole pipeline