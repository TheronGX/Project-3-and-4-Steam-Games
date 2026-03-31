This the README for games.sv

Project Overview
This project analyzes a dataset of Steam games to predict the number of negative reviews a game will receive based on its available metadata.
Steam is a major digital distribution platform for PC games, and understanding factors that influence negative reviews can provide insight into game reception and quality expectations.
The dataset used contains information on games released on Steam up to 2019, including features such as genre, price, playtime, developer, publisher, and user engagement statistics.

Dataset Description
The original dataset contains approximately 27,000 games from Steam. 
For computational efficiency and focus, this project uses a filtered subset containing 901 games released in 2019.
The dataset can be changed to the entire 27,000 games if you download the dataset from https://www.kaggle.com/datasets/nikdavis/steam-store-games , 
and in the code change "steam_test.csv" to should be "steam.csv", if you do this expect to have to a wait when running the code with all 27,000 games.

Goal
The objective is to build predictive models that estimate the number of negative reviews a game will receive based on its features.
This can help:
Understand factors influencing poor reception
Identify risky game attributes before release
Compare effectiveness of different machine learning models

Data Preprocessing
The dataset was cleaned and transformed prior to modeling:
Removal of unnecessary columns
Encoding categorical variables (genres, categories, tags)
Normalization/standardization of features where required
Handling missing or inconsistent data
Feature selection to reduce dimensionality and improve efficiency

Models Used
1. Linear Regression (Normal Equation)
Baseline model
Limited in capturing nonlinear relationships
2. k-Nearest Neighbors (kNN)
Distance-based model
Sensitive to feature scaling and local structure
3. Random Forest
Ensemble tree-based model
Captures nonlinear interactions between features

How to run
Make sure to have the dataset either the 2019 one that I included or the original one that you can get from that link.
Make sure you have numpy and pandas downloaded.
Then you should be able to just run the code and it should make 3 figures and the rest of the data will be printed out.