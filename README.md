# ML Market Prediction Project

## Overview
This project applies two machine learning models, logistic regression and the perceptron, to predict whether NASDAQ (NQ) futures price will rise within the next 5 minutes. Features are based on ICT trading concepts such as SMT divergences, fair value gaps, and liquidity sweeps.

## Algorithms
- Logistic Regression
- Perceptron)

## Dataset
- ~10,000+ 5-minute samples from Jan 1 to May 1, 2024
- Features extracted from correlated ETFs (QQQ, SPY, DIA)
- Labels defined as: 1 = price went up in next 5 min, 0 = it did not

## Folder Structure
- `/images/`: Visuals used in report
- `/data/`: Contains CSVs
- `logistic_regression.py`: NumPy-based logistic regression model
- `perceptron.py`: Perceptron model code
- `ict_dataset.csv`: Preprocessed dataset

## Acknowledgments
- Parts of this project are based on course materials and homework from UNC Charlotte.
- ChatGPT was used for API connection and plotting.