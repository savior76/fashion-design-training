# fashion-design-training
fashion design training was done on 1 epoch natively on rtx 3050 laptop via CUDNN, if increase the epochs the accuracy would be better

 Below is a suggested README.md—structured according to community best practices and focused on explaining how the code works (results are omitted). Citations follow each sentence.

 Summary: This repository provides an end‑to‑end Python pipeline for processing the Fashion Product Images Dataset, from data ingestion and EDA through transfer‑learning model training to final inference 

.

# Project Overview
This project implements a multi‑output classification pipeline that predicts four attributes—base colour, article type, season, and gender—from fashion product images . It uses Keras’s ImageDataGenerator for on‑the‑fly augmentation and a ResNet50 backbone (pretrained on ImageNet) to extract deep features, followed by custom dense heads for each target.

# Key Features
Data Loading & Cleaning: Reads styles.csv, skips malformed rows, and ensures all image IDs are strings ending in .jpg 

.

# Exploratory Data Analysis: Generates bar plots for each attribute (article type, base colour, season, gender) using Matplotlib 
DataDrivenInvestor.

# Data Augmentation: Applies rescaling, horizontal flips, small rotations, and zooms via ImageDataGenerator with a 20% validation split 
Python Tutorials – Real Python.

# Transfer Learning: Freezes ResNet50 layers, adds global average pooling, dropout, and separate softmax heads for each target attribute 
Python Tutorials – Real Python

.

# Sparse‑Categorical Training: Encodes labels to integer codes, uses sparse_categorical_crossentropy, and saves a JSON mapping codes→labels for inference 


.

# Inference Utility: Provides predict_image() which reloads the model and class‑indices JSON to output human‑readable predictions for any input image 


# Directory Structure

.
├── fashion_product_pipeline.py   # Full pipeline script
├── class_indices.json            # Saved mapping of codes → labels
├── fashion_model.h5              # Best‑model checkpoint
├── styles.csv                    # Original metadata CSV
└── images/                       # Folder of product images
This layout follows a simple Python‑project convention where scripts, data artifacts, and resources are colocated under version control 

.

# Prerequisites
Python 3.8+

TensorFlow 2.x

pandas, numpy, matplotlib

A GPU is recommended for training but not required
These dependencies ensure reproducibility and leverage common scientific‑Python tooling 
