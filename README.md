# Piper

Piper is a MLOps tool for facilitating the training and testing phase of ML development. It uses a DAG to chain together custom workflows from data wrangling and preprocessing through training and validation to generating artifacts (trained models). Piper ensures reproducibility by storing execution metadata in a data storage (e.g. SQLite database).