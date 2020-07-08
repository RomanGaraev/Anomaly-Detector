# Anomaly-Detector - simple malicious traffic analyzer 

Dataset: https://www.unb.ca/cic/datasets/ids-2018.html 

AWS commands: 
-ls: aws s3 ls --no-sign-request "s3://cse-cic-ids2018" --recursive --human-readable --summarize
-download: aws s3 sync --no-sign-request --region eu-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" D:/

The main idea of detection is using recurrent neural networks (`LSTM` or `GRU` classes) for sequences of network flows. Speed requierments don't let using RNNs purerly, so you should unite them with some fast and simple filter (`NeuralNetwork` class), or you should reduce the dimension of flows (`Autoencoder` class). This repositiry contains only single ML models; it's not a complete anomaly detection system. You can build a concrete system in the main.py file, but for the traffic capture and representation you need a sniffer like [this](https://www.unb.ca/cic/research/applications.html).

The concept of such a cascade is described [here](https://ieeexplore.ieee.org/document/8283694)
