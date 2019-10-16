# Anomaly-Detector - simple malicious traffic analyzer 

Dataset: https://www.unb.ca/cic/datasets/ids-2018.html 

AWS commands: 
-ls: aws s3 ls --no-sign-request "s3://cse-cic-ids2018" --recursive --human-readable --summarize
-download: aws s3 sync --no-sign-request --region eu-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" D:/
