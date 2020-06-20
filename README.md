# SmartEmailTracker

## Topic : AI/ Machine Learning for Decision Making (Project 2)

The current project has lots of content in emails post which the ops user has to perform some manual operation like collateral postings, collateral booking, dispute discrepancy etc. Using artificial intelligence and machine learning decision making matrix we could automate the processes. The system aims do the following:

- Reading the email content with certain subject line. Set of files treated as an email and extract the contact out of it and train the system to build its decision making capability.

- The system should continuously improve the model’s decision making skill by monitoring all the emails with the defined subject line. To simulate this use case, create a folder in any server and have a background listener continuously monitoring the folder, any new file comes at any moment the system should read its content to train the model.

- Based on the event type received in the email the system will take the desired actions like booking a collateral or resolve the dispute etc.

# For Checkpoint 1:

We have used biLSTM model to classify incoming emails as 'Complete', 'Failed', 'Pending', 'Request', 'Processing', 'General', in regards to payment data.

## How to run:

Go to directory ListenerUIFlask

pip install flask
pip install flask-sqlalchemy

run listener.py (command: python listener.py)
run mailworking.py (command: python mailworking.py)
UI will be rendered on localhost:5000
So, both listener.py and mailworking.py have to be run together.
Input can given from HTML form or directly as a txt file in inputEmails directory

## Files required in same directory are:

- runbiLSTM.py
- glove.6B.300d.txt (can be downloaded from Kaggle)
- model_glove.json
- model_glove.h5
- emaildataset.csv
- inputEmails directory
- static
- templates

## Input Format for Live-User-Input and class prediction:

- Files needed from github: "UniformUI-retrain"
- software: 1- pycharm or sublime text editor
- Libraries- pip install "(missing library-name)"

## type command:
```
export FLASK_APP=start.py
export FLASK_ENV=development
flask run
```

- If directory not found showing, at those place-write absolute path that file in your laptop
- open "User-Interface" file from master(gitHub) and run the above commands

# Email Format assumed:

To: Rahul@CitiBankPune.com 
From: Mike@BNYMellon.com 
Subject: Transaction 608234 Complete 
Hi,
Hope you are well.
Wanted to inform you that transaction has been completed successfully.
Thanks for your assistance!
Mike
