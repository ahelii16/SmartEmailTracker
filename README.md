# ⚡️SmartEmailTracker

## Topic : AI/ Machine Learning for Decision Making (Project 2)

The current project has lots of content in emails post which the ops user has to perform some manual operation like keeping track of processed transactions, failed and pending transactions, status update requests etc. Using artificial intelligence and machine learning decision making matrix we could automate the processes. The system aims do the following:

- Reading the email content with certain subject line and body: The emails can be directly typed into the UI or uploaded as a PDF, TXT or image file. They are then treated as an email, the content is extracted out of it and the system is trained to build its decision making capability.

- The system should continuously improve the model’s decision making skill by monitoring all the emails with the defined subject line. To simulate this use case, create a folder in any server and have a background listener continuously monitoring the folder, any new file comes at any moment the system should read its content to train the model.

- Based on the event type received in the email the system will take the desired actions, i.e. classifying the emails as per the target classes.

## For Checkpoint 1:

We have used BiLSTM model to classify incoming emails as 'Complete', 'Failed', 'Pending', 'Request', 'Processing', 'General', in regards to payment data.

## For Checkpoint 2:

BiLSTM model, though highly accurate, is too robust for a small dataset like the one we are currently using. So we have switched to using XGBoost algorithm for making the predictions. The classifier when used with pre-trained Glove embeddings, gives accurate results (close to 100% accuracy, less overfitting) even with a smaller dataset. 


## 🛠 Installation and quickstart:

Clone the repository in your terminal:
```sh
git clone https://github.com/Ahelii16/SmartEmailTracker.git
```
Go the project repository:
```sh
cd SmartEmailTracker
```
Create a virtual environment for this project using the following command:
```sh
virtualenv venv
```
Activate Your Virtual Environment
```sh
venv/bin/activate
```
Project installations can be done with `pip`:
```sh
pip install -r requirements.txt
```
Install Glove embeddings (1 GB file) from: 
```sh
https://www.kaggle.com/thanakomsn/glove6b300dtxt
```
Change the current directory to Merged UI_Listener:
```sh
cd Merged UI_Listener
```
Run the main file:
```sh
python start.py
```
Open up localhost at http://127.0.0.1:5000/ for live demo of the app.

Email input can given from the form or uploaded as a PDF, text file or an image.
 - NOTE: If any file is found not to be showing, please write the absolute path for that file.

## Sample email format assumed for PDF or txt file:

To: Rahul@CitiBankPune.com 

From: Mike@BNYMellon.com 

Subject: Transaction 608234 Complete 

Hi,
Hope you are well.

Wanted to inform you that transaction has been completed successfully.
Thanks for your assistance!

Mike
