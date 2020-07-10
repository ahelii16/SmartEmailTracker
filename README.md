# ⚡️AI for Decision Making: SmartEmailTracker

## An email management tool for task-based classification of incoming email stream

The current project has lots of content in emails post which the ops user has to perform some manual operation for keeping track of them. Using artificial intelligence and machine learning decision making matrix we could automate the processes. The system aims do the following:

- Reading the email content with certain subject line and body: The emails can be directly <b>typed into the UI</b> or <b>uploaded as a PDF, TXT, handwritten or typed image</b> file. They are then treated as an email, the content is extracted out of it and the system is trained to build its decision making capability.

- Based on the event type received in the email, the system will take the desired actions, i.e. classifying the emails as per the target classes. The repository currently handles transaction related emails, belonging to the classes <b>Complete, Processing, Pending, Request (for status updates), CreditCard, General and Failed</b>.

- The system would continuously improve the model’s decision making skill by monitoring all the emails with the defined subject line and body. To simulate this use case, a folder is created in the server and a background listener continuously monitors the folder; when a new file comes at any moment, the system would read its content to train the model. Users may add new classes as per requirements, and once there are at least 40 emails of the new class, the ML model can be retrained(click the retrain button on UI) to incorporate these new classes.

- Once the emails have been entered into the Database(can be viewed from Show All button), the email thread pertaining to a particular transaction ID can be viewed from the Email Thread button on top.

## Technology Used
- For ML model: Python, Keras, Tensorflow, Scikit Learn, Pandas, Numpy, Spacy, Gensim Word2Vec model, pre-trained Glove embeddings, Matplotlib
- For app development: HTML, CSS, Bootstrap, Flask, SQLAlchemy, Tesseract, PDFMiner

<p align="center">
  For details on all the ML models tried and tested, model comparisons, UI features and better understanding of system workflow, please refer to <a href="https://docs.google.com/document/d/1qj_gYU47MPSgorbo-ho6osTHv2lS3V2bhjiXhigURFA/edit?usp=sharing">this document</a>.
</p>

### Future scope:

In this project, we have used the <b>Few-Shot Learning</b> approach with <b>XGBoost</b> algorithm. The classifier gives accurate results with less overfitting even with a smaller dataset. BiLSTM model is too robust for a small dataset like the one we are currently using and is highly overfitting. So we have switched to using XGBoost algorithm for making the predictions. A deep learning approach with <b>BiLSTM</b> is a viable alternative for extracting features and giving highly accurate results from large real world datasets. 


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
Install glove.6B.300d.txt (1 GB file) from: 
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
 
Emails can also be created from the command line in inputEmails directory. (in Merged UI_listener). For this the listener will have to be started first. 
Command: python listener_xg.py

 - NOTE: If any file is found not to be showing, please write the absolute path for that file.
 - Abbreviations, industry specific keywords and definitions can be specificied as per need in wordfile.py (Merged UI_Listener directory).

## Sample email format assumed:

To: Rahul@CitiBankPune.com 

From: Mike@BNYMellon.com 

Subject: Transaction 608234 Complete 

Hi,
Hope you are well.

Wanted to inform you that transaction has been completed successfully.
Thanks for your assistance!

Mike
