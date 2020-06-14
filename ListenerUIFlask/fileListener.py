
# coding: utf-8

# # Directory Watcher + SQLite

# This code watches a given directory inputEmails. Whenever a new email file is created, it loads the trained model to predict the corresponding output class. Email is then moved to the output class folder and this entry is added to database of processed emails (implemented using SQLite).
#
# Files required in same directory are
# - glove.6B.50d.txt
# - model.json
# - best_model.h5
# - inputEmails folder (where new email files will be created)
#
# Email Format assumed:
#
# To: Rahul@CitiBankPune.com <br>
# From: Mike@BNYMellon.com <br>
# Subject: Transaction 608234 Complete <br>
# Hi,
# Hope you are well.
# Wanted to inform you that transaction has been completed successfully.
# Thanks for your assistance!
# Mike
#

#  ## Initial Setup

# In[10]:


import numpy as np
#import pandas as pd
import re
#import csv
# NLP
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer

from keras.layers import *
from keras.models import Sequential
from keras.models import model_from_json

import sys
import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


# ## Cleaning and Vectorization of Email Text - Subject + Body

# In[11]:


def clean(text):
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","from", "gbp", "usd", "eur", "cad", "sent","thanks", "acc", "ID", "account", "regards", "hi", "hello", "thank you"))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    porter= PorterStemmer()

    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    stem = " ".join(porter.stem(token) for token in normalized.split())

    return normalized

embeddings = {}
with open('./glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')

        embeddings[word] = coeffs
    f.close()
print(len(embeddings))

def getOutputEmbeddings(X):
    X = X.split()
    embedding_matrix_output = np.zeros((1,100,50))
    for jx in range(min(100, len(X))):
        #ignore invalid words
        if (X[jx].lower() in embeddings):
            embedding_matrix_output[0][jx] = embeddings[X[jx].lower()]
    return embedding_matrix_output


# ## Class Labels used for Model Training

# In[12]:


#dependent on model loaded
classes = ['BankFailed', 'BankProgress', 'BankComplete', 'BankRequest',
       'ClientProgress', 'ClientStatus', 'ClientComplete', 'ClientFailed']


# ## Create ProcessedEmails table in SQLite DB

# In[13]:

import sqlite3
'''

conn = sqlite3.connect('processed.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS ProcessedEmails')

cur.execute('CREATE TABLE IF NOT EXISTS "ProcessedEmails" ( "Seq_ID" INTEGER NOT NULL,"To" TEXT,"From" TEXT,"Received_on" TEXT,"Subject" TEXT, "Body" TEXT,"Predicted_Class" TEXT,PRIMARY KEY("Seq_ID" AUTOINCREMENT))')

cur.close()
'''

# ## Email Processing

# In[15]:


def HandleNewEmail(mail_path):
    #Parse email to store in DB
    email=open(mail_path, "r")
    text = ""
    sub = ""
    body = ""
    for line in email:
        print(f"FromFile===>{line}")
        if line.startswith('To: ') or line.startswith('to: '):
            pieces = line.split()
            to_add = pieces[1]
            continue
        if line.startswith('From: ') or line.startswith('from: '):
            pieces=line.split()
            from_add=pieces[1]
            continue
        if line.startswith('Subject: ') or line.startswith('subject: '):
            pieces=line.split()
            subject=pieces[1:]#remove word subject
            for word in subject:
                sub = sub + " " + word
           # text=text+" "
            continue
        body=body + line
    text=sub + " " + body
    receivedDate = time.ctime(os.path.getctime(mail_path))

    print(f"Email  --> {text}")
    email.close()
    #clean email
    clean_text = clean(text)
    print(f"Cleaned email --> {clean_text} \n")
    emb_X = getOutputEmbeddings(clean_text)

    #load model
    with open("model.json", "r") as file:
        model=model_from_json(file.read())
    model.load_weights("best_model.h5")
    #model.summary()
    p = model.predict_classes(emb_X)
    #print (p.shape)
    print(f'Output --> class {classes[p[0]]} \n');

    addtoDB(to_add, from_add, receivedDate, sub, body, classes[p[0]])
    moveEmail(mail_path, classes[p[0]])

def addtoDB(to_add, from_add, receivedDate, sub, body, outputclass):
    #add email and class to DB
    conn = sqlite3.connect('mails.sqlite3')
    cur = conn.cursor()
    cur.execute('''INSERT INTO mails (mto, mfrom, mdate, msubject, mbody, mclass)
               VALUES (?, ?, ?, ?, ?, ?)''', (to_add, from_add, receivedDate, sub, body, outputclass))
    print("Inserted in DB\n")
    conn.commit()
    cur.close()

def moveEmail(mail_path, outputdir):
    #Check if output class directory exists, if not, create it
    CHECK_FOLDER = os.path.isdir(outputdir)
    if not CHECK_FOLDER:
        os.makedirs(outputdir)
        print("created folder : ", outputdir)
    #move email to class output directory
    shutil.move(mail_path, outputdir)
    print("moved to folder : ", outputdir)




# In[16]:


#addtoDB("a@a.com", "b@b.com", "date", "hi complete help", "ClientProgress")


# ## Directory Watcher

# In[17]:


def on_created(event):
    print(f"New email {event.src_path} received!")
    HandleNewEmail(event.src_path)

if __name__ == "__main__":
    patterns = "*"
    ignore_patterns = ""
    ignore_directories = False
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
    #new emails will be created in inputEmails directory
    path = "inputEmails"
    #path = sys.argv[1] if len(sys.argv) > 1 else 'inputEmails'
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    print('====> Observer Started')
    try:
        while True:
             time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        print('====> Observer Stopped')
        my_observer.join()


# In[18]:


#HandleNewEmail("inputEmails/email.txt")


# In[ ]:


#addtoDB("a@a.com", "b@b.com", "date", "hi complete help", "ClientProgress")


# In[20]:


#Interrupt Kernel to stop watcher before running this

#display all emails in DB
'''
conn = sqlite3.connect('processed.sqlite')
cur = conn.cursor()
# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT * FROM ProcessedEmails'

for row in cur.execute(sqlstr):
    print(row)

cur.close()
'''
