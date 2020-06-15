# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 00:04:39 2020

@author: divya
"""


# coding: utf-8

# # Directory Watcher + SQLite

# This code watches a given directory inputEmails. Whenever a new email file is created, it loads the trained model to predict the corresponding output class. Email is then moved to the output class folder and this entry is added to database of processed emails (implemented using SQLite).
#
# Files required in same directory are
# - glove.6B.50d.txt
# - model_glove.json
# - model_glove.h5
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

import numpy as np
#import pandas as pd
import re
from runbiLSTM import inputfunc
#import csv
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import sqlite3


import sys
import os
import shutil
import time


import pandas as pd

def HandleNewEmail(mail_path):
    #Parse email to store in DB
    email=open(mail_path, "r")
    to_add=''
    from_add=''
    sub=''
    body=''
    for line in email:
        #print(f"FromFile===>{line}")
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

    email.close()
    #print(encoded_mail)

    outputclass  = inputfunc(to_add,from_add,sub,body)

    addtoDB(to_add, from_add, receivedDate, sub, body, outputclass)
    moveEmail(mail_path, outputclass)
    print(f'====>Event Processing completed')

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


# ## Directory Watcher

def on_created(event):
    print(f"====>Even Received: {event.src_path} received!")
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

#HandleNewEmail("inputEmails/email.txt")
