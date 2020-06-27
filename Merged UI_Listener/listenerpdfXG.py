"""
Created on Mon Jun 15 00:04:39 2020

@author: divya
"""
# Tuesday 23rd JUNE 2020 10PM
#loads XG boost model

import numpy as np
import pandas as pd
import re
from word2vec_XGB import inp
#from Glove_XGBoost import inp
from file_parser import parse
from parseimage import ocr


from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import sqlite3


import sys
import os
import shutil
import time



def HandleNewEmail(mail_path):
    #Parse email to store in DB
    #print(type(mail_path))
    #added this for image remove if problematic
    if (mail_path.endswith('.png') or mail_path.endswith('.jpg') or mail_path.endswith('.jpeg')):
        #print('Email Image uploaded!!')
        extracted_text = ocr(mail_path)
        email = extracted_text.split('\n')

    elif (mail_path.endswith('.pdf')):
        #print('PDF received')
        extracted_text = parse(mail_path)
        email = extracted_text.split('\n')
        #print(email)

    elif (mail_path.endswith('.txt')):
        #print('txt received')
        email=open(mail_path, "r")
    else:
        '''
        UnHandled file type
        '''
        return '', '', '', '', '', '', ''

    to_add=''
    from_add=''
    sub=''
    body=''
    for line in email:
            #print(f'**{line}**')
            if line.startswith('To: ') or line.startswith('to: '):
                pieces = line.split()
                to_add = pieces[1]
                #print(f'To---->Found!!!{to_add}')
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

    if (mail_path.endswith('.txt')):
        email.close()
        #print(encoded_mail)

    outputclass , id = inp(to_add,from_add,sub,body)

    #addtoDB(to_add, from_add, receivedDate, sub, id, body, outputclass)
    #moveEmail(mail_path, outputclass)
    print(f'====>Event Processing completed')

    return to_add, from_add, receivedDate, sub, id, body, outputclass

def addtoDB(to_add, from_add, receivedDate, sub, id, body, outputclass):
    #add email and class to DB
    #conn = sqlite3.connect('mails.sqlite3')
    conn = sqlite3.connect('User.db')
    cur = conn.cursor()
    cur.execute('''INSERT INTO mails (mto, mfrom, mdate, msubject, ID, mbody, m_class)
               VALUES (?, ?, ?, ?, ?, ?, ?)''', (to_add, from_add, receivedDate, sub, id, body, outputclass))
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
    print(f"====>Event Received: {event.src_path} received!")
    to_add, from_add, receivedDate, sub, id, body, outputclass = HandleNewEmail(event.src_path)
    if outputclass!='':
        addtoDB(to_add, from_add, receivedDate, sub, id, body, outputclass)
        moveEmail(event.src_path, outputclass)

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
