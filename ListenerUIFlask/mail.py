from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
#import pandas as pd
import os
import re
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mails.sqlite3'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class mails(db.Model):
   id = db.Column('mail_id', db.Integer, primary_key = True)
   mfrom = db.Column(db.String(50))
   mto = db.Column(db.String(50))
   mdate = db.Column(db.String(20))
   msubject = db.Column(db.String(200))
   mbody = db.Column(db.String(500))
   mclass = db.Column(db.String(50))

def __init__(self, mfrom,mto,mdate,msubject,mbody,mclass):
   self.mfrom = mfrom
   self.mto = mto
   self.mdate = mdate
   self.msubject = msubject
   self.mbody = mbody
   self.mclass = mclass

def write_mail(mail):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f=open("inputEmails/email_" + str(timestr) + '.txt', "w+")
    f.write("To: %s \n" % mail.mto)
    f.write("From: %s \n" % mail.mfrom)
    f.write("Subject: %s \n\n" % mail.msubject)
    f.write("%s \n" % mail.mbody)
    f.close()

@app.route('/')
def show_all():
   return render_template('show_all.html', mails = mails.query.order_by(mails.id.desc()).all() )

@app.route('/new', methods = ['GET', 'POST'])
def new():
   if request.method == 'POST':
      if not request.form['mto'] or not request.form['mfrom'] or not request.form['msubject']:
         flash('Please enter all the fields', 'error')
      else:
         mail = mails()
         m_class = 'NONE'
         #m_class = ProcessNewEmail("" , "", request.form['msubject'], request.form['mbody'])
         __init__(mail,request.form['mfrom'], request.form['mto'],'NONE',
            request.form['msubject'], request.form['mbody'],m_class)
         write_mail(mail)
         #db.session.add(mail)
         #db.session.commit()
         time.sleep(4)
         flash('Record was successfully added')
         return redirect(url_for('show_all'))
   return render_template('new.html')

if __name__ == '__main__':
   db.create_all()
   app.run(debug = True)
