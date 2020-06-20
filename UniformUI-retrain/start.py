# Saturday 20th JUNE 2020 11 AM
#start20.py is new version of start2.py
#includes wrong class, trans id search
#replaced index1 with index3 here

from flask import Flask, render_template, jsonify, abort, request, redirect, url_for, flash
from junecheckone import inputfunc
from flask_sqlalchemy import SQLAlchemy
import time
import csv

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mails.sqlite3'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)


class mails(db.Model):
    id = db.Column('mail_id', db.Integer, primary_key=True)
    mfrom = db.Column(db.String(50))
    mto = db.Column(db.String(50))
    mdate = db.Column(db.String(20))
    msubject = db.Column(db.String(200))
    tid=db.Column(db.String(10))
    mbody = db.Column(db.String(500))
    mclass = db.Column(db.String(50))


def __init__(self, mfrom, mto, mdate, msubject, tid, mbody, mclass):
    self.mfrom = mfrom
    self.mto = mto
    self.mdate = mdate
    self.msubject = msubject
    self.tid=tid
    self.mbody = mbody
    self.mclass = mclass


# write form input to text file so that listener can detect, classify, move to output class directory, write to DB
def write_mail(mail):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open("./inputEmails/email_" + str(timestr) + '.txt', "w+")
    f.write("To: %s \n" % mail.mto)
    f.write("From: %s \n" % mail.mfrom)
    #f.write("ID %s\n" % mail.mdate)
    f.write("Subject: %s \n\n" % mail.msubject)
    f.write("%s \n" % mail.mbody)
    #f.write("Email_Body: %s \n" % mail.mbody)
    #f.write("class: %s \n" % mail.m_class)
    f.close()


@app.route('/retrain')
def retrain():
    with open('emaildatasetnew.csv', 'a+') as f:
        out = csv.writer(f)
        ct=0
        for mail in mails.query.all():
            out.writerow([mail.mfrom, mail.mto, mail.msubject, mail.mbody, mail.tid, mail.mdate, mail.mclass])
            ct = ct +1
        db.session.query(mails).delete()
        db.session.commit()
        f.close()
        msg = '' + str(ct) + ' mail(s) sent for retraining successfully!'
    return render_template('index2.html',message=msg)


@app.route('/show')
def show_all():
    return render_template('show_all.html', mails=mails.query.order_by(mails.id.desc()).all())


@app.route("/", methods=["GET", "POST"])
def welcome():
    global dict, mclass, tid
    if request.method == "POST":
        dict=inputvalues = {
            "From": request.form['From'],
            "To": request.form['To'],
            "Subject": request.form['Subject'],
            "Message": request.form['Message'],
            "Date": request.form['date']
        }
        m_class, ID = inputfunc(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        mclass = m_class
        tid = ID
        return render_template("index3.html", m=m_class)
    else:
        return render_template("index3.html")


@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    if request.method == "POST":
        mclass = str(request.form.get("class", None))
        if mclass == 'newclass':
            return redirect(url_for('new_class'))
        mail = mails()
        __init__(mail, dict['From'], dict['To'], dict['Date'], dict['Subject'], tid, dict['Message'], mclass)
        #write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        #time.sleep(4)
        flash('Record was successfully added')
        return redirect(url_for('show_all'))
    else:
        return render_template("dropdown.html")


@app.route('/newclass', methods=['GET', 'POST'])
def new_class():
    if request.method == "POST":
        mclass = str(request.form['NewClass'])
        mail = mails()
        __init__(mail, dict['From'], dict['To'], dict['Date'], dict['Subject'], tid, dict['Message'], mclass)
        #write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        # time.sleep(4)
        flash('Record was successfully added')
        return render_template("index3.html")
    else:
        return render_template("new_class.html")


@app.route('/right')
def right():
    mail = mails()
    __init__(mail, dict['From'], dict['To'], dict['Date'], dict['Subject'], tid, dict['Message'], mclass)
    #write_mail(mail)
    db.session.add(mail)
    db.session.commit()
    return redirect(url_for('welcome'))


@app.route('/thread', methods = ['GET','POST'])
def findthread():
    if request.method == "POST":
        id = str(request.form['transacid'])
        return render_template('show_all.html', mails=mails.query.filter(mails.tid == id).all())
    else :
        return render_template('findthread.html')


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
