from flask import Flask, render_template, jsonify, abort, request, redirect, url_for, flash
# from funct import add, pred
# from testing import regressor
from junecheckone import inputfunc
from flask_sqlalchemy import SQLAlchemy
import time

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
    mbody = db.Column(db.String(500))
    m_class = db.Column(db.String(50))


def __init__(self, mfrom, mto, mdate, msubject, mbody, m_class):
    self.mfrom = mfrom
    self.mto = mto
    self.mdate = mdate
    self.msubject = msubject
    self.mbody = mbody
    self.m_class = m_class


# write form input to text file so that listener can detect, classify, move to output class directory, write to DB
def write_mail(mail):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open("/home/siddharth/Desktop/Codes/inputEmails/email_" + str(timestr) + '.txt', "w+")
    f.write("To: %s \n" % mail.mto)
    f.write("From: %s \n" % mail.mfrom)
    f.write("ID %s\n" % mail.mdate)
    f.write("Subject: %s \n\n" % mail.msubject)
    f.write("Email_Body: %s \n" % mail.mbody)
    f.write("class: %s \n" % mail.m_class)
    f.close()


@app.route("/classified_class/<string:mesg>")
def cls(mesg):
    return render_template('outputclass.html', m=mesg)


@app.route('/show')
def show_all():
    return render_template('show_all.html', mails=mails.query.order_by(mails.id.desc()).all())


@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        inputvalues = {
            "To": request.form['To'],
            "From": request.form['From'],
            "Subject": request.form['Subject'],
            "Message": request.form['Message']
        }
        mail = mails()
        m_class, ID = inputfunc(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        __init__(mail, inputvalues['From'], inputvalues['To'], ID,
                 inputvalues['Subject'], inputvalues['Message'], m_class)
        write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        time.sleep(4)
        flash('Record was successfully added')
        return redirect(url_for('cls', mesg=m_class))
    else:
        return render_template("index.html")


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
'''
@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        m = pred(request.form['Experience'], request.form['test_score'], request.form['interview_score'])
        return redirect(url_for('test_ans', ans=m))
    else:
        return render_template("test.html")


@app.route("/test_ans/<int:ans>")
def test_ans(ans):
    return render_template('test_ans.html', ans=ans)

'''
