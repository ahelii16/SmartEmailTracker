from flask import Flask, render_template, jsonify, abort, request, redirect, url_for
from ReadClassify import Preprocess
from funct import add, temp, pred
from testing import regressor
from junecheckone import inputfunc

app = Flask(__name__)

counter = 0

# def card_view(index):
#   cards = db[index]
#   return render_template("cards.html",
#                          cards=cards,
#                          index=index,
#                          max_index=len(db) - 1)


@app.route("/classified_class/<string:mesg>")
def cls(mesg):
    return render_template('outputclass.html', m=mesg)


@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        inputvalues = {
            "To": request.form['To'],
            "From": request.form['From'],
            "Subject": request.form['Subject'],
            "Message": request.form['Message']
        }
        m = inputfunc(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        print(m)
        return redirect(url_for('cls', mesg=m))
    else:
        return render_template("index.html")


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