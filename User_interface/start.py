
from flask import Flask, render_template, jsonify, abort, request, redirect, url_for, flash
# from funct import add, pred
# from testing import regressor

from junecheckone import inputfunc
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, email_validator, ValidationError
from flask_login import LoginManager, login_user, UserMixin, current_user, login_required, logout_user

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mails.sqlite3'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///User.db'

app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}','{self.password}')"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired(), Length(min=6, max=20)])
    confirm_password = StringField('Confirm Password',
                                   validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('SignUp')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('This username is already taken. Please choose a different username')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('This email is already taken. Please choose a different email-address')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired(), Length(min=6, max=20)])
    submit = SubmitField('SignUp')
    remember = BooleanField('Remember Me')


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
    f = open("./inputEmails/email_" + str(timestr) + '.txt', "w+")
    f.write("To: %s \n" % mail.mto)
    f.write("From: %s \n" % mail.mfrom)
    f.write("ID %s\n" % mail.mdate)
    f.write("Subject: %s \n\n" % mail.msubject)
    f.write("Email_Body: %s \n" % mail.mbody)
    f.write("class: %s \n" % mail.m_class)
    f.close()


@app.route('/show')
@login_required
def show_all():
    return render_template('show_all.html', mails=mails.query.filter(mails.mto == current_user.email).all())


@app.route("/", methods=["GET", "POST"])
def welcome():
    global dict, mclass, id
    if request.method == "POST":
        dict=inputvalues = {
            "From": request.form['From'],
            "To": request.form['To'],
            "Subject": request.form['Subject'],
            "Message": request.form['Message']
        }
        # mail = mails()
        m_class, ID = inputfunc(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        mclass = m_class
        id = str(ID)
        return render_template("index1.html", m=m_class)
    else:
        return render_template("index1.html")


@app.route('/newclass', methods=['Get', 'POST'])
def new_class():
    if request.method == "POST":
        mclass = str(request.form['NewClass'])
        mail = mails()
        __init__(mail, dict['To'], dict['To'], id,
                 dict['Subject'], dict['Message'], mclass)
        write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        # time.sleep(4)
        flash('Record was successfully added')
        return render_template("index1.html")
    else:
        return render_template("new_class.html")


@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    if request.method == "POST":
        mclass = str(request.form.get("class", None))
        if mclass == 'newclass':
            return redirect(url_for('new_class'))
        mail = mails()
        __init__(mail, dict['To'], dict['To'], id,
                 dict['Subject'], dict['Message'], mclass)
        write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        #time.sleep(4)
        flash('Record was successfully added')
        return redirect(url_for('show_all'))
    else:
        return render_template("dropdown.html")


@app.route('/right')
def right():
    mail = mails()
    __init__(mail, dict['To'], dict['To'], id,
             dict['Subject'], dict['Message'], mclass)
    write_mail(mail)
    db.session.add(mail)
    db.session.commit()
    time.sleep(4)
    flash('Record was successfully added')
    return redirect(url_for('welcome'))


@app.route('/thread', methods = ['GET','POST'])
@login_required
def findthread():
    if request.method == "POST":
        id = str(request.form['transacid'])
        return render_template('show_all.html', mails=mails.query.filter(mails.mdate == id and mails.mto == current_user.email).all())
    else :
        return render_template('findthread.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('welcome'))
        else:
            flash('Login unsuccessful. You have entered incorrect email or password', 'danger')
    return render_template('login.html', title='login', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('welcome'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
