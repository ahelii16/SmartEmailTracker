from flask import Flask, render_template, redirect, request

# __name__ == __main__
app = Flask(__name__)

friends = ["Prateek", "Jatin", "Sohail"]

num = 5

#we can create routes so user can go to diff URLs
# '/' is the home route
# this means that when user hits this route(URL), hello fn will be triggered

@app.route('/')
def hello():
    #return "Hello World"
    #return render_template("index.html")
	return render_template("index.html", my_friends = friends , number = num )

@app.route('/about')
def about():
	return "<h1> About Page </h1>"


@app.route('/home')
def home():
	return redirect('/')


@app.route('/submit', methods = ['POST']) 
def submit_data():  #we have to mention POST cuz this fn is triggered only when we hit submit URL, with method as POST
	if request.method == 'POST':
		num_str = request.form['no1']
		num1 = int(request.form['no1'])
		num2 = int(request.form['no2'])


		f = request.files['userfile']
		print(f) #prints filename on server(my terminal)
		f.save(f.filename) #save the files to the folder where app.py stored

	#return "<h1> Hello {}".format(num_str)
	return str(num1+num2)




if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)