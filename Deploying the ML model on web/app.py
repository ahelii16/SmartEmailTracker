from flask import Flask, render_template, redirect, request
import Classifier

# __name__ == __main__
app = Flask(__name__)

# @app.route('/about')
# def about():
# 	return "<h1> About Page </h1>"


# @app.route('/home')
# def home():
# 	return redirect('/')


# @app.route('/submit', methods = ['POST']) 
# def submit_data():  #we have to mention POST cuz this fn is triggered only when we hit submit URL, with method as POST
# 	if request.method == 'POST':
# 		num_str = request.form['no1']
# 		num1 = int(request.form['no1'])
# 		num2 = int(request.form['no2'])


# 		f = request.files['userfile']
# 		print(f) #prints filename on server(my terminal)
# 		f.save(f.filename) #save the files to the folder where app.py stored

# 	#return "<h1> Hello {}".format(num_str)
# 	return str(num1+num2)

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/', methods= ['POST'])
def marks():
	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename)# ./static/images.jpg
		f.save(path)

		caption = Caption_it.caption_this_image(path)
        
		print(caption)
		result_dic = {
		'image' : path,
		'caption' : caption
		}

	return render_template("index.html", your_result =result_dic)

if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)