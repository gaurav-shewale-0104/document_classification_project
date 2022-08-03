from flask import Flask, render_template,request
import pickle
import numpy as np

model = pickle.load(open("nb_model.pkl","rb"))
count_vectoriser = pickle.load(open("count_vec.pkl","rb"))
encoder_class = pickle.load(open("encoder.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods= ["GET","POST"])
def predict():
    data = request.form
    # input_data = np.zeros(1)

    # input_data[0]=data["text"]
    text1 = ["".join("text")]
    cv = count_vectoriser.transform(text1)
    
    
    print(cv)
    result = model.predict(cv)
    class_list = encoder_class.classes_
    
    print({class_list[result[0]]})
    
    return render_template("index.html",prediction ={class_list[result[0]]})

if __name__== "__main__":
    app.run(host= "0.0.0.0",port=8080, debug=True) ##### AWS Deployment host = 0.0.0.0 port= 8080 debug= False



