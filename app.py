from pathlib import Path
from flask import Flask, render_template, request
from fastai.vision.all import load_learner


app = Flask(__name__)
model = load_learner('ball_classif.pkl')

@app.route('/', methods = ['GET'])
def home():
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "images" + imagefile.filename
    imagefile.save(image_path)
    prediction = list(model.predict(image_path))
    prediction = prediction[0]
    if prediction == "tennis":
        prediction = "tennis ball"
    ##Path.unlink(image_path)

    return render_template("index.html", prediction = prediction)


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = "8080")