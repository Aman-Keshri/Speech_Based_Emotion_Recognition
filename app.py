from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import emotion_classifier

classifier = emotion_classifier.EmotionClassifier()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index1.html')

@app.route("/second", methods =['GET'])
def second():
    return render_template('index2.html')

@app.route("/refresh", methods=['GET'])
def refresh():
    return render_template('index1.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():    
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        f = str(f)
        fn = f.split("'")
        fn1 = os.path.join(os.getcwd(), fn[1])
        predicted_emotion_array = classifier.classify_audio(str(fn1))
        return render_template('index3.html', names = predicted_emotion_array[0])

@app.route("/record", methods=['GET'])
def record():
    fn = classifier.record_audio()
    predicted_emotion_array = classifier.classify_audio(fn)
    return render_template('index3.html', names = predicted_emotion_array[0])

if __name__ == "__main__":
    app.debug = True
    app.run()