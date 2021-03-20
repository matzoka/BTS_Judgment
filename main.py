import os
from flask import Flask, request, redirect,render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime

CATEGORIES = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
IMG_SIZE = 150
UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = set(["png","jpg","jpeg","gif"])

app = Flask(__name__)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model("./bts_model.h5")

@app.route("/", methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            basename = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            file.save(os.path.join(UPLOAD_FOLDER, basename+".png"))
            filepath = os.path.join(UPLOAD_FOLDER, basename+".png")
            
            img = image.load_img(filepath,grayscale=True,target_size=(IMG_SIZE,IMG_SIZE))
            img = image.img_to_array(img)
            data = np.array([img])
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは" +  CATEGORIES[predicted] + "です"
            
            return render_template("index.html",answer=pred_answer,filename = filepath)
    return render_template("index.html",answer="",filename="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080))
    app.run(host = "0.0.0.0",port = port)
    #app.run()