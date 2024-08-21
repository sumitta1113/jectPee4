from flask import Flask, render_template, request, redirect, url_for, jsonify
import face_recognition
import cv2
import tempfile
import os
import base64
from flask_socketio import SocketIO, emit
from face_image import face_image_bp
from hat_or_glasses import hat_or_glasses_bp


app = Flask(__name__)
#******


# ลงทะเบียนบลูปริ้นต์
app.register_blueprint(face_image_bp)
app.register_blueprint(hat_or_glasses_bp)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/face_image')
def face_image():
    return render_template('face_image.html')

@app.route('/hat_or_glasses')
def hat_or_glasses():
    return render_template('hat_or_glasses.html')

if __name__ == "__main__":
    app.run(debug=True)