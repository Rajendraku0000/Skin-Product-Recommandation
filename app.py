import cv2
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import reco

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_skin_tone_from_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    # Assuming only one face is detected
    (x, y, w, h) = faces[0]
    
    # Extract the region of interest (ROI) which is the face
    face_roi = frame[y:y+h, x:x+w]
    
    # Convert the ROI to the YCbCr color space
    ycbcr_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    
    # Extract the Cr and Cb channels
    cr = ycbcr_face[:,:,1]
    cb = ycbcr_face[:,:,2]
    
    # Compute the mean and standard deviation of Cr and Cb channels
    mean_cr = np.mean(cr)
    mean_cb = np.mean(cb)
    std_cr = np.std(cr)
    std_cb = np.std(cb)
    
    # Skin tone detection based on Cr and Cb values
    if (mean_cr > 135 and mean_cr < 180) and (mean_cb > 85 and mean_cb < 135) and (std_cr < 15 and std_cb < 15):
        return "light skin tone"
    elif (mean_cr > 105 and mean_cr < 135) and (mean_cb > 135 and mean_cb < 175) and (std_cr < 15 and std_cb < 15):
        return "medium skin tone"
    elif (mean_cr > 85 and mean_cr < 105) and (mean_cb > 175 and mean_cb < 225) and (std_cr < 15 and std_cb < 15):
        return "dark-normal skin tone"
   
    else:
        return "dark-normal skin tone"
    

# Initialize Flask app
app = Flask(__name__)

# OpenCV video capture
cap = cv2.VideoCapture(0)
skin_tone=None

def gen_frames():
    while True:
        global skin_tone
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Get skin tone from frame
        skin_tone = get_skin_tone_from_frame(frame)

        if skin_tone:
            cv2.putText(frame, skin_tone, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def new():
    return render_template("index.html",prediction=None)

@app.route('/ragis', methods=['POST'])
def submit():
    # print(skin_tone)
    a=reco.recommend(skin_tone)
    # print(a)
    
    return render_template("index.html",prediction=a)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
