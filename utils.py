import cv2
 
def detect_smile(gray_frame, faces, cascade_smile):
    smiles = []
    for (x, y, w, h) in faces:
        the_face = gray_frame[y:y+h, x:x+w] # get face bounding box
        smiles = cascade_smile.detectMultiScale(the_face,scaleFactor=2, minNeighbors=35) # detect smile
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0,255,0), 2) 
    return len(smiles) > 0