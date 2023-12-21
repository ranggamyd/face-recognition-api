import numpy as np
from siammese_network import SiameseModel, get_siamese_network, extract_encoder


siamese_network = get_siamese_network()
siamese_model = SiameseModel(siamese_network)
siamese_model.load_weights("./model/siamese_model-final")

encoder = extract_encoder(siamese_model)

def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    prediction = np.where(distance<=threshold, 0, 1)
    return prediction