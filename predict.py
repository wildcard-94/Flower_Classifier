import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import numpy as np
import json
from json import loads







#Argument parser 
def parse_args():
    #parser for user input 
    parser = argparse.ArgumentParser(description="Flowers Classifier")
    parser.add_argument('-i', '--image',
                        required=False,
                        default='./test_images/cautleya_spicata.jpg',
                        help='image path /test_images/XX.jpg')
    parser.add_argument('-m', '--model',
                        required=False,
                        default='./train_model.h5',
                        help='Train model path XX.h5')
    parser.add_argument('-t', '--top_k',
                        required=False,
                        default=5,
                        help='Top k predections ')
    parser.add_argument('-c', '--category_names',
                        required=False,
                        default='label_map.json',
                        help='Class label names XX.json')
    args = vars(parser.parse_args)

    return parser.parse_args()

# process images

def process_image(image):
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image


    

#Predicting Classes
def predict(image_path ,model_path , top_k, classes):

    
    top_k = int(top_k)  
    model = load_model(model_path)
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    predict = model.predict(np.expand_dims(processed_test_image, axis = 0))
    predict = predict[0].tolist()
    
    #Top K Classes
    ps, indices = tf.math.top_k(predict, top_k)
    indices = indices.numpy().tolist()
    ps = ps.numpy().tolist()
    
    #Displaying Class Names
    label = [classes[str(i)] for i in indices]
    print("top indices:", indices)
    print("top probs:", ps)
    print(label)       
  


#load label map file  
def load_json(label_map):
    with open(parse_args().category_names, 'r') as f:
        class_names = json.load(f)
    classes = dict()
    for k in class_names:        
        classes[str(int(k)-1)] = class_names[k]            
    return classes


#load the saved model       
def load_model(model_path):
    
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

# call the functions 
def main():
    #check tensorflow cpu/gpu 
    print(tf.test.gpu_device_name())
    args = parse_args()  
    classes = load_json(args.category_names)
    predict(args.image ,args.model, args.top_k, classes)
    
    

if __name__ == "__main__":
    
    main()