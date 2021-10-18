from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import keras
from keras.preprocessing import image

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def load_model():
    # load the pre-trained Keras model
    global model
    #model = ResNet50(weights='imagenet')
    model = keras.models.load_model('/app/app/models/my_model.h5')


def prepare_image(image_in):
    target = (32, 32)

    if image_in.mode != "RGB":
        image_in = image.convert("RGB")

    test_image1 = image_in.resize(target)
    test_image = image.img_to_array(test_image1)
    test_image = np.expand_dims(test_image, axis=0)
    # return the processed image
    return test_image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    result = []

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            # image = flask.request.files["image"].read()
            image_input = flask.request.files["image"].read()
            image_input = Image.open(io.BytesIO(image_input))

            # preprocess the image and prepare it for classification
            #image = prepare_image(image, target=(224, 224))
            test_image = prepare_image(image_input)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            result = model.predict(test_image)
            data["predictions"] = []

            # indicate that the request was a success
            data["success"] = True
            print(result)
            if result[0][0] == 1:
                label = "Aeroplane"
            elif result[0][1] == 1:
                label = 'Automobile'
            elif result[0][2] == 1:
                label = 'Bird'
            elif result[0][3] == 1:
                label = 'Cat'
            elif result[0][4] == 1:
                label = 'Deer'
            elif result[0][5] == 1:
                label = 'Dog'
            elif result[0][6] == 1:
                label = 'Frog'
            elif result[0][7] == 1:
                label = 'Horse'
            elif result[0][8] == 1:
                label = 'Ship'
            elif result[0][9] == 1:
                label = 'Truck'
            else:
                print('Error')

            r = {"label": label}
            data["predictions"].append(r)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
