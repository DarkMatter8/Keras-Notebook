from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

#print(loaded_model.summary())

test_image = image.load_img('random.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = loaded_model.predict(test_image)

if result[0][0] >= 0.5:
	print('Plane')
else:
	print('Car')