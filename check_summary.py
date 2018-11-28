from keras.models import load_model
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

model = load_model(args["model"])

# model.summary()

for idx, layer in enumerate(model.layers):
	print(idx, layer.name)