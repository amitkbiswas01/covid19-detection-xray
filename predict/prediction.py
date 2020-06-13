import cv2
import os
import glob
import imutils
import numpy as np
import matplotlib

import tensorflow as tf
import matplotlib.pyplot as plt

matplotlib.use('TKAgg',warn=False, force=True)


class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
            )
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def overlay_heatmap(self, heatmap, img, alpha=0.5, colormap = cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        img = np.squeeze(img)
        img = cv2.resize(img, (224,224))
        output = cv2.addWeighted(img, alpha, heatmap, alpha, 0)
        return (heatmap, output)


def predict(model, test_path):

    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            visualising_layer = layer.name
            break

    img_list = glob.glob(test_path)
    for img in img_list:
        temp_img = open(img, 'rb').read()
        temp_image = tf.image.decode_jpeg(temp_img, channels=3)
        temp_img = tf.cast(temp_image, tf.float32) * (1./255)
        temp_img = tf.image.resize(temp_img, (224,224))
        expan_img = tf.expand_dims(temp_img, axis=0)

        pred = model.predict(expan_img)
        gc = GradCAM(model, 0, layerName=visualising_layer)
        heatmap = gc.compute_heatmap(expan_img)
        heatmap = cv2.resize(heatmap, (224,224))
        (heatmap, output) = gc.overlay_heatmap(heatmap, temp_image)
        image = cv2.resize(np.array(temp_image), (224,224))

        output = np.vstack([image, output])
        output = imutils.resize(output, height=700)
        # plt.imshow(output)
        # plt.show()
        # print(pred)
        pred_val = "Positive" if pred[0][0] < 0.3 else "Negative"
        str_val = str(pred_val) + "_____" + str(pred[0])
        # plt.savefig(str_val + '.png')
        print(str(img))
        print(str_val)
        print('\n========================================\n')


def main():
    model = tf.keras.models.load_model('trained_models/vgg.h5')
    dir = 'predict/test_images/*'
    predict(model, dir)


if __name__ == '__main__':
    main()
