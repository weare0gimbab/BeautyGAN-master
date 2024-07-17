from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import base64
import os
import glob
import imageio.v2 as imageio
import cv2

app = Flask(__name__, template_folder='C:/Users/joeup/Desktop/weare0/BeautyGAN-master/templates')

def preprocess(img):
    return img / 127.5 - 1

def deprocess(img):
    return (img + 1) * 127.5

# Initialize TensorFlow model
model_path = 'C:/Users/joeup/Desktop/weare0/BeautyGAN-master/models'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint(model_path))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        in_memory_file = file.read()
        npimg = np.frombuffer(in_memory_file, np.uint8)
        no_makeup = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        no_makeup = cv2.cvtColor(no_makeup, cv2.COLOR_BGR2RGB)
        
        img_size = 256
        no_makeup = cv2.resize(no_makeup, (img_size, img_size))
        X_img = np.expand_dims(preprocess(no_makeup.astype(np.float32)), 0)
        makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
        result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3), dtype=np.float32)

        result[img_size: 2 * img_size, :img_size] = deprocess(X_img[0])

        for i, makeup_path in enumerate(makeups):
            makeup = imageio.imread(makeup_path)
            makeup = cv2.resize(makeup, (img_size, img_size))
            Y_img = np.expand_dims(preprocess(makeup.astype(np.float32)), 0)
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            Xs_ = deprocess(Xs_[0])
            result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup
            result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_

        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # Correct color channel order
        img_bytes = cv2.imencode('.jpg', result)[1].tobytes()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify({'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True)
