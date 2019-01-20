from flask import Flask, jsonify, request, make_response, render_template
from flask_cors import CORS
import api
app = Flask(__name__,static_folder="../../frontend/build",template_folder='../../frontend/build')
CORS(app)
# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    return render_template('../../frontend/build/index.html')

# GET
# @app.route('/api')
# def get_api():
#   img = request.get_json()
#   api_response = api.send_request(img)
#   return jsonify(api_response)

# POST
@app.route('/api', methods=['POST'])
def send_image():
  json = request.get_json(force=True)
  img = json['img']['data']['data']
  img_width = json['img']['width']
  img_height = json['img']['height']
  api_response = api.send_request(img,img_width,img_height)
  resp = make_response(jsonify(api_response),200)
  resp.mimetype = "application/json"
  resp.headers['X-Content-Type-Options'] = 'nosniff'
  resp.headers['Content-Type'] = "application/json"
  return resp