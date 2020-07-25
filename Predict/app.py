'''
Flask API to make predictions
'''
import logging
import os
from flask import Flask, request
import prediction

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))


@app.route('/', methods=['POST'])
def api_root():

    '''Function to call when a POST request is made.

        Parameters:
            None
        Return Value:
            Predictions List.
    '''

    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.is_json:
        json_data = request.get_json()        
        files = []
        for i in json_data['records']:
            files.append(i['Image URI'])
        model_path = json_data['model_path']
        print(model_path)
        data = prediction.load_preprocess_data(files)
        print("Downloading Model")
        status = prediction.download_and_load_model(model_path)
        print(status)
        predict = prediction.predict_on_data(data)
        return predict


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
