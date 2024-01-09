from flask import Flask, request, jsonify
from predict.predict import run as run_predict
import json

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    artefacts_path = 'C:/Users/adrie/Desktop/MDE_5A/from_proc_to_prod/poc-to-prod-capstone/poc-to-prod-capstone/train/data/artefacts/2024-01-09-10-57-13'
    model = run_predict.TextPredictionModel.from_artefacts(artefacts_path)

    # Text for prediction
    user_text = "Is it possible to execute the procedure of a function in the scope of the caller?"

    # Perform prediction
    predictions = model.predict([user_text])

    results = [model.labels_to_index[str(idx)] for idx in predictions[0]]
    result_json = json.dumps({'input_text': user_text, 'predictions': results})
    return result_json


if __name__ == '__main__':
    app.run(debug=True)
