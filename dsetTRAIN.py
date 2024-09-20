from flask import Flask, request, jsonify, Response
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import json
import os

app = Flask(__name__)

device =torch.device("cpu")
hf_token = "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"


model = BertForSequenceClassification.from_pretrained("berkegokce/sent_trained_data", use_auth_token=hf_token).to(device)
tokenizer = BertTokenizer.from_pretrained("berkegokce/sent_trained_data", use_auth_token=hf_token)

ds = load_dataset("turkish-nlp-suite/sinefil-movie-reviews")
test_verisi = ds['test']


def tahmini_data_ensemble(dataset, model, tokenizer):
    model_outputs = []
    for item in dataset:
        text = item['text']


        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            model_outputs.append(pred)

    return model_outputs


def veri_ratingleri(tahminler, labels):
    doğru_tahmin = sum(p == l for p, l in zip(tahminler, labels))
    doğruluk_payı = doğru_tahmin / len(labels) * 100
    return doğruluk_payı


@app.route('/', methods=['GET', 'POST'])
def anasayfa():
    if request.method == 'POST':
        if not request.json or 'review' not in request.json:
            return jsonify({'error': 'Geçersiz veri'}), 400

        review_text = request.json['review']
        tahminler = tahmini_data_ensemble([{'text': review_text}], model, tokenizer)
        sentiments = {0: 'Pozitif', 1: 'Nötr', 2: 'Negatif'}
        sentiment_label = sentiments.get(tahminler[0], "Unknown")

        response_data = {"review": review_text, "sentiment": sentiment_label}
        response_json = json.dumps(response_data, ensure_ascii=False)
        return Response(response_json, content_type='application/json; charset=utf-8')

    return "Duygu Analizi API"


@app.route('/predict_dataset', methods=['GET'])
def dataset_analizi():
    review_limit = 50
    dataset_subset = test_verisi.select(range(min(review_limit, len(test_verisi))))
    reviews = [item['text'] for item in dataset_subset]
    labels = [item['score'] for item in dataset_subset]

    tahminler = tahmini_data_ensemble(dataset_subset, model, tokenizer)
    doğruluk_payı = veri_ratingleri(tahminler, labels)

    predictions = []
    for i, review in enumerate(reviews):
        sentiments = {0: 'Pozitif', 1: 'Nötr', 2: 'Negatif'}
        sentiment_label = sentiments.get(tahminler[i], "Unknown")
        predictions.append({
            "review": review,
            "sentiment": sentiment_label
        })

    performance_report = {
        "accuracy": f"{doğruluk_payı:.2f}%",
        "predictions": predictions
    }

    response_json = json.dumps(performance_report, ensure_ascii=False)
    return Response(response_json, content_type='application/json; charset=utf-8')


@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'review' not in request.json:
        return jsonify({"error": "Geçersiz veri"}), 400

    review_text = request.json['review']
    tahminler = tahmini_data_ensemble([{'text': review_text}], model, tokenizer)
    sentiments = {0: 'Pozitif', 1: 'Nötr', 2: 'Negatif'}
    sentiment_label = sentiments.get(tahminler[0], "Unknown")

    response_data = {"review": review_text, "sentiment": sentiment_label}
    response_json = json.dumps(response_data, ensure_ascii=False)
    return Response(response_json, content_type='application/json; charset=utf-8')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
