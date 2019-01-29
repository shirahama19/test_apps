from sklearn.externals import joblib
import flask
import numpy as np

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
model = None


def load_model():
    
    '''
    学習済モデルを読み込む関数
    パスに指定したpklファイルをmodelに読み込む
    '''

    global model
    print(" * Loading pre-trained model ...")
    model = joblib.load("./model/sample-model.pkl")
    print(' * Loading end')


@app.route("/predict", methods=["POST"])
def predict():
    # レスポンスタイプを先に定義
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # POST method のリクエストを処理
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            # read feature from json
            feature = flask.request.get_json().get("feature")

            # preprocess for classification(分類器に入れるときにnumpy arrayに変換する)
            # list  -> np.ndarray
            feature = np.array(feature).reshape((1, -1))

            # 変換したfeatureを用いて、学習済みモデルで推論する
            # .tolist() でリスト型に変換
            response["prediction"] = model.predict(feature).tolist()

            # indicate that the request was a success
            response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print(" * starting server...")
    app.run(host='0.0.0.0', port=5000)