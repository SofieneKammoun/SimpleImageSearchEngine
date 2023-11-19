import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from HistExtractor import HistExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
hist_fe = HistExtractor()
features = []
hists=[]
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)
for hist_path in Path("./static/hist").glob("*.npy"):
    hists.append(np.load(hist_path))
hists = np.array(hists)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        method = request.form['method']  # Assuming you have a form field for selecting the method
        num_res = int(request.form.get('num_res', 10))  # Default to 10 if not provided

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        if method == 'feature':
            query = fe.extract(img)
            dists = np.linalg.norm(features - query, axis=1)
        elif method == 'hist':
            query_hist = hist_fe.extract(Image.open(uploaded_img_path))
            dists = np.linalg.norm(hists - query_hist, axis=1)
        # Run search
        ids = np.argsort(dists)[:num_res]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
