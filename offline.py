from PIL import Image
from feature_extractor import FeatureExtractor
from HistExtractor import HistExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()   
    hist_fe = HistExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)
        
        # Extract histogram using the new histogram extractor
        hist_feature = hist_fe.extract(Image.open(img_path))
        hist_path = Path("./static/hist") / (img_path.stem + "_hist.npy")
        np.save(hist_path, hist_feature)
