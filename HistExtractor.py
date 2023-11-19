import numpy as np
from PIL import Image
import scipy.misc
from matplotlib.pyplot import imread
class HistExtractor:
     
  def extract(self, img):
    # Convert the image to grayscale
    img = img.convert('L')
    # Calculate histogram
    hist = np.histogram(img, bins=20, range=(0, 256))[0]
    # Normalize the histogram
    hist = hist / hist.sum()
    return hist