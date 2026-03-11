from PIL import Image # type: ignore
import numpy as np

img = Image.open("/home/leon/Documenti/TestTesi/Immagini/VIA_DELL'ABETONE_E_DEL_BRENNERO_PIEVEPELAGO/5.6.png")
array = np.array(img)

