import os
import re
from PIL import Image, ImageOps
import uuid
import pandas as pd


root = "./data"
csv_path = "./data/noisy_imagenette.csv"
filesArray = []

def create_uuid(): 
    return str(uuid.uuid4().hex[:10])

# load all existing files
for path, subdirs, files in os.walk(root):
    for name in files:
        if "JPEG" in name:
            classes = re.findall('n[0-9]{8}', path)[0]
            filesArray.append([os.path.join(path, name), classes])


#load csv for paths and classes
classes_df = pd.read_csv(csv_path)
#print(classes_df.head)

#generate new files and add to csv

for img_data in filesArray:
    img = Image.open(img_data[0])
    im_flip = ImageOps.flip(img)
    im_mirror = ImageOps.mirror(img)


    im_class = img_data[1]
    im_flip_path = './data/aug/flip/'+im_class+'_'+create_uuid()+'.JPEG'
    im_mirror_path = './data/aug/mirror/'+im_class+'_'+create_uuid()+'.JPEG'

    im_flip.save(im_flip_path, quality=100)
    im_mirror.save(im_mirror_path, quality=100)

    df2 = pd.DataFrame({"path":[im_flip_path.replace("./data/", ""),im_mirror_path.replace("./data/", "")],"noisy_labels_0": [im_class, im_class]})
    classes_df = classes_df.append(df2, ignore_index=True)



classes_df.to_csv(path_or_buf= "./data/noisy_imagenette_extended.csv", index=False)

