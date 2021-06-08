import os
from PIL import Image
folder_path = 'datasets/medium10000_twoClasses'
bad_list = []
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for classes in os.listdir(sub_folder_path):
        classes_path = os.path.join(sub_folder_path, classes)
        if os.path.isdir(classes_path):
            for file in os.listdir(classes_path):
                file_path = os.path.join(classes_path, file)
                print('** Path: {}  **'.format(file_path), end="\r", flush=True)
                ext = file.split('.')[1]
                if ext  not in ['jpg', 'png', 'bmp', 'gif']:
                    print(f'file {file_path}  has an invalid extension {ext}')
                    bad_list.append(file_path)                    
                else:
                    try:
                        im = Image.open(file_path)
                        rgb_im = im.convert('RGB')
                    except:
                        print(f'file {file_path} is not a valid image file ')
                        bad_list.append(file_path)
                       
print (bad_list)
