import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False
def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=False, num_cols=2, figsize=(30, 20), title_fontsize=15):
    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    """if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))
    """
    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


imgs0 = load_images_from_folder('/home/rodrigo/Workspace/IA_2022/trocr/images/input2')
#show_image_list(imgs0)


imgs = []
for img in imgs0:
  #imgs.append(img[1100:2100,800:2200])
  imgs.append(img[1300:2300,900:2000])


grays = []

for img in imgs:
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grays.append(gray)

#show_image_list(grays)


ths = []
umbral = 70
for gray in grays:
  th = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY_INV)[1]
  ths.append(th)

#show_image_list(ths,figsize=(50,60))


contours_list = []
canvas_list = []
cant = len(imgs)

for i in range(cant):
  canvas_list.append(np.zeros_like(imgs[i]))

for i in range(cant):
  contours = cv2.findContours(ths[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
  contours_list.append(contours)
  canvas_list[i] = np.zeros_like(imgs[i])
  cv2.drawContours(canvas_list[i] , contours, -1, (0, 255, 0), 2)


#show_image_list(canvas_list,grid=True,figsize=(50,60))


candidatos_list = []

canvas_list = []
for i in range(cant):
  canvas_list.append(np.zeros_like(imgs[i]))

for i in range(len(contours_list)):
  candidatos = []

  for cnt in contours_list[i]:
    x, y, w, h = cv2.boundingRect(cnt)
    if ( ( 70>w > 25) and (70>h>25) and ( not y ==0 ) and (not x == 0)):
      candidatos.append(cnt)
  candidatos_list.append(candidatos)
  cv2.drawContours(canvas_list[i] , candidatos, -1, (0, 255, 0), 2)
#show_image_list(canvas_list,figsize=(50,60),grid=True)


canvas_list2 = []
for i in range(cant):
  canvas_list2.append(np.zeros_like(imgs[i]))


candidatos_list2 = []
for i in range(cant):
  #print(i)
  candidatos = []
  for j in range(len(candidatos_list[i])):
    cnt = candidatos_list[i][j]
    x1,y1,_,_ = cv2.boundingRect(cnt)
    contador = 0
    for k in range(len(candidatos_list[i])):
      x2,y2,_,_ = cv2.boundingRect(candidatos_list[i][k])
      distancia = math.sqrt((y2-y1)**2)
      if (distancia < 50):
        contador += 1
    if contador >= 4:
      candidatos.append(cnt)
      #print(f'{y1} es candidato ')
  candidatos_list2.append(candidatos)
  cv2.drawContours(canvas_list2[i] , candidatos, -1, (0, 255, 0), 2)
#show_image_list(canvas_list2,figsize=(50,60))


canvas_list = []
placa = []
for i in range(cant):
  canvas_list.append(np.zeros_like(imgs[i]))

defecto = candidatos_list2[0][1]

for i in range(cant):
  xs = []
  candidates = candidatos_list2[i]
  if len(candidates)>1:
    for cnt in candidates:
        x, y, w, h = cv2.boundingRect(cnt)
        xs.append(x)
    license = candidates[np.argmin(xs)]
  elif len(candidates)== 1 :
    license = candidates[0]
  elif len(candidates)==0:
    license = defecto
  placa.append(license)
  cv2.drawContours(canvas_list[i] , [license], -1, (0, 255, 0), 2)


#show_image_list(canvas_list,figsize=(50,60))

recortes = []

for i in range(len(placa)):
  license = placa[i]
  #x, y, w, h = cv2.boundingRect(license)
  #cropped = grays[i][y:y+h,x:x+w]
  x, y, w, h = cv2.boundingRect(license)
  cropped2 = imgs[i][y-75:y+75,x-75:x+400]

  recortes.append(cropped2)

for i in range(len(recortes)):
  cv2.imwrite(f'/home/rodrigo/Workspace/IA_2022/trocr/images/output2/plate{i}.png',recortes[i])

#show_image_list(recortes)