import json
import numpy as np
from shapely.geometry import Point, Polygon
import cv2

HEIGHT = 1072
WIDTH  = 1920

scenes = ['Gura_Portitei_1', 'Gura_Portitei_2', 'Delta_Crisan_1']

for scene in scenes:
    json_path = f'dataset/{scene}_train/labels.json'

    with open(json_path) as f:
        data = json.load(f)

    for frame in data.keys():
        image = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)

        for i in range(WIDTH):
            for j in range(HEIGHT):
                p = Point(i, j)

                for region in data[frame]['regions']:
                    points_x = data[frame]['regions'][region]['shape_attributes']['all_points_x']
                    points_y = data[frame]['regions'][region]['shape_attributes']['all_points_y']

                    n = len(points_x)

                    polygon = Polygon([(points_x[k], points_y[k]) for k in range(n)])

            
                    if p.within(polygon):
                        image[j][i] = [255, 255, 255]
                        break

        path = f'dataset/{scene}_train/masks/{frame}'

        cv2.imwrite(path, image)