import numpy as np
from graphics import *
import json

import cv2


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class SplitImage(object):

    def __init__(self, use_fix_size=True, use_least_pieces_size=True):
        self.use_fix_size = use_fix_size
        self.use_least_pieces_size = use_least_pieces_size

    def clip(self, subjectPolygon, clipPolygon):
        def inside(p):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

        def computeIntersection():
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        outputList = subjectPolygon
        cp1 = clipPolygon[-1]

        for clipVertex in clipPolygon:
            cp2 = clipVertex
            inputList = outputList
            if not inputList:
                return None
            outputList = []
            s = inputList[-1]

            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e):
                    if not inside(s):
                        outputList.append(computeIntersection())
                    outputList.append(e)
                elif inside(s):
                    outputList.append(computeIntersection())
                s = e
            cp1 = cp2
        if len(outputList) != 0:
            return np.array(outputList)
        else:
            return None

    def polygon(self, points):
        points_1 = []
        for i in range(len(points)):
            pt = Point(points[i][0], points[i][1])
            points_1.append(pt)
        Polygon(*points_1)
        # return points_1

    def __get_corners(self, image_size, piece, stride, overlap):
        def left_move(image_size, corners):
            iw, ih = image_size
            new_corners = []
            for sx, sy, ex, ey in corners.reshape(-1, 4):
                if ex > iw:
                    sx -= (ex - iw)
                    ex = iw
                if ey > ih:
                    sy -= (ey - ih)
                    ey = ih
                sx, sy = max(0, sx), max(0, sy)
                new_corners.append([sx, sy, ex, ey])
            return np.array(new_corners).reshape(corners.shape)

        corners = []
        for x in range(piece[0]):
            col_corners = []
            for y in range(piece[1]):
                sx, sy = np.array([x, y]) * stride
                ex, ey = (np.array([x, y]) + 1) * stride + overlap
                col_corners.append([sx, sy, ex, ey])
            corners.append(col_corners)
        corners = np.array(corners)
        if self.use_fix_size:
            corners = left_move(image_size, corners)
        else:
            corners[:, :, [0, 2]] = np.clip(corners[:, :, [0, 2]], 0, image_size[0])
            corners[:, :, [1, 3]] = np.clip(corners[:, :, [1, 3]], 0, image_size[1])
        return corners.transpose((1, 0, 2))

    def __get_sub_image_corners_by_pieces(self, image_size, pieces=(1, 1), overlap=(0, 0)):
        """
        :param image_size: tuple, (w, h) of origin image .
        :param pieces: tuple, (pw, ph), how many pieces to split image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """
        size, pieces, overlap = np.array(image_size), np.array(list(pieces)), np.array(list(overlap))
        stride = np.ceil((size - overlap) / pieces).astype(np.int)
        return self.__get_corners(image_size, pieces, stride, overlap)

    def __get_sub_image_corners_by_size(self, image_size, sub_image_size, overlap=(0, 0)):
        """
        :param image_size: tuple, (w, h) of origin image .
        :param sub_image_size: tuple, (sw, sh), size of sub image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """

        def least_pieces_size(image_size, size):
            c1 = np.ceil(image_size[0] / size[0]) * np.ceil(image_size[1] / size[1])
            c2 = np.ceil(image_size[0] / size[1]) * np.ceil(image_size[1] / size[0])
            if c1 <= c2:
                return size
            else:
                return size[1], size[0]

        if self.use_least_pieces_size:
            sub_image_size = least_pieces_size(image_size, sub_image_size)

        size, sub_image_size, overlap = np.array(image_size), np.array(list(sub_image_size)), np.array(list(overlap))
        stride = np.ceil(sub_image_size - overlap).astype(np.int)
        piece = np.ceil((size - overlap) / (sub_image_size - overlap)).astype(int)
        return self.__get_corners(image_size, piece, stride, overlap)

    def get_sub_image_corners(self, image_size, pieces=None, sub_image_size=None, overlap=(0, 0)):
        """
            cut function for image
        :param image_size: tuple, (w, h) of origin image .
        :param pieces: tuple, (pw, ph), how many pieces to split image.
        :param sub_image_size: tuple, (sw, sh), size of sub image.
        :param overlap: tuple, (ow, oh), overlap of two sub neighbour image.
        :return: np.ndarray, (x1, y1, x2, y2), shape=(ph, pw, 4), corner of sub images.
        """
        assert (pieces is None) ^ (sub_image_size is None), \
            '"pieces" and "sub_image_size" can only specified one of them. but got {} and {}'.format(
                pieces, sub_image_size)
        if sub_image_size is not None:
            return self.__get_sub_image_corners_by_size(image_size, sub_image_size, overlap)
        elif pieces is not None:
            return self.__get_sub_image_corners_by_pieces(image_size, pieces, overlap)

    def get_sub_image_mask(self, corner: np.ndarray, polygon_points: np.ndarray):
        """
            cut function for annotation
        :param corner: sub image's corner
        :param annos: annotations in origin image
        :param annos_area: annotations's area in origin image
        :param keep_overlap: IOU(annotation in sub image, annotation in origin image) > keep_overlap will keep, or ignore.
        :return:
            annos: annos in sub_image
            annos_id[keep]: corresponding annos id in origin image
        """
        polygon_points = polygon_points.copy()
        # Attention!!!!! clipPolygon should be defined by clockwise !!!
        clipPolygon = [[corner[0], corner[1]], [corner[2], corner[1]], [corner[2], corner[3]], [corner[0], corner[3]]]
        self.polygon(clipPolygon)
        sub_Poly = []
        for i in range(len(polygon_points)):
            subjectPolygon = polygon_points[i]
            self.polygon(subjectPolygon)
            clipped = self.clip(subjectPolygon, clipPolygon)
            if clipped is not None:
                clipped[:, 0] = clipped[:, 0] - corner[0]
                clipped[:, 1] = clipped[:, 1] - corner[1]

                sub_Poly.append(clipped)

        return np.array(sub_Poly)

    def get_sub_image_corner_and_mask(self, image_size, polygon_points: np.ndarray, pieces=None,
                                      sub_image_size=None, overlap=(0, 0)):
        corners = self.get_sub_image_corners(image_size, pieces, sub_image_size, overlap)
        corners = corners.reshape((-1, 4))
        if len(polygon_points) > 0:
            # polygon_area = cv2.contourArea(polygon_points)
            sub_polygon = []
            for corner in corners:
                sub_poly = self.get_sub_image_mask(corner, polygon_points)
                sub_polygon.append(sub_poly)

        else:
            sub_polygon = [np.array([]).reshape((0, 0, 2)) for _ in corners]
        return corners, sub_polygon

    def show_poly(self, sub_img, sub_poly):

        if len(sub_poly) != 0:
            for nb_poly in range(len(sub_poly)):
                pts = np.array(sub_poly[nb_poly]).astype(np.int)
                cv2.polylines(sub_img, [pts], True, (255, 255, 0), 1)

        cv2.imshow("img", sub_img)
        cv2.waitKey(0)


def save_json_as_sub_image(save_file, sub_width=1280, sub_height=860):
    with open(annotations, encoding="utf-8")as f:
        inf = json.load(f)
        img_list = list(inf.keys())
        img_base = r"../ATD-UT-PD-01/img"
        img_name = [inf[img_key]['filename'] for img_key in inf.keys()]
        bbox = {}
        sub_inf = {"sub_images": [], "images": []}
        for idx in range(len(inf)):
            bbox[idx] = []
        split_img = SplitImage()
        image_ids = 0

        for i in range(10):
            for p in range(len(inf[img_list[i]]['regions'])):
                assert len(inf[img_list[i]]['regions'][p]['shape_attributes']['all_points_x']) == len(
                    inf[img_list[i]]['regions'][p]['shape_attributes']['all_points_y'])
                temp_row = []
                length = len(inf[img_list[i]]['regions'][p]['shape_attributes']['all_points_x'])
                for point in range(length):
                    x = inf[img_list[i]]['regions'][p]['shape_attributes']['all_points_x'][point]
                    y = inf[img_list[i]]['regions'][p]['shape_attributes']['all_points_y'][point]
                    temp_row.append([x, y])
                bbox[i].append(temp_row)
            img = cv2.imread(os.path.join(img_base, img_name[i]))
            img_h, img_w = img.shape[:2]

            corners, sub_polygon = split_img.get_sub_image_corner_and_mask((img_w, img_h),
                                                                           np.array(bbox[i]),
                                                                           None, (sub_width, sub_height), (128, 86))
            sub_inf["images"].append({"filename": img_name[i], "width": img_w,
                                      "height": img_h,
                                      "id": i, "polygon": bbox[i]})
            for n in range(len(corners)):
                #sub_img = img[corners[n][1]:corners[n][3], corners[n][0]:corners[n][2]]
                sub_poly = sub_polygon[n]
                sub_inf["sub_images"].append(
                    {"filename": img_name[i], "width": corners[n][2] - corners[n][0],
                     "height": corners[n][3] - corners[n][1],
                     "sub_id": image_ids, "id": i, "corner": corners[n], "polygon": list(sub_poly)})

                image_ids += 1
                #split_img.show_poly(sub_img, sub_poly)
    with open(save_file, 'w') as result_file:
        json.dump(sub_inf, result_file, cls=NpEncoder)


if __name__ == '__main__':
    annotations = r"./pic10.json"
    save_file = r'./Sub_text.json'
    save_json_as_sub_image(save_file)
    # new_inf["annotations"].append(
    #     {'segmentation': [x1, y1, x1, y2, x2, y2, x2, y1], 'bbox': [x1, y1, w, h], 'category_id': [],
    #      'area': [], 'is_crowd': [], 'image_id': [], 'id': [], 'ignore': [], 'uncertain': [],
    #      'logo': [], 'in_dense_image': [], 'size': []})
    # for idx in range(len(sub_annos[n])):
    #     cv2.rectangle(sub_img, (int(sub_annos[n][idx][0]), int(sub_annos[n][idx][1])),
    #                   (int(sub_annos[n][idx][2]), int(sub_annos[n][idx][3])), (0, 255, 0))
    #     # print(corners[i])
    #     # print(sub_annos[i])
    #     # print("\n")
    #     sub_img = img[corners[n][1]:corners[n][3], corners[n][0]:corners[n][2]]
    #     print(sub_img.shape)
    #
    #
    #     if len(sub_annos[n]) > 0:
    #         for idx in range(len(sub_annos[n])):
    #             cv2.rectangle(sub_img, (int(sub_annos[n][idx][0]), int(sub_annos[n][idx][1])),
    #                           (int(sub_annos[n][idx][2]), int(sub_annos[n][idx][3])), (0, 255, 0))
    #
    #     cv2.imshow("img", sub_img)
    #     cv2.waitKey(0)
