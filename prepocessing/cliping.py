from graphics import *

win = GraphWin("Sutherland-Hodgman_polygon_clipping", 500, 500)
win1 = GraphWin("Output", 500, 500)


def clip(subjectPolygon, clipPolygon):
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
    return (outputList)


def polygon(points, prop_1, prop_2, output, wait):
    points_1 = []
    for i in range(len(points)):
        pt = Point(points[i][0], points[i][1])
        points_1.append(pt)
    poly = Polygon(*points_1)
    # if prop_1 == 1:
    #     poly.setFill('Red')
    # if prop_2 == 1:
    #     poly.setWidth(2)
    # if output == 1:
    #     poly.draw(win1)
    # else:
    #     poly.draw(win)
    # if wait == 1:
    #     if output == 1:
    #         win1.getMouse()
    #     else:
    #         win.getMouse()


def clicked(num):
    list = []
    for i in range(num):
        pt = win.getMouse()
        q = pt.getX()
        w = pt.getY()
        e = (q, w)
        list.append(e)
        pt.draw(win)
    return list


# num_subjectPolygon = int(input("Enter number of points for number of Subject Polygon : "))
# subjectPolygon = clicked(num_subjectPolygon)
# polygon(subjectPolygon, 1, 0, 0, 0)
# num_clipPolygon = int(input("Enter number of pints for Clip Polygon : "))
# clipPolygon = clicked(num_clipPolygon)
# polygon(clipPolygon, 0, 1, 0, 1)
# clipped = clip(subjectPolygon, clipPolygon)
# polygon(clipPolygon, 0, 1, 1, 0)
# polygon(clipped, 1, 0, 1, 1)
subjectPolygon = [[918, 672], [1140, 678], [1135, 659], [1165, 662], [1171, 675], [1226, 674], [1222, 655], [1250, 660],
                  [1260, 675], [1261, 663], [1292, 668], [1297, 673], [1308, 672], [1303, 659], [1370, 660],
                  [1375, 670], [1382, 670], [1381, 657], [1430, 668], [1514, 668], [1511, 654], [1536, 658],
                  [1546, 669], [1647, 668], [1639, 654], [1668, 655], [1673, 667], [1767, 670], [1765, 653],
                  [1789, 659], [1797, 669], [2175, 673], [2167, 657], [2195, 658], [2200, 673], [2488, 678],
                  [2483, 657], [2508, 658], [2515, 677], [2650, 677], [2660, 702], [2649, 717], [2635, 707],
                  [2277, 703], [2266, 718], [2247, 717], [2237, 704], [2030, 700], [2026, 713], [1996, 707],
                  [1989, 697], [1858, 699], [1846, 714], [1815, 698], [1732, 698], [1722, 717], [1693, 713],
                  [1679, 697], [1481, 702], [1462, 712], [1428, 700], [1297, 707], [1144, 710], [989, 705], [928, 705]]

polygon(subjectPolygon, 1, 0, 0, 0)
clipPolygon = [[512, 640], [1024, 640],[1024, 1280],[512, 1280]]
polygon(clipPolygon, 1, 0, 0, 0)
clipped = clip(subjectPolygon, clipPolygon)
polygon(clipped, 1, 0, 1, 1)