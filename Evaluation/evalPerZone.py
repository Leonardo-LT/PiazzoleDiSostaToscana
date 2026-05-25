def pointToZone(px, py, bbox, numRows, numCols):
    min_x, min_y, max_x, max_y = bbox

    if not (min_x <= px <= max_x and min_y <= py <= max_y):
        return None

    width = (max_x - min_x) / numCols
    height = (max_y - min_y) / numRows

    colonna = int((px - min_x) / width)

    riga = int((max_y - py) / height)

    if colonna == numCols:
        colonna -= 1
    if riga == numCols:
        riga -= 1

    return riga, colonna
