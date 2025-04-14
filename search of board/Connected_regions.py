
import numpy as np
def find_connected_regions(matrix, threshold,distance=90):
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    regions = []
    x0=0
    y0=0
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j]:
                value = matrix[i][j]
                mask = [[0] * cols for _ in range(rows)]
                queue = [(i, j)]
                visited[i][j] = True
                mask[i][j] = 1
                region_size = 1

                while queue:
                    x, y = queue.pop(0)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < rows and 0 <= ny < cols:
                            if not visited[nx][ny] and matrix[nx][ny] == value:
                                if matrix[nx][ny] == 1:
                                    if abs(nx-x0)+abs(ny-y0)<distance:
                                        visited[nx][ny] = True
                                        queue.append((nx, ny))
                                        mask[nx][ny] = 1
                                        region_size += 1

                                else:
                                    visited[nx][ny] = True
                                    queue.append((nx, ny))
                                    mask[nx][ny] = 1
                                    region_size += 1

                if region_size >= threshold:
                    regions.append(np.asarray(mask))

    return regions
