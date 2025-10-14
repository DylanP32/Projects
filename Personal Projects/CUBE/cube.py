import numpy as np
import time
from time import sleep
import sys
import os
import math


# setup initial display buffers, geometry, and symbols
def init_display(rows, cols):
    
    screen = np.full(shape=(rows, cols), fill_value=' ', dtype='<U1')
    buffer = np.full(shape=(3 + rows * (cols + 1)), fill_value=' ', dtype='<U1')
    symbols = ['$', '$', '=', '=', '+', '+', '-', '-', '@', '@', '*', '*']
    
    vertices = np.array([
        [-1, -1, -1],
        [-1,  1, -1],
        [ 1,  1, -1],
        [ 1, -1, -1],
        [ 1,  1,  1],
        [ 1, -1,  1],
        [-1, -1,  1],
        [-1,  1,  1],
    ], dtype=np.float32)
    
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [3, 2, 4],
        [3, 4, 5],
        [5, 4, 7],
        [5, 7, 6],
        [6, 7, 1],
        [6, 1, 0],
        [6, 0, 3],
        [6, 3, 5],
        [1, 7, 4],
        [1, 4, 2],
    ], dtype=np.uint8)
    
    camera_vector = [0, 0, 1]
    
    return screen, buffer, symbols, vertices, triangles, camera_vector


# set up timing for target fps
def set_target_fps(fps):
    
    prev_time = time.perf_counter_ns()
    target_delta = 1_000_000_000 // fps
    
    return prev_time, target_delta

# measure delta time and control frame rate
def get_delta_time(prev_time, target_delta):
    
    now = time.perf_counter_ns()
    delta_time = now - prev_time
    sleep_time = target_delta - delta_time
    
    if sleep_time > 1_000_000:
        sleep(sleep_time / 1_000_000_000.0)
    
    # get time after sleep
    now = time.perf_counter_ns()
    final_delta = now - (prev_time - delta_time)
    
    return final_delta / 1_000_000_000.0, now

# print frame to terminal
def display_screen(screen, buffer, rows, cols):
    
    for r in range(rows):
        offset = 3 + r * (cols + 1)
        
        buffer[offset : offset + cols] = screen[r, :]
        buffer[offset + cols] = "\n"
        
    frame_string = "".join(buffer)
    sys.stdout.write(frame_string)
    sys.stdout.flush()


# 3d math helper functions
def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def cross_product(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])

# project 3d point to 2d screen
def project(v, cols, rows):
    return (round(v[0]/v[2] + cols/2), round(v[1]/v[2] + rows/2))


# fill a horizontal line between two x points
def draw_scan_line(screen, y, x0, x1, symbol, rows, cols):
    
    left = min(x0, x1)
    right = max(x0, x1)
    y = int(y)
    
    if y < 0 or y >= rows:
        return
    
    left = max(0, left)
    right = min(cols - 1, right)
    
    if left <= right:
        screen[y, left:right+1] = symbol

# fill upper half of triangle
def draw_flat_top(screen, t0, t1, b, symbol, rows, cols):
    
    x_b = t0[0]
    x_e = t1[0]
    
    x_inc_0 = (b[0] - t0[0]) / (b[1] - t0[1])
    x_inc_1 = (b[0] - t1[0]) / (b[1] - t1[1])
    
    y_b = int(t0[1])
    y_e = int(b[1]+1)
    
    for y in range(y_b, y_e):
        draw_scan_line(screen, y, round(x_b), round(x_e), symbol, rows, cols)
        
        x_b += x_inc_0
        x_e += x_inc_1

# fill lower half of triangle
def draw_flat_bottom(screen, t, b0, b1, symbol, rows, cols):
    
    x_b = t[0]
    x_e = t[0]
    
    x_de_0 = (t[0] - b0[0]) / (b0[1] - t[1])
    x_de_1 = (t[0] - b1[0]) / (b1[1] - t[1])
    
    y_b = int(t[1])
    y_e = int(b0[1]+1)
    
    for y in range(y_b, y_e):
        draw_scan_line(screen, y, round(x_b), round(x_e), symbol, rows, cols)
        
        x_b -= x_de_0
        x_e -= x_de_1

# split and draw a full triangle
def draw_triangle(screen, vec0, vec1, vec2, symbol, rows, cols):
    
    v0, v1, v2 = vec0, vec1, vec2
    
    if v0[1] > v1[1]:
        v0, v1 = v1, v0
    if v1[1] > v2[1]:
        v1, v2 = v2, v1
    if v0[1] > v1[1]:
        v0, v1 = v1, v0
    if v2[1] == v1[1]:
        draw_flat_bottom(screen, v0, v1, v2, symbol, rows, cols)
        return
    if v0[1] == v1[1]:
        draw_flat_top(screen, v0, v1, v2, symbol, rows, cols)
        return
    
    midpoint = (v0[0] + (v2[0]-v0[0])*(v1[1]-v0[1]) / (v2[1]-v0[1]), v1[1])
    draw_flat_bottom(screen, v0, v1, midpoint, symbol, rows, cols)
    draw_flat_top(screen, v1, midpoint, v2, symbol, rows, cols)

# transform, project, then render all cube faces
def draw_cube(screen, vertices, triangles, camera_vector, symbols, rx, ry, rz, rows, cols):
    
    # cache trig values
    cosx, sinx = math.cos(rx), math.sin(rx)
    cosy, siny = math.cos(ry), math.sin(ry)
    cosz, sinz = math.cos(rz), math.sin(rz)
    
    for i, t in enumerate(triangles):
        transformed_vertices = np.zeros(shape=(3, 3))
        
        # transform image
        for j in range(3):
            v = vertices[t[j]]

            # rotate points using rotation matrices
            
            # rotate x
            y = cosx * v[1] - sinx * v[2]
            z = sinx * v[1] + cosx * v[2]
            v = (v[0], y, z)
            
            # rotate y
            x = cosy * v[0] + siny * v[2]
            z = -siny * v[0] + cosy * v[2]
            v = (x, v[1], z)
            
            # rotate z
            x = cosz * v[0] - sinz * v[1]
            y = sinz * v[0] + cosz * v[1]
            v = (x, y, v[2])

            # position and scale
            v = list(v)
            v[2] += 8
            scale = 100
            v[1] *= scale
            v[0] *= scale * 2
            
            transformed_vertices[j] = v
        
        # back-face culling
        v_01 = (transformed_vertices[1][0] - transformed_vertices[0][0], 
                transformed_vertices[1][1] - transformed_vertices[0][1],
                transformed_vertices[1][2] - transformed_vertices[0][2])
        v_02 = (transformed_vertices[2][0] - transformed_vertices[0][0], 
                transformed_vertices[2][1] - transformed_vertices[0][1],
                transformed_vertices[2][2] - transformed_vertices[0][2])
        normal = cross_product(v_01, v_02)
        
        if dot_product(camera_vector, normal) >= 0:
            continue
        
        projected_points = np.zeros(shape=(3, 2))
        
        # convert 3d to 2d
        for j in range(3):
            projected_points[j] = project(transformed_vertices[j], cols, rows)
        
        draw_triangle(screen, projected_points[0], projected_points[1], projected_points[2], symbols[i], rows, cols)

# main loop
def main():
    
    # display setup
    
    size = os.get_terminal_size()
    TERM_COLS = size.columns
    TERM_ROWS = size.lines
    
    screen, buffer, symbols, vertices, triangles, camera_vector = init_display(TERM_ROWS, TERM_COLS)
    prev_time, target_delta = set_target_fps(60) # change 60 to whatever refresh rate wanted
    
    sys.stdout.write("\u001b[2J\u001b[?25l")
    sys.stdout.flush()
    buffer[0], buffer[1], buffer[2] = "\u001b", "[", "H"
    
    rx = ry = rz = 0
    
    try:
        # rotate and render cube continuously
        while True:
            delta_time, prev_time = get_delta_time(prev_time, target_delta)
            screen.fill(' ')
            
            draw_cube(screen, vertices, triangles, camera_vector, symbols, rx, ry, rz, TERM_ROWS, TERM_COLS)
            display_screen(screen, buffer, TERM_ROWS, TERM_COLS)
            
            # rotation angle change
            rx = (rx + 0.5 * delta_time) % (2 * math.pi)
            ry = (ry + 0.5 * delta_time) % (2 * math.pi)
            rz = (rz + 0.5 * delta_time) % (2 * math.pi)
            
    finally:
        sys.stdout.write("\u001b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()