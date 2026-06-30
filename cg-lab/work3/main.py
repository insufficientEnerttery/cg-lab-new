import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

mode = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])

def de_casteljau(points, t):
    n = len(points)
    if n == 1:
        return points[0]
    new_points = [np.array((1 - t) * points[i] + t * points[i + 1], dtype=np.float32) for i in range(n - 1)]
    return de_casteljau(new_points, t)

def b_spline(points, t):
    n = len(points)
    if n < 4:
        return None
    i = int(t * (n - 3))
    i = min(max(i, 0), n - 4)
    t_local = t * (n - 3) - i
    basis_matrix = np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0]
    ]) / 6.0
    segment = np.array(points[i:i+4])
    t_vec = np.array([t_local**3, t_local**2, t_local, 1])
    return t_vec @ basis_matrix @ segment

def main():
    window = ti.ui.Window("Bezier & B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    mode[None] = 0

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(np.array([pos[0], pos[1]], dtype=np.float32))
            elif e.key == 'c':
                control_points.clear()
            elif e.key == 'b':
                mode[None] = 1 - mode[None]

        clear_pixels()

        current_count = len(control_points)
        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                if mode[None] == 0:
                    curve_points_np[i] = de_casteljau(control_points, t)
                else:
                    res = b_spline(control_points, t)
                    curve_points_np[i] = res if res is not None else [0,0]
            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        #控制点
        if current_count > 0:
            gui_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            gui_np[:current_count] = control_points
            gui_points.from_numpy(gui_np)
            canvas.circles(gui_points, radius=0.005, color=(1, 0, 0))

            #连线
            if current_count >= 2:
                indices = []
                for i in range(current_count - 1):
                    indices.append(i)
                    indices.append(i+1)
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                np_indices[:len(indices)] = indices
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(1,1,1))

        window.show()

if __name__ == '__main__':
    main()