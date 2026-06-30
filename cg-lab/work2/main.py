import taichi as ti
import math

ti.init(arch=ti.cpu)

width, height = 700, 700
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

@ti.func
def get_model_matrix(angle: ti.f32):
    """
    获取模型变换矩阵（绕Z轴旋转）
    """
    # 将角度转换为弧度（三角函数使用弧度制）
    rad = angle * 3.1415926535 / 180.0
    cos_r = ti.cos(rad)
    sin_r = ti.sin(rad)
    # 绕Z轴旋转的模型矩阵
    return ti.Matrix([
        [cos_r, -sin_r, 0.0, 0.0],
        [sin_r,  cos_r, 0.0, 0.0],
        [  0.0,    0.0, 1.0, 0.0],
        [  0.0,    0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos_x: ti.f32, eye_pos_y: ti.f32, eye_pos_z: ti.f32):
    """
    获取视图变换矩阵（将相机平移到原点）
    """
    # 将相机位置移到原点
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos_x],
        [0.0, 1.0, 0.0, -eye_pos_y],
        [0.0, 0.0, 1.0, -eye_pos_z],
        [0.0, 0.0, 0.0,        1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """
    获取透视投影矩阵

    """
    # 将视场角转换为弧度
    fov_rad = eye_fov * 3.1415926535 / 180.0
        # zNear和zFar是正值距离，实际坐标中为负值
    n = -zNear  # 近截面
    f = -zFar   # 远截面
    
    # 计算透视平截头体的边界
    # t = tan(fov/2) * |n|，使用zNear的绝对值
    t = ti.tan(fov_rad / 2.0) * zNear
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 透视到正交矩阵（将透视平截头体挤压为正交长方体）
    M_p2o = ti.Matrix([
        [n,   0.0, 0.0,   0.0],
        [0.0, n,   0.0,   0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0,   0.0]
    ])
    
    # 正交投影矩阵（将正交长方体映射到NDC立方体[-1,1]^3）
    M_ortho = ti.Matrix([
        [2.0 / (r - l), 0.0,           0.0,            0.0],
        [0.0,           2.0 / (t - b), 0.0,            0.0],
        [0.0,           0.0,           2.0 / (n - f),  -(n + f) / (n - f)],
        [0.0,           0.0,           0.0,            1.0]
    ])
    
    # 透视投影矩阵 = 正交投影矩阵 @ 透视到正交矩阵
    return M_ortho @ M_p2o

@ti.func
def draw_line(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    dx = x1 - x0
    dy = y1 - y0
    # 计算需要绘制的步数
    steps = ti.max(ti.abs(dx), ti.abs(dy))
    if steps > 0:
        # 计算每一步的增量
        x_inc = dx / steps
        y_inc = dy / steps
        x = ti.f32(x0)
        y = ti.f32(y0)
        for _ in range(ti.i32(steps) + 1):
            px = ti.i32(x)
            py = ti.i32(y)
            if 0 <= px < width and 0 <= py < height:
                pixels[px, py] = ti.Vector([r, g, b])
            x += x_inc
            y += y_inc

@ti.kernel
def render(angle: ti.f32, eye_pos_x: ti.f32, eye_pos_y: ti.f32, eye_pos_z: ti.f32):
    """
    绘制3D三角形的投影

    """
    # 清空屏幕为黑色
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])
    
    # 定义三角形的三个顶点
    v0 = ti.Vector([ 2.0, 0.0, -2.0, 1.0])
    v1 = ti.Vector([ 0.0, 2.0, -2.0, 1.0])
    v2 = ti.Vector([-2.0, 0.0, -2.0, 1.0])
    
    # 计算MVP矩阵
    # MVP = 投影矩阵 @ 视图矩阵 @ 模型矩阵
    M = get_model_matrix(angle)
    V = get_view_matrix(eye_pos_x, eye_pos_y, eye_pos_z)
    P = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    MVP = P @ V @ M
    
    # 对每个顶点应用MVP变换
    v0_trans = MVP @ v0
    v1_trans = MVP @ v1
    v2_trans = MVP @ v2
    
    # 透视除法
    v0_ndc = ti.Vector([v0_trans.x / v0_trans.w, v0_trans.y / v0_trans.w, v0_trans.z / v0_trans.w])
    v1_ndc = ti.Vector([v1_trans.x / v1_trans.w, v1_trans.y / v1_trans.w, v1_trans.z / v1_trans.w])
    v2_ndc = ti.Vector([v2_trans.x / v2_trans.w, v2_trans.y / v2_trans.w, v2_trans.z / v2_trans.w])
    
    # 将NDC坐标映射到屏幕坐标
    screen_x0 = ti.i32((v0_ndc.x + 1.0) * 0.5 * width)
    screen_y0 = ti.i32((v0_ndc.y + 1.0) * 0.5 * height)
    screen_x1 = ti.i32((v1_ndc.x + 1.0) * 0.5 * width)
    screen_y1 = ti.i32((v1_ndc.y + 1.0) * 0.5 * height)
    screen_x2 = ti.i32((v2_ndc.x + 1.0) * 0.5 * width)
    screen_y2 = ti.i32((v2_ndc.y + 1.0) * 0.5 * height)
    
    # 三角形的三条边
    draw_line(screen_x0, screen_y0, screen_x1, screen_y1, 0.0, 1.0, 0.0)
    draw_line(screen_x1, screen_y1, screen_x2, screen_y2, 0.0, 1.0, 0.0)
    draw_line(screen_x2, screen_y2, screen_x0, screen_y0, 0.0, 1.0, 0.0)

def main():
    gui = ti.GUI("Experiment 1", res=(width, height))
    angle = 0.0
    eye_pos = [0.0, 0.0, 5.0]

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == gui.ESCAPE:
                break
        
        if gui.is_pressed('a'):
            angle += 2.0
        if gui.is_pressed('d'):
            angle -= 2.0
            
        render(angle, eye_pos[0], eye_pos[1], eye_pos[2])
        # 更新GUI显示
        gui.set_image(pixels)
        gui.show()

if __name__ == '__main__':
    main()