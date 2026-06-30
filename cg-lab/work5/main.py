import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

# 窗口分辨率与图像缓存
width, height = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

# 交互参数（控制光源位置和最大弹射次数）
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

@ti.kernel
def init_params():
    light_pos[None] = ti.Vector([2.0, 4.0, 3.0])
    max_bounces[None] = 3

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

# --- 1. 场景隐式几何体求交 ---

@ti.func
def intersect_plane(ro, rd):
    """任务1：无限大平面求交 (y = -1.0)"""
    t = -1.0
    if ti.abs(rd.y) > 1e-5:
        t = (-1.0 - ro.y) / rd.y
    return t

@ti.func
def get_plane_color(p):
    """任务1：黑白棋盘格纹理计算"""
    # 通过交点 x 和 z 坐标的奇偶性来判断颜色
    scale = 1.0
    chk = ti.floor(p.x * scale) + ti.floor(p.z * scale)
    color = ti.Vector([0.9, 0.9, 0.9]) # 白色格子
    if int(chk) % 2 == 0:
        color = ti.Vector([0.1, 0.1, 0.1]) # 黑色格子
    return color

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交测试"""
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4 * a * c
    t = -1.0
    if discriminant >= 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        if t < 1e-4:
            t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
    return t

# --- 2. 核心硬阴影判定 ---
@ti.func
def is_in_shadow(p, n):
    """任务3：硬阴影判定与避免自相交 Bug"""
    l_pos = light_pos[None]
    shadow_dir = normalize(l_pos - p)
    # 核心避坑点 (Shadow Acne)：沿法线方向向外偏移 1e-4
    shadow_origin = p + n * 1e-4 
    dist_to_light = (l_pos - p).norm()
    
    in_shadow = False
    # 检查是否撞击地面
    t_p = intersect_plane(shadow_origin, shadow_dir)
    if 1e-4 < t_p < dist_to_light:
        in_shadow = True
    # 检查是否撞击左侧红球
    if not in_shadow:
        t_s1 = intersect_sphere(shadow_origin, shadow_dir, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
        if 1e-4 < t_s1 < dist_to_light:
            in_shadow = True
    # 检查是否撞击右侧银色镜面球
    if not in_shadow:
        t_s2 = intersect_sphere(shadow_origin, shadow_dir, ti.Vector([1.5, 0.0, 0.0]), 1.0)
        if 1e-4 < t_s2 < dist_to_light:
            in_shadow = True
            
    return in_shadow

# --- 3. 基于迭代的光线追踪器 ---

@ti.kernel
def render():
    # 场景物体定义
    sphere1_center = ti.Vector([-1.5, 0.0, 0.0]) # 左侧红色球
    sphere2_center = ti.Vector([1.5, 0.0, 0.0])  # 右侧银色镜面球
    
    for i, j in pixels:
        screen_x = (i - width / 2.0) / height * 2.0
        screen_y = (j - height / 2.0) / height * 2.0
        
        # 初始化光线起点和方向
        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([screen_x, screen_y, -1.0]))
        
        # 任务2：初始化光线吞吐量（衰减系数）与最终颜色值
        throughput = ti.Vector([1.0, 1.0, 1.0])
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        # 开始光线多次弹射迭代循环
        for bounce in range(max_bounces[None]):
            min_t = 1e9
            hit_obj = 0 # 材质 ID: 0-未击中, 1-地面, 2-红球, 3-镜面球
            
            hit_p = ti.Vector([0.0, 0.0, 0.0])
            hit_n = ti.Vector([0.0, 0.0, 0.0])
            
            # 深度测试：寻找最近几何体
            t_plane = intersect_plane(ro, rd)
            if 1e-4 < t_plane < min_t:
                min_t = t_plane
                hit_obj = 1
                hit_p = ro + rd * t_plane
                hit_n = ti.Vector([0.0, 1.0, 0.0]) # 地面法线朝上
                
            t_s1 = intersect_sphere(ro, rd, sphere1_center, 1.0)
            if 1e-4 < t_s1 < min_t:
                min_t = t_s1
                hit_obj = 2
                hit_p = ro + rd * t_s1
                hit_n = normalize(hit_p - sphere1_center)
                
            t_s2 = intersect_sphere(ro, rd, sphere2_center, 1.0)
            if 1e-4 < t_s2 < min_t:
                min_t = t_s2
                hit_obj = 3
                hit_p = ro + rd * t_s2
                hit_n = normalize(hit_p - sphere2_center)
                
            # 如果没有击中任何物体，直接融入背景色并终止弹射
            if hit_obj == 0:
                final_color += throughput * ti.Vector([0.05, 0.15, 0.15])
                break
                
            # 双面法线修正
            if hit_n.dot(rd) > 0:
                hit_n = -hit_n
                
            # --- 材质交互分支判断 ---
            if hit_obj == 1 or hit_obj == 2:
                # 【漫反射物体】：计算当前点的局部光照，然后直接 break 终止
                base_col = get_plane_color(hit_p) if hit_obj == 1 else ti.Vector([0.8, 0.1, 0.1])
                
                # 计算基本的环境光和硬阴影漫反射
                ambient = 0.2 * base_col
                diffuse = ti.Vector([0.0, 0.0, 0.0])
                
                if not is_in_shadow(hit_p, hit_n):
                    L = normalize(light_pos[None] - hit_p)
                    diffuse = 0.8 * ti.max(0.0, hit_n.dot(L)) * base_col
                    
                # 累加颜色：当前贡献 = 局部光照 * 之前的光线衰减系数
                final_color += throughput * (ambient + diffuse)
                break # 漫反射物体不继续弹射光线，直接跳出
                
            elif hit_obj == 3:
                # 【纯镜面反射物体】：更新光线起点和方向，乘上反射率后继续下一次循环弹射
                # 核心避坑点 (Shadow Acne)：沿法线方向向外偏移 1e-4 重新发射
                ro = hit_p + hit_n * 1e-4
                rd = normalize(rd - 2.0 * rd.dot(hit_n) * hit_n) # 计算反射射线方向
                
                # 更新光线衰减系数 (每次乘以反射率 0.8)
                throughput *= 0.8 
                
        pixels[i, j] = tm.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (width, height))
    canvas = window.get_canvas()
    gui = window.get_gui()
    init_params()
    
    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.65, 0.05, 0.3, 0.25):
            # 任务4：UI 面板控制光源位置和弹射次数
            light_pos[None].x = gui.slider_float("Light X", light_pos[None].x, -5.0, 5.0)
            light_pos[None].y = gui.slider_float("Light Y", light_pos[None].y, 1.0, 8.0)
            light_pos[None].z = gui.slider_float("Light Z", light_pos[None].z, -5.0, 5.0)
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
            
        window.show()

if __name__ == "__main__":
    main()