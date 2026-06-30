import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

# 窗口分辨率
width, height = 800, 600

# 全局材质参数
Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

# 图像存储
img = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

@ti.kernel
def init_params():
    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def intersect_sphere(origin, direction, center, radius):
    oc = origin - center
    a = direction.dot(direction)
    b = 2.0 * oc.dot(direction)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4 * a * c
    t = -1.0
    if discriminant >= 0:
        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
        if t < 1e-4:
            t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
    return t

@ti.func
def intersect_cone(origin, dir, apex, height, radius):
    k = radius / height
    k2 = k * k
    
    co = origin - apex
    a = dir.x * dir.x + dir.z * dir.z - k2 * dir.y * dir.y
    b = 2.0 * (co.x * dir.x + co.z * dir.z - k2 * co.y * dir.y)
    c = co.x * co.x + co.z * co.z - k2 * co.y * co.y
    
    t_res = -1.0
    disc = b * b - 4.0 * a * c
    if disc >= 0:
        sqrt_disc = ti.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        if t1 > t2: t1, t2 = t2, t1
            
        if t1 > 1e-4:
            y = co.y + dir.y * t1
            if -height <= y <= 0.0:
                t_res = t1
        if t_res < 0 and t2 > 1e-4:
            y = co.y + dir.y * t2
            if -height <= y <= 0.0:
                t_res = t2
    return t_res

@ti.func
def cone_normal(p, apex, height, radius):
    k = radius / height
    k2 = k * k
    p_local = p - apex
    n = normalize(ti.Vector([p_local.x, -k2 * p_local.y, p_local.z]))
    return n

@ti.func
def is_in_shadow(light_pos, p, normal, sphere_center, sphere_r, cone_apex, cone_h, cone_r):
    shadow_dir = normalize(light_pos - p)
    shadow_origin = p + normal * 3e-3  # 沿法线偏移避免自阴影
    dist_to_light = (light_pos - p).norm()
    
    in_shadow = False
    t_sphere = intersect_sphere(shadow_origin, shadow_dir, sphere_center, sphere_r)
    if 1e-4 < t_sphere < dist_to_light:
        in_shadow = True
        
    if not in_shadow:
        t_cone = intersect_cone(shadow_origin, shadow_dir, cone_apex, cone_h, cone_r)
        if 1e-4 < t_cone < dist_to_light:
            in_shadow = True
    return in_shadow

@ti.kernel
def render():
    # 严格对齐场景物体与光源参数
    light_pos = ti.Vector([2.0, 3.0, 4.0])
    view_pos = ti.Vector([0.0, 0.0, 5.0])
    
    sphere_center = ti.Vector([-1.2, -0.2, 0.0])
    sphere_r = 1.2
    
    cone_apex = ti.Vector([1.2, 1.2, 0.0]) 
    cone_h = 2.6 
    cone_r = 1.2
    
    # 颜色常数
    background = ti.Vector([0.05, 0.15, 0.15])
    sphere_col = ti.Vector([0.8, 0.1, 0.1])
    cone_col = ti.Vector([0.6, 0.2, 0.8])
    light_color = ti.Vector([1.0, 1.0, 1.0])

    for i, j in ti.ndrange(width, height):
        screen_x = (i - width / 2.0) / height * 2.0
        screen_y = (j - height / 2.0) / height * 2.0
        
        ray_origin = view_pos
        ray_dir = normalize(ti.Vector([screen_x, screen_y, -1.0]))
        
        final_col = background
        t_min = 1e9
        p = ti.Vector([0.0, 0.0, 0.0])
        n = ti.Vector([0.0, 0.0, 0.0])
        col = ti.Vector([0.0, 0.0, 0.0])
        
        # 深度测试 (Z-buffer 逻辑)
        t_sphere = intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_r)
        if 1e-4 < t_sphere < t_min:
            t_min = t_sphere
            p = ray_origin + ray_dir * t_sphere
            n = normalize(p - sphere_center)
            col = sphere_col
        
        t_cone = intersect_cone(ray_origin, ray_dir, cone_apex, cone_h, cone_r)
        if 1e-4 < t_cone < t_min:
            t_min = t_cone
            p = ray_origin + ray_dir * t_cone
            n = cone_normal(p, cone_apex, cone_h, cone_r)
            col = cone_col
            
        if t_min < 1e8:
            if n.dot(ray_dir) > 0:
                n = -n
            
            if is_in_shadow(light_pos, p, n, sphere_center, sphere_r, cone_apex, cone_h, cone_r):
                final_col = Ka[None] * light_color * col
            else:
                L = normalize(light_pos - p)
                V = normalize(view_pos - p)
                
                # 【已修复报错】标准 Phong 模型：由入射光反方向反射出来的 R 与视线 V 计算镜面高光
                R = tm.reflect(-L, n).normalized()
                
                ambient = Ka[None] * light_color * col
                diffuse = Kd[None] * ti.max(0.0, n.dot(L)) * light_color * col
                specular = Ks[None] * ti.pow(ti.max(0.0, R.dot(V)), shininess[None]) * light_color
                
                final_col = ambient + diffuse + specular
                
        img[i, j] = tm.clamp(final_col, 0.0, 1.0)

def main():
    window = ti.ui.Window("Custom Phong Shading System", (width, height))
    canvas = window.get_canvas()
    gui = window.get_gui()
    init_params()
    
    while window.running:
        render()
        canvas.set_image(img)
        
        with gui.sub_window("Material Parameters", 0.05, 0.05, 0.4, 0.35):
            Ka[None] = gui.slider_float("Ka (Ambient)", Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float("Kd (Diffuse)", Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float("Ks (Specular)", Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)
            if gui.button("Reset Default"):
                init_params()
                
        window.show()

if __name__ == "__main__":
    main()