import taichi as ti

ti.init(arch=ti.gpu)  

# --- 严格对齐老师的物理与网格参数 ---
N = 20                         
W, H = N, N
mass = 1.0                     
dt = 5e-4                      
k_s = 10000.0                  
k_d = 5.0                      # 【修改这里：阻尼系数切换为 5.0】
gravity = ti.Vector([0.0, -9.8, 0.0])  
max_velocity = 50.0            

pixel_size = 512
num_particles = W * H

pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vel = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

current_method = ti.field(dtype=ti.i32, shape=())  
is_paused = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def init_particles():
    for i, j in ti.ndrange(W, H):
        idx = i * H + j
        pos[idx] = ti.Vector([
            (i / (W - 1) - 0.5) * 1.0,
            0.6,
            (j / (H - 1) - 0.5) * 1.0
        ])
        vel[idx] = ti.Vector([0.0, 0.0, 0.0])

def init_cloth():
    init_particles()
    current_method[None] = 1  
    is_paused[None] = 0

@ti.func
def compute_forces_on(idx, p_pos):
    # 使用修改后的 k_d = 5.0 计算真实的粘滞力
    f_total = gravity * mass + -k_d * vel[idx]
    
    i = idx // H
    j = idx % H
    for di, dj in ti.ndrange((-1, 2), (-1, 2)):
        if di == 0 and dj == 0:
            continue
        ni, nj = i + di, j + dj
        if 0 <= ni < W and 0 <= nj < H:
            n_idx = ni * H + nj
            r = pos[n_idx] - p_pos
            dist = r.norm()
            rest_len = ti.Vector([di / (W - 1), 0.0, dj / (H - 1)]).norm() * 1.0
            if dist > 1e-5:
                f_total += k_s * (dist - rest_len) * (r / dist)
    return f_total

@ti.func
def clamp_velocity(v):
    v_norm = v.norm()
    if v_norm > max_velocity:
        v = (v / v_norm) * max_velocity
    return v

@ti.kernel
def step_explicit():
    for idx in range(num_particles):
        if idx == 0 or idx == H - 1: continue
        f = compute_forces_on(idx, pos[idx])
        pos[idx] += vel[idx] * dt
        new_vel = vel[idx] + (f / mass) * dt
        vel[idx] = clamp_velocity(new_vel)

@ti.kernel
def step_semi_implicit():
    for idx in range(num_particles):
        if idx == 0 or idx == H - 1: continue
        f = compute_forces_on(idx, pos[idx])
        new_vel = vel[idx] + (f / mass) * dt
        vel[idx] = clamp_velocity(new_vel)
        pos[idx] += vel[idx] * dt

@ti.kernel
def step_implicit_iter():
    for idx in range(num_particles):
        if idx == 0 or idx == H - 1: continue
        predicted_pos = pos[idx] + vel[idx] * dt
        f_implicit = compute_forces_on(idx, predicted_pos)
        new_vel = vel[idx] + (f_implicit / mass) * dt
        vel[idx] = clamp_velocity(new_vel)
        pos[idx] += vel[idx] * dt

def update_physics():
    if not is_paused[None]:
        if current_method[None] == 0: step_explicit()
        elif current_method[None] == 1: step_semi_implicit()
        elif current_method[None] == 2: step_implicit_iter()

def main():
    init_cloth()
    window = ti.ui.Window("Mass Spring System (k_d = 5.0)", (pixel_size, pixel_size))
    canvas = window.get_canvas()
    scene = window.get_scene()
    
    camera = ti.ui.Camera()
    camera.position(0.0, 2.5, 1.5)  
    camera.lookat(0.0, 0.4, 0.0)
    
    while window.running:
        for _ in range(15):  
            update_physics()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(0.5, 10.0, 2.0), color=(1, 1, 1))
        scene.particles(pos, radius=0.015, color=(0.2, 0.6, 1.0))
        canvas.scene(scene)
        
        window.GUI.begin("Control Panel", 0.05, 0.05, 0.38, 0.28)
        if window.GUI.checkbox("Pause Simulation", is_paused[None]):
            is_paused[None] = 1 if is_paused[None] == 0 else 0
        if window.GUI.button("Reset Cloth"): init_cloth()
        window.GUI.text("Integration Method:")
        if window.GUI.button("Explicit Euler"): current_method[None] = 0
        if window.GUI.button("Semi-Implicit Euler"): current_method[None] = 1
        if window.GUI.button("Implicit Euler"): current_method[None] = 2
        method_names = ["Explicit", "Semi-Implicit", "Implicit"]
        window.GUI.text(f"Active Method: {method_names[current_method[None]]}")
        window.GUI.text(f"Damping Coeff (k_d): {k_d}")
        window.GUI.end()
        window.show()

if __name__ == "__main__":
    main()