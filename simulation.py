from vpython import *
import csv
from ball_data import *
from record_config import *

# 全局常量
dx              = 1.0
max_x_pos       = 10
max_x_neg       = 5
x_min           = -max_x_neg
x_max           = max_x_pos
k               = 9e9
dt              = 0.005
r_min           = 0.1
SIM_ORDER = MIRROR_ITER_ORDER
THEORY_ORDER = THEORY_MIRROR_ORDER

# 场景
scene = canvas(
    width=1600, height=700,
    center=vector(0, 1, 0),
    background=color.white,
    title="一维带电小球碰撞｜迭代镜像法（自身理论误差评估）"
)
time_label  = label(pos=vector(x_max + 0.5, 4.8, 0), text="时间：0.000 s", color=color.blue, height=22, box=False)
vel_label   = label(pos=vector(x_max + 0.5, 4.1, 0), text="vA=0.000 m/s  vB=0.000 m/s", color=color.green, height=22, box=False)
p_label     = label(pos=vector(x_max + 0.5, 3.4, 0), text="总动量=0.000 kg·m/s", color=color.orange, height=22, box=False)
ke_label    = label(pos=vector(x_max + 0.5, 2.7, 0), text="总动能=0.000 J", color=color.purple, height=22, box=False)
err_label   = label(pos=vector(x_max + 0.5, 2.0, 0), text="镜像法总能量相对截断误差：0.0000%", color=color.black, height=22, box=False)
f_label     = label(pos=vector(x_max + 0.5, 1.3, 0), text="A受力(模拟N阶镜像)：0.000 N", color=color.red, height=22, box=False)

collision_log_label = label(pos=vector(0, -1.3, 0), text="碰撞记录：", color=color.red, height=20, box=False, align="center")
# 坐标轴
x_axis = arrow(pos=vector(x_min, 0, 0), axis=vector(x_max - x_min, 0, 0), shaftwidth=0.03, color=color.black)
y_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 4, 0), shaftwidth=0.03, color=color.black)
for i in range(x_min, x_max + 1):
    label(pos=vector(i, -0.4, 0), text=f"x={i} m", box=False)
floor = box(pos=vector(0, -0.25, 0), size=vector(x_max - x_min, 0.5, 1), color=color.gray(0.8))

# 参数输入框
scene.caption = "\n\n\n\n"
scene.append_to_caption(" A质量(kg)：")
mA_input = winput(bind=lambda: None, text=str(BALL_A_MASS), width=70)
scene.append_to_caption(" B质量(kg)：")
mB_input = winput(bind=lambda: None, text=str(BALL_B_MASS), width=70)
scene.append_to_caption(" | A初始x(m)：")
posA_input = winput(bind=lambda: None, text=str(BALL_A_POS), width=70)
scene.append_to_caption(" B初始x(m)：")
posB_input = winput(bind=lambda: None, text=str(BALL_B_POS), width=70)
scene.append_to_caption(" | A初速度(m/s)：")
velA_input = winput(bind=lambda: None, text=str(BALL_A_VEL), width=70)
scene.append_to_caption(" B初速度(m/s)：")
velB_input = winput(bind=lambda: None, text=str(BALL_B_VEL), width=70)
scene.append_to_caption(" | A电量(μC)：")
qA_input = winput(bind=lambda: None, text=str(BALL_A_CHARGE * 1e6), width=70)
scene.append_to_caption(" B电量(μC)：")
qB_input = winput(bind=lambda: None, text=str(BALL_B_CHARGE * 1e6), width=70)
scene.append_to_caption("\n\n")

start_btn = button(bind=lambda: start(), text="▶ 开始")
pause_btn = button(bind=lambda: pause(), text="⏸ 暂停/继续")
reset_btn = button(bind=lambda: reset(), text=" 重置")

# 小球
ballA = sphere(pos=vector(BALL_A_POS, 0.2, 0), velocity=vector(BALL_A_VEL, 0, 0), radius=0.2, color=BALL_A_COLOR, make_trail=True)
ballB = sphere(pos=vector(BALL_B_POS, 0.2, 0), velocity=vector(BALL_B_VEL, 0, 0), radius=0.2, color=BALL_B_COLOR, make_trail=True)

# 全局状态
total_time      = 0
is_running      = False
is_paused       = False
collision_times = []
last_record_time = -999.0
first_collision_pos = None
has_collided = False
has_exchanged_charge = False

# 1. 原始点电荷势能（仅作对照）
def point_coulomb_ep(xa, xb, qa, qb):
    r = max(abs(xb - xa), r_min)
    return k * qa * qb / r

# 2. 通用镜像力/势能函数（可传入任意阶数）
def mirror_calc(xa, xb, qa, qb, xL, xR, order):
    L = xR - xL
    F_A = 0.0
    F_B = 0.0
    Ep = 0.0
    # 真实电荷互作用
    r0 = max(abs(xb - xa), r_min)
    f0 = k * qa * qb / r0**2
    ep0 = k * qa * qb / r0
    Ep += ep0
    if xb > xa:
        F_A -= f0
        F_B += f0
    else:
        F_A += f0
        F_B -= f0
    # 迭代镜像
    for n in range(1, order + 1):
        sign = (-1)**n
        # A镜像
        qaL = sign * qa
        xaL = xL - (xa - xL) + 2 * n * L
        qaR = sign * qa
        xaR = xR + (xR - xa) - 2 * n * L
        # B镜像
        qbL = sign * qb
        xbL = xL - (xb - xL) + 2 * n * L
        qbR = sign * qb
        xbR = xR + (xR - xb) - 2 * n * L
        # A与B镜像
        for qm, xm in [(qbL, xbL), (qbR, xbR)]:
            r = max(abs(xa - xm), r_min)
            f = k * qa * qm / r**2
            Ep += k * qa * qm / r
            F_A -= f if xm > xa else -f
        # B与A镜像
        for qm, xm in [(qaL, xaL), (qaR, xaR)]:
            r = max(abs(xb - xm), r_min)
            f = k * qb * qm / r**2
            Ep += k * qb * qm / r
            F_B -= f if xm > xb else -f
    return F_A, F_B, Ep

# CSV初始化（修复newline错误）
def init_csv():
    with open(SAVE_FILE_NAME, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "时间(s)",
            "A_x(m)", "A_v(m/s)", "A_Ek(J)", "B_x(m)", "B_v(m/s)", "B_Ek(J)",
            "模拟阶数N库仑力(N)", "模拟N阶电势能Ep_sim(J)", "理论高阶Ep_theory(J)",
            "模拟总机械能E_sim(J)", "理论总机械能E_theory(J)",
            "Ep绝对误差ΔEp(J)", "Ep相对误差δEp(%)",
            "E总绝对误差ΔE(J)", "E总相对误差δE(%)",
            "是否碰撞", "qA前(μC)", "qB前(μC)", "qA后(μC)", "qB后(μC)"
        ])

def save_data(t, xA, vA, xB, vB, mA, mB, F_sim, Ep_sim, Ep_theo, E_sim, E_theo,
              dEp, dEp_pct, dE, dE_pct, is_collide, qA_b, qB_b, qA_a, qB_a):
    ekA = 0.5 * mA * vA **2
    ekB = 0.5 * mB * vB **2
    flag = "Y" if is_collide else "N"
    dp = DECIMAL_PLACES
    dep = ERR_DECIMAL
    # 修复：newline=""，编码放encoding参数
    with open(SAVE_FILE_NAME, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([
            round(t, dp),
            round(xA, dp), round(vA, dp), round(ekA, dp), round(xB, dp), round(vB, dp), round(ekB, dp),
            round(F_sim, dp), round(Ep_sim, dp), round(Ep_theo, dp),
            round(E_sim, dp), round(E_theo, dp),
            round(dEp, dep), round(dEp_pct, dep), round(dE, dep), round(dE_pct, dep),
            flag, round(qA_b*1e6, dp), round(qB_b*1e6, dp), round(qA_a*1e6, dp), round(qB_a*1e6, dp)
        ])

init_csv()

# 控制函数
def start():
    global is_running, is_paused
    is_running = True
    is_paused = False

def pause():
    global is_paused
    if is_running:
        is_paused = not is_paused

def reset():
    global total_time, last_record_time, is_running, is_paused, collision_times, first_collision_pos, has_collided, has_exchanged_charge
    is_running = False
    is_paused = False
    total_time = 0
    last_record_time = -999.0
    collision_times = []
    first_collision_pos = None
    has_collided = False
    has_exchanged_charge = False
    collision_log_label.text = "碰撞记录："
    try:
        mA = float(mA_input.text)
        mB = float(mB_input.text)
        xA = float(posA_input.text)
        xB = float(posB_input.text)
        vA = float(velA_input.text)
        vB = float(velB_input.text)
        qA = float(qA_input.text) * 1e-6
        qB = float(qB_input.text) * 1e-6
    except:
        mA, mB = 1.0, 1.0
        xA, xB = 0.0, 2.0
        vA, vB = 0.5, 0.0
        qA, qB = 100e-6, -100e-6
    ballA.pos = vector(xA, 0.2, 0)
    ballB.pos = vector(xB, 0.2, 0)
    ballA.velocity = vector(vA, 0, 0)
    ballB.velocity = vector(vB, 0, 0)
    ballA.clear_trail()
    ballB.clear_trail()
    time_label.text = "时间：0.000 s"
    init_csv()

# 主循环
while True:
    rate(200)
    if not is_running or is_paused:
        continue
    try:
        mA = float(mA_input.text)
        mB = float(mB_input.text)
        qA = float(qA_input.text) * 1e-6
        qB = float(qB_input.text) * 1e-6
    except:
        mA, mB, qA, qB = 1.0, 1.0, 100e-6, -100e-6

    total_time += dt
    t = total_time
    xA = ballA.pos.x
    xB = ballB.pos.x
    vA = ballA.velocity.x
    vB = ballB.velocity.x
    d_sep = abs(xB - xA)
    collision_happened = False

    # 1. 你的仿真方法：N阶镜像
    F_sim, F_B_sim, Ep_sim = mirror_calc(xA, xB, qA, qB, x_min, x_max, SIM_ORDER)
    # 2. 理论真值：超高阶近似无穷镜像
    _, _, Ep_theory = mirror_calc(xA, xB, qA, qB, x_min, x_max, THEORY_ORDER)

    qA_before, qB_before = qA, qB
    qA_after, qB_after = qA, qB

    # 碰撞逻辑
    if d_sep < 0.4 and not has_collided:
        collision_happened = True
        has_collided = True
        vA_new = ((mA - mB) * vA + 2 * mB * vB) / (mA + mB)
        vB_new = (2 * mA * vA + (mB - mA) * vB) / (mA + mB)
        ballA.velocity.x = vA_new
        ballB.velocity.x = vB_new
        if not has_exchanged_charge:
            q_total = qA + qB
            qA_after = q_total / 2
            qB_after = q_total / 2
            qA = qA_after
            qB = qB_after
            qA_input.text = str(round(qA * 1e6, 3))
            qB_input.text = str(round(qB * 1e6, 3))
            has_exchanged_charge = True
        if first_collision_pos is None:
            first_collision_pos = sphere(pos=vector((xA+xB)/2,0.2,0), radius=0.08, color=color.red)
        collision_times.append(round(t,3))
        lines = ["碰撞记录："]
        line = ""
        for i, tm in enumerate(collision_times):
            line += f"t={tm:.3f}s  "
            if (i+1)%5==0:
                lines.append(line.strip())
                line = ""
        if line:
            lines.append(line.strip())
        collision_log_label.text = "\n".join(lines)
    else:
        ballA.velocity.x += F_sim / mA * dt
        ballB.velocity.x += F_B_sim / mB * dt

    # 更新位置
    ballA.pos += ballA.velocity * dt
    ballB.pos += ballB.velocity * dt
    new_d = abs(ballB.pos.x - ballA.pos.x)
    if new_d >= 0.4:
        has_collided = False
        has_exchanged_charge = False

    # 动能、总能量
    vA_curr = ballA.velocity.x
    vB_curr = ballB.velocity.x
    ekA = 0.5 * mA * vA_curr**2
    ekB = 0.5 * mB * vB_curr**2
    Ek_total = ekA + ekB
    E_sim = Ek_total + Ep_sim
    E_theory = Ek_total + Ep_theory

    # 计算【该镜像方法自身截断误差】（模拟值 VS 理论精确值）
    delta_Ep = abs(Ep_sim - Ep_theory)
    delta_E = abs(E_sim - E_theory)
    if abs(Ep_theory) < 1e-12:
        deltaEp_pct = 0.0
    else:
        deltaEp_pct = delta_Ep / abs(Ep_theory) * 100
    if abs(E_theory) < 1e-12:
        deltaE_pct = 0.0
    else:
        deltaE_pct = delta_E / abs(E_theory) * 100

    # 界面文字
    time_label.text = f"时间：{t:.3f} s"
    vel_label.text  = f"vA={vA_curr:.3f} m/s  vB={vB_curr:.3f} m/s"
    p_label.text    = f"总动量={(mA*vA_curr + mB*vB_curr):.3f} kg·m/s"
    ke_label.text   = f"总动能={Ek_total:.3f} J"
    err_label.text  = f"总能量相对截断误差：{deltaE_pct:.4f} %"
    f_label.text    = f"A受力(N阶镜像)：{F_sim:.3f} N"

    # 保存数据
    if t - last_record_time >= RECORD_INTERVAL - 1e-6:
        save_data(t, xA, vA_curr, xB, vB_curr, mA, mB, F_sim, Ep_sim, Ep_theory,
                  E_sim, E_theory, delta_Ep, deltaEp_pct, delta_E, deltaE_pct, False,
                  qA_before, qB_before, qA_after, qB_after)
        last_record_time = t
    if collision_happened:
        save_data(t, xA, vA_curr, xB, vB_curr, mA, mB, F_sim, Ep_sim, Ep_theory,
                  E_sim, E_theory, delta_Ep, deltaEp_pct, delta_E, deltaE_pct, True,
                  qA_before, qB_before, qA_after, qB_after)
        last_record_time = t

    # 边界停止
    if xA < x_min or xA > x_max or xB < x_min or xB > x_max:
        is_running = False
        time_label.text += " | 仿真停止"