import numpy as np
import matplotlib.pyplot as plt


# Функция для генерации случайной шероховатости поверхности
def generate_rough_surface(size_x, size_y, scale_x, scale_y):
    x = np.linspace(0, size_x, scale_x)
    y = np.linspace(0, size_y, scale_y)
    x, y = np.meshgrid(x, y)
    z = np.sin(x) * np.cos(y)
    return x, y, z


# Функция для расчета нормалей поверхности
def calculate_normals(x, y, z):
    dzdx, dzdy = np.gradient(z)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(z)
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= length
    ny /= length
    nz /= length
    return nx, ny, nz


# Функция для моделирования отражения света
def blinn_phong_reflection(nx, ny, nz, light_dir, view_dir, shininess):
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = view_dir / np.linalg.norm(view_dir)

    dot_nl = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0, 1)
    h = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    dot_nh = np.clip(nx * h[0] + ny * h[1] + nz * h[2], 0, 1)

    ambient = 0.1
    diffuse = 0.7 * dot_nl
    specular = 0.2 * (dot_nh ** shininess)

    return ambient + diffuse + specular


# Параметры
size_x = 10
size_y = 10
scale_x = 10
scale_y = 10
locate_light = [10, 5, 10]
locate_view = [1, 1, 5]
shininess = 32

light_dir = np.array(locate_light)
view_dir = np.array(locate_view)

# Генерация поверхности и нормалей
x, y, z = generate_rough_surface(size_x, size_y, scale_x, scale_y)
nx, ny, nz = calculate_normals(x, y, z)

# Расчет отражения
reflection = blinn_phong_reflection(nx, ny, nz, light_dir, view_dir, shininess)

# Визуализация
fig = plt.figure(figsize=(12, 6))

# Поверхность
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_title('Шероховатая поверхность')
ax1.scatter(*locate_light, color='red')
ax1.scatter(*locate_view, color='blue')

# Отражение
ax2 = fig.add_subplot(122)
ax2.imshow(reflection, extent=(0, scale_x, 0, scale_y), origin='lower')
ax2.set_title('Отражение света')

plt.show()
