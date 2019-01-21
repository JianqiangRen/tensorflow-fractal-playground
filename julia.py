# coding=utf-8
# summary:
# author: xueluo
# date:

import tensorflow as tf
import numpy as np
import cv2

R = 4
ITER_NUM = 200


def is_pure(img):
    _img = img
    if len(img.shape) > 1:
        _img = img[:, :, 0]
    return np.max(_img) - np.min(_img) <= 1


def get_color(bg_ratio, ratio):
    def color(z, i):
        if abs(z) < R:
            return 0, 0, 0
        v = np.log2(i + R) / 3
        
        if v < 1.0:
            return v ** bg_ratio[0], v ** bg_ratio[1], v ** bg_ratio[2]
        else:
            v = max(0, 2 - v)
            return v ** ratio[0], v ** ratio[1], v ** ratio[2]
    
    return color


def gen_julia(Z, c, bg_ratio, ratio, zs_exp, xs_exp):
    xs = tf.constant(np.full(shape=Z.shape, fill_value=c, dtype=Z.dtype))
    zs = tf.Variable(Z)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))
    with tf.Session():
        tf.global_variables_initializer().run()
        zs_ = tf.where(tf.abs(zs) < R, zs ** zs_exp + xs ** xs_exp, zs)
        not_diverged = tf.abs(zs_) < R
        step = tf.group(
            zs.assign(zs_),
            ns.assign_add(tf.cast(not_diverged, tf.float32))
        )
        
        for i in range(ITER_NUM):
            step.run()
        final_step = ns.eval()
        final_z = zs_.eval()
    r, g, b = np.frompyfunc(get_color(bg_ratio, ratio), 2, 3)(final_z, final_step)
    
    r = r * 255
    g = g * 255
    b = b * 255
    
    v_max = 200
    
    r_flatten = r.flatten()
    r_idx = np.argwhere(r_flatten >= v_max)
    r_flatten_new = np.delete(r_flatten, r_idx.flatten())
    
    r_max = np.max(r_flatten_new)
    r[r >= v_max] = r_max
    
    g_flatten = g.flatten()
    g_idx = np.argwhere(g_flatten >= v_max)
    g_flatten_new = np.delete(g_flatten, g_idx.flatten())
    g_max = np.max(g_flatten_new)
    g[g >= v_max] = g_max
    
    b_flatten = b.flatten()
    b_idx = np.argwhere(b_flatten >= v_max)
    b_flatten_new = np.delete(b_flatten, b_idx.flatten())
    b_max = np.max(b_flatten_new)
    b[b >= v_max] = b_max
    
    img_array = np.dstack((r, g, b))
    return np.uint8(img_array)


if __name__ == '__main__':
    target_width = 1125
    target_height = 352
    
    for zs_exp in range(2, 6):
        for xs_exp in range(1, 4):
            for c_1 in [x * 0.2 + 1 for x in range(1, 30)]:
                for c_2 in [x * 0.2 + 1 for x in range(1, 30)]:
                    
                    start_x = -1.9  # x range
                    end_x = 1.9
                    start_y = -1.1  # y range
                    end_y = 1.1
                    width = 1200  # image width
                    # c = -0.835 - 0.2321 * 1j
                    c = c_1 + c_2 * 1j
                    bg_ratio = (1, 2.5, 4)
                    ratio = (0.9, 0.9, 0.9)
                    
                    step = (end_x - start_x) / width
                    Y, X = np.mgrid[start_y:end_y:step, start_x:end_x:step]
                    Z = X + 1j * Y
                    
                    img = gen_julia(Z, c, bg_ratio, ratio, zs_exp, xs_exp)
                    
                    if is_pure(img):
                        print("whole image is pure")
                        continue
                    
                    # img = cv2.blur(img,(3,3))
                    
                    w = np.shape(img)[1]
                    h = np.shape(img)[0]
                    
                    img_a = img[0: target_height, 0: target_width, :]
                    if not is_pure(img_a):
                        cv2.imwrite("julias/julia_{}_{}_{:.2f}_{:.2f}_a.png".format(zs_exp, xs_exp, c_1, c_2), img_a)
                    
                    img_b = img[100: 100 + target_height, 0:target_width, :]
                    if not is_pure(img_b):
                        cv2.imwrite("julias/julia_{}_{}_{:.2f}_{:.2f}_b.png".format(zs_exp, xs_exp, c_1, c_2), img_b)
                    
                    img_c = img[h - target_height: h, 0: target_width, :]
                    if not is_pure(img_c):
                        cv2.imwrite("julias/julia_{}_{}_{:.2f}_{:.2f}_c.png".format(zs_exp, xs_exp, c_1, c_2), img_c)
                    
                    print("{}_{}_{:.2f}_{:.2f}".format(zs_exp, xs_exp, c_1, c_2))