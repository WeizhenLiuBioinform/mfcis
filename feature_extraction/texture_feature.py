import homcloud.interface as hc
import os
import numpy as np


def compute_texture_pd(pict, name, save_path, paralles=2):
    dest_pd0_path = os.path.join(save_path, name+'_texture_pd0.txt')
    dest_pd1_path = os.path.join(save_path, name+'_texture_pd1.txt')
    if os.path.exists(dest_pd1_path) and os.path.exists(dest_pd1_path):
        return None
    else:
        pd = hc.PDList.from_bitmap_levelset(pict, "sublevel", "bitmap", parallels=2)
        dump_pd_as_txt(pd, dest_pd0_path, dest_pd1_path)
        return pd


def dump_pd_as_txt(pd, dest_pd0_path, dest_pd1_path):
    pds = pd
    pd0 = pds.dth_diagram(0)
    pd0_pairs = list(zip(pd0.births, pd0.deaths))
    pd1 = pds.dth_diagram(1)
    pd1_pairs = list(zip(pd1.births, pd1.deaths))
    np.savetxt(dest_pd0_path, pd0_pairs)
    np.savetxt(dest_pd1_path, pd1_pairs)







