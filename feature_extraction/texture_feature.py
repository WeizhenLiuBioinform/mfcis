import homcloud.interface as hc


def compute_texture_pd(pict, save_path, paralles=2):
    pd = hc.PDList.from_bitmap_levelset(pict, "sublevel", "bitmap", parallels=2, save_to=save_path)
    return pd


def dump_pd_as_txt(source_file_path, dest_pd0_path, dest_pd1_path):
    pds = hc.PDList(source_file_path)
    pd0 = pds.dth_diagram(0)
    pd0_pairs = list(zip(pd0.births, pd0.deaths))
    pd1 = pds.dth_diagram(1)
    pd1_pairs = list(zip(pd1.births, pd1.deaths))






