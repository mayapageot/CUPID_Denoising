import os

import numpy as np
def write_data_stream(file_path, data: np.ndarray) -> None:
    data = (((data + 10.24) * (2 ** 23) / 10.24).astype(np.uint32) << 8)
    with open(file_path, "wb") as f:
        data.tofile(f)
    print("Derivative data saved to:", file_path)
# for ch in [3,5]:
#     dir = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{ch}"
#     for file in os.listdir(dir):
#         if "bin" in file and "diff" not in file:
#             if "corrected" in file:
#                 file_path = os.path.join(dir,file)
#                 bin_file = np.fromfile(file_path, dtype=np.uint32, offset=4)
#             else :
#                 file_path = os.path.join(dir, file)
#                 bin_file = np.fromfile(file_path, dtype=np.uint32)
#             bin_file = ((np.right_shift(bin_file, 8).astype(np.float32) / (2 ** 23)) * 10.24 - 10.24).astype(np.float32)
#             bin_file_dif = np.diff(bin_file)
#             if not os.path.isdir(os.path.join(dir,"dif")):
#                 os.mkdir(os.path.join(dir,"dif"))
#             write_data_stream(os.path.join(dir,"dif",file[:-4]+"_dif.bin"),bin_file_dif)

for ch in [3,5]:
    dir_dif = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{ch}"
    dir_raw = f"/feynman/work/projets/cupid/data/LSM/RUN9/000813_20230628T161508_{ch:03}_000.bin"
    bin_file = np.fromfile(dir_raw, dtype=np.uint32, offset=4)
    bin_file = ((np.right_shift(bin_file, 8).astype(np.float32) / (2 ** 23)) * 10.24 - 10.24).astype(np.float32)
    bin_file_dif = np.diff(bin_file)[:len(bin_file)//4]
    write_data_stream(os.path.join(dir_dif, "dif", "raw_noise_diff.bin"), bin_file_dif)