import os
import os.path as osp
num_car = 0
num_plane = 0

path = "data/UCAS_AOD/Annotations"

with open('data/UCAS_AOD/ImageSets/trainval.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        img_id = line.strip()
        txtpath = osp.join(
            ("/").join(path.split("/")[:-2]),
            "UCAS_AOD/Annotations",
            "{}.txt".format(img_id),
        )
        with open(txtpath, "r") as f:
            lines = f.readlines()
            splitlines = [
                x.strip().replace("\t", " ").replace("  ", " ").split(" ")
                for x in lines
            ]
            for i, splitline in enumerate(splitlines):
                if splitline[0] == "car":
                    num_car += 1
                elif splitline[0] == "airplane":
                    num_plane += 1

print("num_car ",num_car, "\nnum_plane ",num_plane)