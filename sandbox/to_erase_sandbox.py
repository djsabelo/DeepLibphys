import os

dataset_dir='/media/belo/Storage/owncloud/Research Projects/DeepLibphys/Current Trained/' + 'ECG_BIOMETRY[{0}.{1}]'.format(32, 512)
files = os.listdir(dataset_dir)

for file in files:
    file_name_array = [name if i != 3 else "M1" for i, name in enumerate(file.split("_"))]
    file_name = file_name_array[0]
    for name in file_name_array[1:]:
        file_name += "_" + name

    print("Renaming {0} to {1}".format(dataset_dir + "/" + file, dataset_dir + "/" + file_name))
    os.rename(src=dataset_dir + "/" + file, dst=dataset_dir + "/" + file_name)
