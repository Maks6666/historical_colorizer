import os

def choose_file(root_dir):
    root_dir = "grayscaled_images"
    gray_list = os.listdir(root_dir)

    for i, obj in enumerate(gray_list):
        print(f"{i + 1}: {obj}")

    idx = int(input("Choose a file: "))
    path = os.path.join(root_dir, gray_list[idx-1])
    return path

