# 我的数据集处理方法
import os
import pickle as pk
import argparse

def create_image_path_dict(data_path, subset):
    hr_path = os.path.join(data_path, subset, 'HR')
    lq_path = os.path.join(data_path, subset, 'LQ')

    hr_images = [os.path.join(hr_path, img) for img in os.listdir(hr_path) if img.endswith('.jpg') or img.endswith('.png')]
    lq_images = [os.path.join(lq_path, img) for img in os.listdir(lq_path) if img.endswith('.jpg') or img.endswith('.png')]

    image_path_dict = {}
    for index, (hr_img, lq_img) in enumerate(zip(hr_images, lq_images)):
        image_path_dict[index] = [hr_img, lq_img]  # 使用索引作为键

    return image_path_dict

def create_train_splits_mydataset(data_path):
    train_dict = create_image_path_dict(data_path, 'train')
    val_dict = create_image_path_dict(data_path, 'val')
    test_dict = create_image_path_dict(data_path, 'test')

    os.makedirs('pkl_files', exist_ok=True)

    with open('pkl_files/mydataset_train.pkl', 'wb') as f:
        pk.dump(train_dict, f)
    with open('pkl_files/mydataset_val.pkl', 'wb') as f:
        pk.dump(val_dict, f)
    with open('pkl_files/mydataset_test.pkl', 'wb') as f:
        pk.dump(test_dict, f)

    print("train set: {}, val set: {}, test set: {}".format(len(train_dict), len(val_dict), len(test_dict)))

def main(args):
    if args.dataset =='mydataset':
        create_train_splits_mydataset(args.data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['mydataset'], help="the name of dataset")
    parser.add_argument("--data_path", type=str, help="dataset directory path")

    args = parser.parse_args()
    main(args)