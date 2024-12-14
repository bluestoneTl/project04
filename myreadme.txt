ZZCX
    -train
        -HR    HR和LQ的图片一一对应
        -LQ
    -test
        -HR
        -LQ
    -val
        -HR
        -LQ

数据集处理
python preprocess_mydataset.py --dataset mydataset --data_path /path/to/your/dataset

训练命令
train_favae.py --ds output --batch_size 2 --print_steps 1000 --img_steps 10000 
                        --codebook_size 2048 --disc_start_epochs 20 --embed_dim 256 --use_l2_quantizer --use_cosine_sim --num_groups 32 
                        --with_fcm --ffl_weight 1.0 --use_same_conv_gauss --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0 
                        --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.25 --base_lr 2.0e-6 
                        --train_file ffhq_train.pkl 
                        --test_file ../datasets/pkl_files/ffhq_test.pkl 
                        --val_jlib_file 

python train_favae.py --ds output --batch_size 2 --print_steps 1000 --img_steps 10000 
                        --codebook_size 2048 --disc_start_epochs 20 --embed_dim 256 --use_l2_quantizer --use_cosine_sim --num_groups 32 
                        --with_fcm --ffl_weight 1.0 --use_same_conv_gauss --DSL_weight_features 0.01 --gaussian_kernel 9 --dsl_init_sigma 3.0 
                        --codebook_weight 1.0 --perceptual_weight 1.0 --disc_weight 0.25 --base_lr 2.0e-6 
                        --train_file 
                        --test_file ../datasets/pkl_files/ffhq_test.pkl 
                        --val_jlib_file 