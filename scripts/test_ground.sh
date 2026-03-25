python train.py --val --pretrain saved_models/your_model_droneaerial_model_best_0.xxxxxx.pth.tar --emb_size 512 --img_size 1024 --data_root /home/fhr/HAGeo/data --data_name CVOGL_SVI --savename test_model_svi --gpu 0 --batch_size 8 --num_workers 16 --print_freq 50

python train.py --test --pretrain saved_models/your_model_droneaerial_model_best_0.xxxxxx.pth.tar --emb_size 512 --img_size 1024 --data_root /home/fhr/HAGeo/data --data_name CVOGL_SVI --savename test_model_svi --gpu 0 --batch_size 8 --num_workers 16 --print_freq 50
