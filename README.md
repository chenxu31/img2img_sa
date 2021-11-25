# code

## preprocessing
download code to Lian_project_code/
 
## train img2img_sa
cd MUNIT
python train_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --log_dir {LOG_DIR} --checkpoint_dir {CHECKPOINT_DIR} --logfile {LOGFILE_NAME}

## test img2img_sa
python test_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --output_dir {OUTPUT_DIR}
