pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

pip install accelerate
pip install transformers==4.45.2

conda activate omnigen

ln -s /raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python3 ./python3


sudo /raid/home/CAMCA/hj880/miniconda3/bin/conda run -n omnigen python try.py

sudo /raid/home/CAMCA/hj880/miniconda3/bin/conda run -n omnigen python try.py

your_script.py


/raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python -m pip install -U requests
/raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python -m pip uninstall -U mpmath
/raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python -m pip install --force-reinstall datasets
/raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python -m pip install huggingface-hub==0.25.0
/raid/home/CAMCA/hj880/miniconda3/bin/conda run -n omnigen python -c "import PIL; print(PIL.__file__)"

/raid/home/CAMCA/hj880/miniconda3/envs/omnigen/bin/python -m pip install --force-reinstall diffusers==0.30.3

cd /raid/home/CAMCA/hj880/wt/code/gpu_train
./python3 ./gpu_train_sp.py --gpu-ids 2 3 --target-percentage 0.95