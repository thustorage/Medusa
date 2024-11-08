mkdir breakdowns

mount -o size=16G -t tmpfs none /mnt/memfs/

mkdir /mnt/memfs/model_tokenizers
cp -r /home/zsx/vllm/model_tokenizers/* /mnt/memfs/model_tokenizers/

mkdir /mnt/memfs/data/
mkdir /mnt/memfs/data/Qwen-0.5B
mkdir /mnt/memfs/data/Qwen-1.8B
mkdir /mnt/memfs/data/Qwen-4B
mkdir /mnt/memfs/data/Qwen-7B
mkdir /mnt/memfs/data/Yi-6B
mkdir /mnt/memfs/data/Llama-7B
mkdir /mnt/memfs/data/Falcon-7B
mkdir /mnt/memfs/data/Yi-9B
mkdir /mnt/memfs/data/Llama-13B
mkdir /mnt/memfs/data/Qwen-14B

ln -s /mnt/memfs/data /home/zsx/vllm/data

# cp -r /home/zsx/vllm/data-backup/* /mnt/memfs/data/