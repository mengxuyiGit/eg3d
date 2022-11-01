CUDA_LAUNCH_BLOCKING=1

python gen_videos_condition.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=2x2 \
    --cfg ABO \
    --pointcloud_files "['/home/xuyi/Data/renderer/output_abo/B07B4G5YWH/sample/**.csv']" \
    --pose_file /home/xuyi/Data/renderer/output_abo/B07DBJX741/render/transforms.json \
    --network=/home/xuyi/Repo/eg3d/eg3d/pretrained_models/conditional-patchD-z-from-pc-abo-128-completed-white-gpus4-batch16-003440.pkl

#  --pointcloud_files "['/home/xuyi/Data/renderer/output_abo/B07QFP4M23/sample/pc.csv']" \
# --pointcloud_files "['/home/xuyi/Data/renderer/output_abo/B07B4D48Q5/sample/pc_rgbedit.csv',
#                         '/home/xuyi/Data/renderer/output_abo/B07B4D48Q5/sample/pc.csv',
#                         '/home/xuyi/Data/renderer/output_abo/B07B4D48Q5/sample/pc_green.csv',
#                         '/home/xuyi/Data/renderer/output_abo/B07B4D48Q5/sample/pc_rgbedit_green.csv']"

# python gen_videos_c2w.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=2x2 \
#     --cfg ABO --pointcloud_files "['/home/xuyi/Data/renderer/output_abo/B07QJJKZL2/sample/pc.csv']" \
#     --pose_file /home/xuyi/Data/renderer/output_abo/B07DBJX741/render/transforms.json \
#     --network=/home/xuyi/Repo/eg3d/eg3d/pretrained_models/network-snapshot-synunet-000800.pkl

# python gen_videos.py --outdir=out --trunc=0.7 --seeds=0-3 --grid=2x2 \
#     --cfg ABO --pointcloud_files "['/home/xuyi/Data/renderer/output_abo/B07JPGPBL2/sample/pc.csv']" \
#     --network=/home/xuyi/Repo/eg3d/eg3d/pretrained_models/network-snapshot-synunet-000600.pkl
#     # --network=/home/xuyi/Repo/eg3d/eg3d/pretrained_models/network-snapshot-original-abo-001600.pkl
#     # 

# # 'B07B4MHTG1', 'B07B4MF6P2', 'B07RTZ54B1', 'B075X4F3Z2', 'B07JL5QBC2', 'B075YPKYM1', 'B073NZGLT1', 'B07DYK2Y61'
# B07JXF7251: 花枕头
# B07JPGPBL2：红椅子 
# B07QFP4M23: brown sofa
# B07QJJKZL2: blue blanket

#### not-so-good cases 
# B07QHKQMY4L red cup
