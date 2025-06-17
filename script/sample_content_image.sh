python sample.py \
    --ckpt_dir="/scratch/yl10337/FontDiffuser/outputs/FontDiffuser_MR_Aug/global_step_135000" \
    --content_image_path="data_examples/sampling/c.jpg" \
    --style_image_path="data_examples/sampling/s.jpg" \
    --save_image \
    --save_image_dir="outputs/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

# python sample.py \
#     --ckpt_dir="outputs/FontDiffuser_MR/global_step_40000" \
#     --content_dir="data_examples/train/sampled_output/content_img" \
#     --style_dir="data_examples/train/sampled_output/style_img" \
#     --save_image \
#     --save_image_dir="outputs/batch_output" \
#     --device="cuda:0" \
#     --algorithm_type="dpmsolver++" \
#     --guidance_type="classifier-free" \
#     --guidance_scale=7.5 \
#     --num_inference_steps=20 \
#     --method="multistep"