
scripts=(
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_uncontrolled_aird.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_uncontrolled_aird.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_salad.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_flames.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_fake_alignment.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_mm_safetybench.sh
    /mnt/shared-storage-user/zhangbo1/MBEF/scripts/submit_vlsbench.sh
)

yaml_files=(
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/uncontrolled_aird_exp1_actor_v01.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/uncontrolled_aird_exp2_saferlhf_simple_v01.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/salad_judge_v02_qwen1.5-0.5b.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/flames_judge_v01_qwen1.5-0.5b.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/fake_alignment_qwen-1.5-0.5b.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/mm_safetybench_v01.yaml
    /mnt/shared-storage-user/zhangbo1/MBEF/configs/eval_tasks/demo_vlsbench.yaml
)

for i in "${!scripts[@]}"; do
    echo "=========================================="
    echo "开始执行任务 $((i+1))/${#scripts[@]}: ${scripts[$i]}"
    echo "配置文件: ${yaml_files[$i]}"
    echo "=========================================="
    bash ${scripts[$i]} "${yaml_files[$i]}"
    sleep 300s
    if [ $? -ne 0 ]; then
        echo "错误: 任务 $((i+1)) 执行失败，中止后续任务"
        exit 1
    fi
    echo "任务 $((i+1)) 完成"
    echo ""
done

echo "所有任务执行完成！"