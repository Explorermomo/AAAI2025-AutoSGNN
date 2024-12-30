#! /bin/bash

LLM_generate_num=12
pop_generation_num=20   
conditate_pop_num=30

LLM_generating='/home/jing/MO/AutoSGNN/EoH-main/examples/GNN_layer_generation/runEoH.py'
training_testing='/home/jing/MO/AutoSGNN/EoH-main/examples/GNN_layer_generation/filter/evaluation/src/train_model_raw.py'
pops_concat='/home/jing/MO/AutoSGNN/EoH-main/examples/GNN_layer_generation/filter/evaluation/src/pops_concat.py'

TIMEOUT_DURATION=600
NUM_GPU=4  
PIDS=()
	
export PYTHONPATH="/home/jing/MO/AutoSGNN/EoH-main:$PYTHONPATH"

pyfile_running() {
	local pop_index=$1
	local gpu_id=$2

	CUDA_VISIBLE_DEVICES=$gpu_id timeout $TIMEOUT_DURATION /home/jing/anaconda3/envs/H_GFM/bin/python $training_testing --index $pop_index  &

	PIDS+=($!)
}


main() {

	# Builting Initial Individuals
	source_file="/home/jing/MO/AutoSGNN/EoH-main/examples/result/Initial_Individuals/DenseSplit_PubMed_Initial.json"
	destination_file="/home/jing/MO/AutoSGNN/EoH-main/examples/result/LLM_generation/conditate_population/conditate_pops.json"
	cp "$source_file" "$destination_file"

	sum_llm=0
	start_time=$(date +%s.%N)
	# /home/jing/anaconda3/envs/H_GFM/bin/python /home/jing/MO/the_5th_paper_AutoSGNN/filter/GPRGNN-master/src/project_start_time.py
	for j in $(seq 1 ${pop_generation_num}); do
		cd /home/jing/MO/AutoSGNN/EoH-main
		start_time1=$(date +%s.%N)
		/home/jing/anaconda3/envs/H_GFM/bin/python $LLM_generating &
		wait
		echo $j
		end_time1=$(date +%s.%N)
		sum_llm=$(echo "$sum_llm + $end_time1 - $start_time1" | bc)
		echo "运行时间：$sum_llm 秒"

		cd /home/jing/MO/AutoSGNN/EoH-main/examples/GNN_layer_generation/filter/evaluation

		for i in $(seq 0 ${LLM_generate_num}); do
			gpu_id=$(( (i) % ${NUM_GPU} ))
			echo ${i}
			pyfile_running $i $gpu_id
			if [ $(( (i+1) % ${NUM_GPU} )) -eq 0 ] || [ $i -eq $(($LLM_generate_num - 1)) ]; then
                        	wait "${PIDS[@]}"
				PIDS=()
                	fi
	        done
	        wait "${PIDS[@]}"
            /home/jing/anaconda3/envs/H_GFM/bin/python $pops_concat --conditate_pop_num $conditate_pop_num &
		wait
	done
	end_time=$(date +%s.%N)
	elapsed_time=$(echo "$end_time - $start_time" | bc)
	echo "运行时间：$sum_llm 秒"
	echo "运行时间：$elapsed_time 秒"
}


main


