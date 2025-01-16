#!/bin/bash
CONFIG=$1
GPU=$2
times_one_config(){
	 startTime=`date +%Y%m%d-%H:%M:%S`
         startTime_s=`date +%s`
         echo "the run command: $1"
         $1
         #sleep 5
         endTime=`date +%Y%m%d-%H:%M:%S`
	 endTime_s=`date +%s`
         sumTime=$[ $endTime_s - $startTime_s ]
         echo "$startTime ---> $endTime" "Total:$sumTime seconds"
	 hour=$(( $sumTime/3600 ))
         min=$(( ($sumTime-${hour}*3600)/60 ))
         sec=$(( $sumTime-${hour}*3600-${min}*60 ))

         echo "The config run time is ${hour}:${min}: ${sec}"
         #echo "The config run time is $(($sumTime/3600))h $(($sumTime%3600))min  $(($sumTime%3600))s"
	}

#times_one_config "python ./src/main.py --cfg $CONFIG --datamodule.data_dir ../../datasets/ --trainer.gpus $GPU --trainer.accelerator ddp"
times_one_config "python ./src/main.py --cfg $CONFIG --datamodule.data_dir ../../datasets/ --trainer.gpus $GPU"
cat $CONFIG
