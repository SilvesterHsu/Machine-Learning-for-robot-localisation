import delimited /mnt/disk0/maps/michigan/eval/eval_08-21_16:25/eval_output.txt, delimiter(whitespace) 
drop if status==0
label define localization_status -1 "success" 1 "failure" 
graph box elapsed, over(status) ytitle(Localization time [s]) ylabel(30(2)0, angle(default)) title(Localization time) caption(Maximum  50s) legend(on)



