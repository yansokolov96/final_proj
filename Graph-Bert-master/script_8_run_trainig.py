import script_0_delete_random_links
import script_1_preprocess
import script_2_pre_train
import script_3_fine_tuning
import script_5_load_run_data

for i in range(8, 11):
    print(i)
    script_0_delete_random_links.run_script_0()
    script_1_preprocess.run_script_1()
    script_2_pre_train.run_script_2()
    script_3_fine_tuning.run_script_3()
    script_5_load_run_data.run_script_5(i)
