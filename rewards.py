import json
import os

os.chdir('./Reward_weights')

metrics = ['v_degraded', 
           'v1_not_sent_from_s1', 
           'v3_not_sent_from_s3', 
           'v_not_found_in_last_station', 
           'z1_VSAV_sent', 
           'rupture_ff']

for m in metrics:

    dic_tarif_sent_disp = {'v_required': 0,
                    'v_sent': 0,
                    'v_sent_full':0,
                    'v_degraded':0,
                    'cancelled':0, #cancel departure
                    'function_not_found':0,
                    'v1_not_sent_from_s1':0,
                    'v3_not_sent_from_s3':0,
                    'v_not_found_in_last_station':0,
                    'ff_required':0,
                    'ff_sent':0,
                    'rupture_ff':0,       
                    'z1_VSAV_sent': 0,
                    'z1_FPT_sent': 0,
                    'z1_EPA_sent': 0,
                     'VSAV_needed':0,
                     'FPT_needed':0,
                     'EPA_needed':0,
                     'VSAV_disp':0,
                     'FPT_disp':0,
                     'EPA_disp':0,
                    'skill_lvl':0
                    } 

    dic_tarif_sent_disp[m] = -100

    if m == 'v_degraded':
        
        dic_tarif_sent_disp['v_sent_full'] = 10


    with open(f"rw_"+ m +".json", "w") as f:
        json.dump(dic_tarif_sent_disp, f)

print("rewards generated")

os.chdir("../")