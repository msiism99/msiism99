function [K_HOPC_x K_HOPC_y K_iHOPC_x_ovo3 K_iHOPC_y_ovo3 K_cpe_input] = ADI_selfcorr (num_adi_target, hoc_mode, ihoc_mode, adi_trocs_mode, edge_exclusion, shot_pitch, shot_shift, die_mat, q_scale)

flag_osr = [0,0,0];
plot_title = '';
[fit_HOC_OSR_corr resi_HOC_OSR_corr K_HOPC_x K_HOPC_y K_iHOPC_x_ovo3 K_iHOPC_y_ovo3 K_HOPC_nm K_iHOPC_nm resi_3s] = Set_OSRvarorder_ovo3_rev6 ...
    (num_adi_target, hoc_mode, ihoc_mode, edge_exclusion, shot_pitch, shot_shift, flag_osr, q_scale, plot_title);    

num_s_mFi = resi_HOC_OSR_corr;
fig_yn_mfi = 0;
[mFi num_mFi num_mFiout ] = cal_mFi_mFiout (num_s_mFi, edge_exclusion, shot_pitch, shot_shift,fig_yn_mfi, q_scale);

G_opt_spec=1;
ovo3_yn=0;
if strmatch(adi_trocs_mode,'38par')
    ovo3_yn=1;
elseif strmatch(adi_trocs_mode,'33par')
    ovo3_yn=1;
end

trocs_source = num_mFiout;
mFiout_yn=1;
if ovo3_yn==1        
    [K_cpe_input] = Set_CPE_OCM_OVO3ridge_rev1 (trocs_source, adi_trocs_mode, edge_exclusion,die_mat,G_opt_spec, mFiout_yn);
    [fit_all_cpe resi_all_cpe] = Set_CPE_extK_ovo3_rev5 (trocs_source, K_cpe_input);
    [field_num_ovo3_gopt] = field_num_creat_rev1s (trocs_source, edge_exclusion, shot_pitch, shot_shift);        
else
    gopt_weight=0;            
    G_opt_except_dieNO=[];
    [field_num_ovo3_gopt K_cpe_input fit_all_cpe resi_all_cpe] = SetAll_CPE_varorder_ovo3_rev11 (trocs_source, adi_trocs_mode, edge_exclusion,die_mat,G_opt_spec, mFiout_yn, gopt_weight, G_opt_except_dieNO);                 
end
