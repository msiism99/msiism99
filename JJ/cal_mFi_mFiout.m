function [mFi num_mFi num_mFiout ] = cal_mFi_mFiout (num_s_mFi, edge_exclusion, shot_pitch, shot_shift,fig_yn, q_scale)
  
%% example

% num_s_mFi = resi_all_cpe_overlap; 
% fig_yn = 0; 
% q_scale = 3;
% edge_exclusion = 6;
% [mFi num_mFi num_mFiout ] = cal_mFi_mFiout (num_s_mFi, edge_exclusion, shot_pitch, shot_shift,fig_yn, q_scale);

%%

%     [field_num] = field_num_creat_rev1s (num_s_mFi, edge_exclusion, shot_pitch, shot_shift);
    [field_num] = field_num_creat_ocm (num_s_mFi, edge_exclusion, shot_pitch, shot_shift);
    
    mFi = mFi_creation_0t1_rev1(num_s_mFi, field_num);
    fig_name = strcat('');
    mFi_draw(mFi, shot_pitch, fig_name,q_scale, fig_yn);    
    num_mFi = [];
    for ff=1:size(mFi,1)
        temp_out_fbf = num_s_mFi(ismember(num_s_mFi(:,[7,8]),mFi(ff,[1,2]),'rows'),:);
        temp_sub = repmat(mFi(ff,[3,4]),size(temp_out_fbf,1),1);
        temp_out_fbf2 = temp_out_fbf;
        temp_out_fbf2(:,9:10) = temp_sub(:,1:2);
        num_mFi = [num_mFi;temp_out_fbf2];
    end

    num_s_mFi = sortrows(num_s_mFi, [1 -6 5 -8 7]);
    num_mFi = sortrows(num_mFi, [1 -6 5 -8 7]);    
    num_mFiout = num_s_mFi;
    num_mFiout(:,9:10) = num_s_mFi(:,9:10) - num_mFi(:,9:10);
%     resi_3s_cpe_mFiout = [3*std(num_mFiout(:,9)) 3*std(num_mFiout(:,10))];

    if fig_yn == 1;
        
        dataExpression = strcat('num_mFi'); 
        [raw_3s raw_nns] = drawWaferVector_addmean (num_mFi, dataExpression,field_num, shot_pitch, q_scale ); 

        dataExpression = strcat('num_mFiout'); 
        [raw_3s raw_nns] = drawWaferVector_addmean (num_mFiout, dataExpression,field_num, shot_pitch, q_scale ); 
        
    end
    

    
function mFi = mFi_creation_0t1_rev1(num_source_for_mFi, field_num)

    field_num_inner = field_num(field_num(:,5) == 1,:);

    %     wf_no = unique(num_source_for_mFi(:,1));
    %     waferPosition = num_source_for_mFi(num_source_for_mFi(:,1) == wf_no(1),:);
    %     num_source = DATA.num_raw_onefile_full;
        waferPosition = getWaferPosition(num_source_for_mFi, 1);  
        test_position_t = unique (waferPosition(:,[7:8 2]),'rows');
        test_position_t = sortrows(test_position_t, [-2 1]);
    %     mFi = Set_fingerprint_creation (num_source, waferPosition_t, test_position_t, DATA.field_num_fullsample);




        [waferAverage average_residual] = getWaferAverage(num_source_for_mFi,waferPosition);
        test_position_temp = sortrows(waferAverage(:,[7 8]), [-2 1]);
    %     test_position_temp2 = round(test_position_temp*10^5)/10^5;
        test_position = unique(test_position_temp, 'rows');    

    % mFi_source = waferAverage(...
    %     ismember(waferAverage(:,3), field_num(field_num(:,5) == 1,1)) & ...
    %     ismember(waferAverage(:,4), field_num(field_num(:,5) == 1,2)), :);

    mFi_source = waferAverage(...
        ismember(waferAverage(:,3:4), field_num_inner(:,1:2),'rows'), :);

    mFi = test_position;
    for m = 1: length(test_position);
        mFi(m, 3) = mean(...
            mFi_source(...
            abs(mFi_source(:, 7) - test_position(m, 1)) < 0.5 & ...
            abs(mFi_source(:, 8) - test_position(m, 2)) < 0.5, 9 ...
            )...
            );

        mFi(m, 4) = mean(...
            mFi_source(...
            abs(mFi_source(:, 7) - test_position(m, 1)) < 0.5 & ...
            abs(mFi_source(:, 8) - test_position(m, 2)) < 0.5, 10 ...
            )...
            );
    end

    mFi=sortrows(mFi, [-2 1]);
    if isequal(mFi(:,1:2), test_position_t(:,1:2))
        mFi(:,5) = test_position_t(:,3);
    else
        mFi(:,5) = 1:size(mFi,1);
    end
    mFi = sortrows(mFi, 5);

end

function [field_num_t] = field_num_creat_rev1s (num_raw_onefile, edge_exclusion, shot_pitch, shot_shift)

    % 1. inner/edge shot 구분 - 갯수 기준 (column5)

    field_num_t = unique(num_raw_onefile(:,3:6),'rows');
    field_num_t = sortrows(field_num_t,[1 -2]);

    test_position_t = unique(num_raw_onefile(:,[7:8 2]),'rows');
    [num_raw_onefile_full] = creat_fullsample_v5 (test_position_t, edge_exclusion, shot_pitch, shot_shift);

    for m=1:size(field_num_t,1)
        num_raw_onefile_fbf = ...
             num_raw_onefile_full(ismember(num_raw_onefile_full(:,3:4),field_num_t(m,1:2),'rows'),:); 
         if size(num_raw_onefile_fbf,1) == size(test_position_t,1)
             field_num_t(m,5) = 1; %key 전부 있으면 innershot
         else
             field_num_t(m,5) = 2; %key 하나라도 없으면 edgeshot
         end
    end

end

function [raw_3s raw_nns] = drawWaferVector_addmean (num_source, dataExpression,field_num, shot_size, q_scale )


    ma_data_for_plot = num_source;

    figure('color','w','name', strcat(dataExpression));

    ma_wafer_plot(ma_data_for_plot, q_scale, field_num, shot_size)
    title(strcat(dataExpression),'interpreter','none');
    text(150,-145, ['mean: ' [num2str(round(100*abs(mean(ma_data_for_plot(:,9))))/100) ' / ' num2str(round(100*abs(mean(ma_data_for_plot(:,10))))/100)] ],'HorizontalAlignment','right');
    source_nns = ma_data_for_plot; [nns] = nns_filter(source_nns);
    text(150,-160, ['3s: ' [num2str(round(100*3.*std(ma_data_for_plot(:,9)))/100) ' / ' num2str(round(100*3.*std(ma_data_for_plot(:,10)))/100)] ],'HorizontalAlignment','right');
    text(150,-175, ['m3s: ' [num2str(round(100*abs(mean(ma_data_for_plot(:,9)))+100*3.*std(ma_data_for_plot(:,9)))/100) ' / ' num2str(round(100*abs(mean(ma_data_for_plot(:,10)))+100*3.*std(ma_data_for_plot(:,10)))/100)] ],'HorizontalAlignment','right');
    source_nns = ma_data_for_plot; [nns] = nns_filter(source_nns);
    text(150,-190, ['nns: ' [num2str(round(100*nns(1, 1))/100) ' / ' num2str(round(100*nns(1, 2))/100)]],'HorizontalAlignment','right') ;

    raw_3s = [3.*std(ma_data_for_plot(:,9)) 3.*std(ma_data_for_plot(:,10))];
    raw_nns = nns_filter(ma_data_for_plot);
    
end
end
