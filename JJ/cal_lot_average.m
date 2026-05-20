function [num_raw_overlap_avg num_raw_overlap_avgout] = cal_lot_average(num_raw_overlap)

    clear waferPosition_t
    waferPosition_t(:,[2:8 11:12]) = unique(num_raw_overlap(:,[2:8 11:12]),'rows');
    waferPosition_t = sortrows(waferPosition_t, [1 -6 5 -8 7]);
    num_raw_overlap_avg = waferPosition_t;
    
    %Initalize waitbar
    h = waitbar(0,'Loading...',...
            'Name','Matlabvn_waitbar');
    steps = size(num_raw_overlap_avg,1);
    
    num_raw_overlap_avgout = [];
    for pp=1:size(num_raw_overlap_avg,1)
        clear num_temp;
        num_temp = num_raw_overlap(ismember(num_raw_overlap(:,11:12),num_raw_overlap_avg(pp,11:12),'rows'),:);
        num_raw_overlap_avg(pp,9:10) = mean(num_temp(:,9:10),1);
        for tt=1:size(num_temp,1)
            num_temp(tt,19:20) = num_temp(tt,9:10) - num_raw_overlap_avg(pp,9:10);
        end
        num_raw_overlap_avgout = [num_raw_overlap_avgout;num_temp];
        % Update waitbar
        waitbar(pp / steps,h,sprintf('Processing...%.2f%%',pp/steps*100));
    end
    %Delete waitbar
    delete(h)

    num_raw_overlap_avgout = sortrows(num_raw_overlap_avgout, [13 -6 5 -8 7]);
    num_raw_overlap_avgout(:,9:10) = num_raw_overlap_avgout(:,19:20);
