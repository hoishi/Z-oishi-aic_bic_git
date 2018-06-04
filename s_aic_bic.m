function s_aic_bic

% calculates aic and bic of linear reggression models and geneate statistic profile of 
% The minimum aic or bic model is the best model explaining the object
% variable, behavior data (Here, stereoacuity).
% Then, check if the model fitting is signiticant using statistic profile. In
% parcicular, check if the pvalue of explanatory variables are significant 
% The function, fitlm needs matlab after version 2013 and Statistics and Machine Learning
% Toolbox.
% MODELVIF : VIFs (variance inflation factor) of the target model to check
% the multiple colinearity. When VIF > 10, there are multicollinearity.
% Hiroki Oishi 2018 0604

%% Load tractprofile e.g. MTV  The required format is 100 nodes*subjectnum
RVOF = load('Right_VOF_gradient2_AFQ330_median_qMRI.mat');
LVOF = load('Left_VOF_gradient2_AFQ330_median_qMRI.mat');
CFM = load('CFM_manual_gradient2_AFQ330_median_qMRI.mat');
LILF = load('LILF_gradient2_AFQ330_median_qMRI.mat');
RILF = load('RILF_gradient2_AFQ330_median_qMRI.mat');
LOR = load('LOR_top50000_AFQ330_qMRI.mat');
ROR = load('ROR_top50000_AFQ330_qMRI.mat');

%% Load behavior data The behavior data needs to be included in a row matrix.
load('threshold_84%correctrate_withcondition.mat');
usesbj = find(~isnan(value));% remove outlier subjects
stereoacuity = value(usesbj)';% object veariable here
n = length(stereoacuity);% subject num to be analyzed

% average tract profile (here, MTV) along a tract
usenodes = 11:90;%exclude the both side nodes
RVOF_mtv_dist_mean = mean(RVOF.mtv_dist(usenodes,usesbj,1))';
LVOF_mtv_dist_mean = mean(LVOF.mtv_dist(usenodes,usesbj,1))';
CFM_mtv_dist_mean = mean(CFM.mtv_dist(usenodes,usesbj,1))';
LILF_mtv_dist_mean = mean(LILF.mtv_dist(usenodes,usesbj,1))';
RILF_mtv_dist_mean = mean(RILF.mtv_dist(usenodes,usesbj,1))';
LOR_mtv_dist_mean = mean(LOR.mtv_dist(usenodes,usesbj,1))';
ROR_mtv_dist_mean = mean(ROR.mtv_dist(usenodes,usesbj,1))';
tracts = horzcat(RVOF_mtv_dist_mean, LVOF_mtv_dist_mean, CFM_mtv_dist_mean, LILF_mtv_dist_mean, RILF_mtv_dist_mean, LOR_mtv_dist_mean, ROR_mtv_dist_mean);

%% generate all of the possible regressor combinations of explanatory variables
regid = cell(size(tracts,2),1);
for i = 1:1:size(tracts,2)
    regid{i} = nchoosek(1:size(tracts,2),i);
end   

%% AIC and BIC calculation
alpha = 0.05;% significance level
for i = 1:length(regid)
    for j = 1:size(regid{i},1)
        [b,bint,R{i,j},rint,stats] = regress(stereoacuity,[ones(size(stereoacuity)) ,tracts(:,regid{i}(j,:))],alpha);
        p(i,j) = size(regid{i}(j,:),2);%tract num
        AIC(i,j) = n*(1 + log(2*pi*sum(R{i,j}.^2))/n) + 2*(p(i,j) + 2);
        BIC(i,j) = n*(1 + log(2*pi*sum(R{i,j}.^2))/n) +log(n)*p(i,j);
        R0=corrcoef([tracts(:,regid{i}(j,:))]);
        MODELVIF{i,j}=diag(inv(R0))';%Mmulticollinearity
    end
end
AIC(find(AIC==0)) = NaN;
AICminregid = find(AIC == min(min(AIC)));
BIC(find(BIC==0)) = NaN;
BICminregid = find(BIC == min(min(BIC)));
save('aicbicmat.mat','AIC','BIC', 'AICminregid', 'BICminregid', 'MODELVIF')%save AIC and BIC of all cominations and min combination

%% geneate statistic profile of the best linear reggresion model estimated by BIC
[explainnum,combination] = ind2sub(size(BIC), BICminregid);
md1 = fitlm(tracts(:,regid{explainnum}(combination,:)), stereoacuity,'linear','RobustOpts','off')
% md1 = fitlm(tracts(:,regid{explainnum}(combination,:)), stereoacuity,'linear','RobustOpts','off','VarNames',{'RVOF','stereoacuity'})


% 
% md1 = fitlm(tracts(:,1), stereoacuity,'linear','RobustOpts','off','VarNames',{'RVOF','stereoacuity'}) 
% md1 = fitlm(tracts(:,2), stereoacuity,'linear','RobustOpts','off','VarNames',{'LVOF','stereoacuity'})
% md1 = fitlm(tracts(:,3), stereoacuity,'linear','RobustOpts','off','VarNames',{'CFM','stereoacuity'})
% md1 = fitlm(tracts(:,4), stereoacuity,'linear','RobustOpts','off','VarNames',{'LILF','stereoacuity'})
% md1 = fitlm(tracts(:,5), stereoacuity,'linear','RobustOpts','off','VarNames',{'RILF','stereoacuity'})
% md1 = fitlm(tracts(:,6), stereoacuity,'linear','RobustOpts','off','VarNames',{'LOR','stereoacuity'})
% md1 = fitlm(tracts(:,7), stereoacuity,'linear','RobustOpts','off','VarNames',{'ROR','stereoacuity'})
% 
% md1 = fitlm(horzcat( tracts(:,1),tracts(:,4)), stereoacuity,'linear','RobustOpts','off','VarNames',{'RVOF','LILF','stereoacuity'})
% md1 = fitlm(tracts, stereoacuity,'linear','RobustOpts','off','VarNames',{'RVOF','LVOF','CFM','LILF','RILF','LOR','ROR','stereoacuity'})
% % save('aicbicmat.mat','AIC','BIC')
% 
keyboard;