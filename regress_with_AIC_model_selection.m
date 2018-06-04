function [B,BINT,R,RINT,STATS,AIC,MODEL,MODELVIF]=regress_with_AIC_model_selection(Y,X,alpha,AIC_method,use_DC,use_AICc)

% Conducts multiple regression with a model selection procedure based on Akaike-Information-Criterion.
% function [B,BINT,R,RINT,STATS,AIC,MODEL,MODELVIF]=regress_with_AIC_model_selection(Y,X,:alpha,:AIC_method,:use_DC,:use_AICc)
% (: is optional)
%
% Conducts multiple regression for the observation(Y) using model(X). Thus, Y=X*b
% Akaike Information Criterion (AIC) is also computed and enables the best model selection
% among all the possible models
%
% [input]
% Y     : dependent variable, [n(observation) x 1] matrix
% X     : independent (explanation) variables, [observation x num model], design matrix
% alpha : (optional) if specified, a 100*(1-ALPHA)% confidence level to
%         compute BINT, and a (100*ALPHA)% significance level to compute RINT
% AIC_method : (optional) method to calculate AIC, 'conventional' or 'step'
%              'conventional' by default
% use_DC     : add DC component to regressor [0/1], 1 by default
% use_AICc   : (optional) use AICc (AIC with a correction for finite sample sizes), [0/1]
%              0 by default
%
% [output]
% B     : regression coefficients in vector model Y=X*b
% BINT  : 95% confidence intervals for B
% R     : a vector R of residuals
% RINT  : a matrix RINT of intervals that can be used to diagnose outliers.
%         If RINT(i,:) does not contain zero, then the i-th residual is
%         larger than would be expected, at the 5% significance level.
%         This is evidence that the I-th observation is an outlier.
% STATS : a vector STATS containing, in the following order,
%         1. the R-square statistic,
%         2. the F statistic,
%         3. p value for the full model
%         4. an estimate of the error variance.
% AIC   : AIC (Akaike's Information Criteria) value,
%         You can select the best B, BINT, ... from the model with the smallest AIC
% MODEL : MODEL{ii,jj} is a model used for regression and corresponds to the AIC{ii,jj}
% MODELVIF : VIFs (variance inflation factor) of the target model to check the multiple colinearity.
%
% [note]
% all the output variables are cell structures
%
% [reference]
% Akaike, Hirotugu (1974). "A new look at the statistical model identification".
% IEEE Transactions on Automatic Control 19 (6): 716�E23. doi:10.1109/TAC.1974.1100705. MR0423716.
%
% Burnham, K. P., and Anderson, D.R. (2002).
% Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach, 2nd ed.
% Springer-Verlag. ISBN 0-387-95364-7. [This has over 10000 citations on Google Scholar.]
%
%
% Created    : "2011-06-17 09:53:54 banh"
% Last Update: "2017-09-04 20:07:57 ban"

% Oishi edit

% check input variables
if nargin<2, help(mfilename()); end
if nargin<3 || isempty(alpha), alpha=0.05; end
if nargin<4 || isempty(AIC_method), AIC_method='conventional'; end
if nargin<5 || isempty(use_DC), use_DC=1; end
if nargin<6 || isempty(use_AICc), use_AICc=0; end

if size(Y,1)~=size(X,1), error('mismatch: the size of Y & X. check input variable'); end
if ~strcmp(AIC_method,'conventional') && ~strcmp(AIC_method,'step'),
  error('AIC_method shoud be ''conventioanl'' or ''step''. check input variable');
end

% generate all of the possible regressor combinations
regid=cell(size(X,2),1);
for ii=1:1:size(X,2)
  regid{ii}=nchoosek(1:size(X,2),ii);
end

% initialize output variables
B     = cell(length(regid),size(X,2));
BINT  = cell(length(regid),size(X,2));
R     = cell(length(regid),size(X,2));
RINT  = cell(length(regid),size(X,2));
STATS = cell(length(regid),size(X,2));
AIC   = zeros(length(regid),size(X,2));
MODEL = cell(length(regid),size(X,2));
MODELVIF = cell(length(regid),size(X,2));

% DC component
if use_DC, DC=ones(size(Y,1),1); end

% conduct regressions and calculate AIC over all the possible models
for ii=1:1:length(regid)
  for jj=1:1:size(regid{ii},1)

    % check multiple colinearity by VIF
    R0=corrcoef([X(:,regid{ii}(jj,:))]);
    MODELVIF{ii,jj}=diag(inv(R0))';

    % multiple regression
    if use_DC
      [B{ii,jj},BINT{ii,jj},R{ii,jj},RINT{ii,jj},STATS{ii,jj}]=regress(Y,[X(:,regid{ii}(jj,:)),DC],alpha);
    else
      [B{ii,jj},BINT{ii,jj},R{ii,jj},RINT{ii,jj},STATS{ii,jj}]=regress(Y,X(:,regid{ii}(jj,:)),alpha);
    end

    % calculate AIC (Akaike Information Criterion)
    % In general linear model (multiple regressions), AICs are defined as
    %   AIC = n*(1+log(2*pi*sum(residual(fit).^2)/n))+2*(p+1) [conventional AIC]
    %   AIC = n*log(sum(resid(fit).^2)/n) + 2*p [step AIC]
    % here, n: number of observations, p: number of parameters
    if strcmp(AIC_method,'conventional') % conventional AIC
      if use_DC % in the case of the polynominal have constant term
        AIC(ii,jj)=size(Y,1)*(1+log(2*pi*sum(R{ii,jj}.^2)/size(Y,1)))+2*(numel(regid{ii}(jj,:))+1+1);
      else
        AIC(ii,jj)=size(Y,1)*(1+log(2*pi*sum(R{ii,jj}.^2)/size(Y,1)))+2*(numel(regid{ii}(jj,:))+1);
      end
    else % step AIC
      if use_DC
        AIC(ii,jj)=size(Y,1)*log(sum(R{ii,jj}.^2)/size(Y,1))+2*(numel(regid{ii}(jj,:))+1);
      else
        AIC(ii,jj)=size(Y,1)*log(sum(R{ii,jj}.^2)/size(Y,1))+2*numel(regid{ii}(jj,:));
      end
    end

    % calculate AICc (AIC with a correction for finite sample sizes)
    % ref: Burnham, K. P., and Anderson, D.R. (2002)
    % the paper above strongly recommend to use AICc instead of AIC as the initial criteria
    if use_AICc
      if use_DC
        AIC(ii,jj)=AIC(ii,jj)+2*(numel(regid{ii}(jj,:))+1)*((numel(regid{ii}(jj,:))+1)+1)/(size(Y,1)-(numel(regid{ii}(jj,:))+1)-1);
      else
        AIC(ii,jj)=AIC(ii,jj)+2*numel(regid{ii}(jj,:))*(numel(regid{ii}(jj,:))+1)/(size(Y,1)-numel(regid{ii}(jj,:))-1);
      end
    end

    % set used model
    MODEL{ii,jj}=regid{ii}(jj,:);

  end % for jj=1:1:size(regid{ii},1)
end % for ii=1:1:length(regid)
