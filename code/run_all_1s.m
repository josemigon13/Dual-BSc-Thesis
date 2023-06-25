%% Calibración

load('datos_lab_v3','X','Y','obs_l');
X2 = X;
obs_l2 = obs_l;
load 'datos_lab_v4' X obs_l
[C,IA,IB] = intersect(obs_l2,obs_l);
X = [X2(IA,:) X(IB,:)];
Xsc = X;
Y = Y(IA,:);
Ysc = Y;

% Compresión directa [1 semana]
Xc = padarray(X, [ceil(size(Xsc, 1)/10080)*10080-size(Xsc, 1), 0], 0, 'post');
Xc = squeeze(sum(reshape(Xc,10080,[], size(Xc, 2)), 1));

Yc = padarray(Y, [ceil(size(Ysc, 1)/10080)*10080-size(Ysc, 1), 0], 0, 'post');
Yc = squeeze(sum(reshape(Yc,10080,[], size(Yc, 2)), 1));

%% Test

load('datos_lab_v3','test','Yt');
test2 = test;
load 'datos_lab_v4' test
test = [test2 test];
testsc = test;
Ytsc = Yt;

% Compresión directa [1 semana]
testc = padarray(test, [ceil(size(testsc, 1)/10080)*10080-size(testsc, 1), 0], 0, 'post');
testc = squeeze(sum(reshape(testc,10080,[], size(testc, 2)), 1));

Ytc = padarray(Yt, [ceil(size(Ytsc, 1)/10080)*10080-size(Ytsc, 1), 0], 0, 'post');
Ytc = squeeze(sum(reshape(Ytc,10080,[], size(Ytc, 2)), 1));

%% Y (comentar alguna de las alternativas)
% Sin comprimir
%{
yt = Ytsc>0;
y = sum(Ytsc(:,5:end),2)>0;
%}
% Comprimido
yt = Ytc>0;
y = sum(Ytc(:,5:end),2)>0;

%% Compresión directa

%% Modelo MSNM completo
[Xsc_p,m,dt] = preprocess2D(Xsc,2);
testsc_p = preprocess2Dapp(testsc,m,dt);
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xsc_p,1:1,testsc_p,0,000);
pred = 1/268*Dstt/UCLd(1) + 267/268*Qstt/UCLq(1);

[XRocSC,YRocSC,TRocSC,AUCSC] = perfcurve(y,pred,true);

%% Compresión directa en fase I
[~,m,dt] = preprocess2D(Xsc,2);
Xc_p = preprocess2Dapp(Xc,m,dt);
testsc_p = preprocess2Dapp(testsc,m,dt);
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xc_p,1:1,testsc_p,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + 267/268*Qstt/UCLq(1);

[XRocF1,YRocF1,TRocF1,AUCF1] = perfcurve(y,pred,true);

%% Compresión directa en fase II
[~,m,dt] = preprocess2D(Xc,2);
Xsc_p = preprocess2Dapp(Xsc,m,dt);
testc_p = preprocess2Dapp(testc,m,dt);
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xsc_p,1:1,testc_p,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + 267/268*Qstt/UCLq(1);

[XRocF2,YRocF2,TRocF2,AUCF2] = perfcurve(y,pred,true);

%% ADICOV

%% ADICOV PHASE I

%% Preprocesamiento de los datos en phase I
[Xsc_p,m,dt] = preprocess2D(Xsc,2);
Xc_p = preprocess2Dapp(Xc,m,dt);
testsc_p = preprocess2Dapp(testsc,m,dt);

%% Aproximación directa ADICOV (phase I) Alternativa I
P = loadings_pca(Xsc_p,1:1,0,0);
Pr = loadings_pca(Xsc_p,2:268,0,0);
XX = Xsc_p'*Xsc_p;
Xa = zeros(size(Xc_p,1),size(Xc_p,2));
Xr = zeros(size(Xc_p,1),size(Xc_p,2));
for i=1:size(Xc_p,1)
    Xa(i,:) = ADICOV(XX,Xc_p(i,:),1,P);
    Xr(i,:) = ADICOV(XX,Xc_p(i,:),268-1,Pr);
end
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xa+Xr,1:1,testsc_p,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + (268-1)/268*Qstt/UCLq(1);

[XRoc1,YRoc1,TRoc1,AUC1] = perfcurve(y,pred,true);

%% Aproximación directa ADICOV (phase I) Alternativa II
Xa = zeros(size(Xc_p,1),size(Xc_p,2));
Xr = zeros(size(Xc_p,1),size(Xc_p,2));
for i=0:size(Xc_p,1)-2
    XX = Xsc_p(i*10080+1:i*10080+10080,:)'*Xsc_p(i*10080+1:i*10080+10080,:);
    Xa(i+1,:) = ADICOV(XX,Xc_p(i+1,:),1,P);
    Xr(i+1,:) = ADICOV(XX,Xc_p(i+1,:),268-1,Pr);
end

% Última iteración
XX = Xsc_p(70561:72644,:)'*Xsc_p(70561:72644,:);
Xa(8,:) = ADICOV(XX,Xc_p(8,:),1,P);
Xr(8,:) = ADICOV(XX,Xc_p(8,:),268-1,Pr);
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xa+Xr,1:1,testsc_p,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + (268-1)/268*Qstt/UCLq(1);

[XRoc2,YRoc2,TRoc2,AUC2] = perfcurve(y,pred,true);

%% Aproximación directa ADICOV (phase I) Alternativa III
XX = Xsc_p'*Xsc_p;
Xa = ADICOV(XX,Xc_p,1,P);
Xr = ADICOV(XX,Xc_p,268-1,Pr);
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xa+Xr,1:1,testsc_p,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + (268-1)/268*Qstt/UCLq(1);

[XRoc3,YRoc3,TRoc3,AUC3] = perfcurve(y,pred,true);

%% ADICOV PHASE II

%% Preprocesamiento de los datos en phase II
[Xc_p,m,dt] = preprocess2D(Xc,2);
Xsc_p = preprocess2Dapp(Xsc,m,dt);
testc_p = preprocess2Dapp(testc,m,dt);

%% Aproximación directa ADICOV (phase II) Alternativa I
P = loadings_pca(Xsc_p,1:1,0,0);
Pr = loadings_pca(Xsc_p,2:268,0,0);
XX = Xsc_p'*Xsc_p;
Xta = zeros(size(testc_p,1),size(testc_p,2));
Xtr = zeros(size(testc_p,1),size(testc_p,2));
for i=1:size(testc_p,1)
    Xta(i,:) = ADICOV(XX,testc_p(i,:),1,P);
    Xtr(i,:) = ADICOV(XX,testc_p(i,:),268-1,Pr);
end
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xsc_p,1:1,Xta+Xtr,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + (268-1)/268*Qstt/UCLq(1);

[XRoc4,YRoc4,TRoc4,AUC4] = perfcurve(y,pred,true);

%% Aproximación directa ADICOV (phase II) Alternativa II
XX = Xsc_p'*Xsc_p;
Xta = ADICOV(XX,testc_p,1,P);
Xtr = ADICOV(XX,testc_p,268-1,Pr);
tic
[~,~,Dstt,Qstt,UCLd,UCLq] = mspc_pca(Xsc_p,1:1,Xta+Xtr,0,000);
toc
pred = 1/268*Dstt/UCLd(1) + (268-1)/268*Qstt/UCLq(1);

[XRoc5,YRoc5,TRoc5,AUC5] = perfcurve(y,pred,true);

%% Aproximación directa ADICOV (phase II) Alternativa III
elapsed_time = 0;
XX = Xsc_p'*Xsc_p;
indD = zeros(size(testc_p,1),1);
indQ = zeros(size(testc_p,1),1);
for i=1:size(testc_p,1)
    Xta = ADICOV(XX,testc_p(i,:),1,P);
    Xtr = ADICOV(XX,testc_p(i,:),268-1,Pr);
    tic;
    indD(i) = ADindex(testc_p(i,:),Xta,P);
    indQ(i) = ADindex(testc_p(i,:),Xtr,Pr);
    elapsed_time = elapsed_time+toc;
end

pred = 1/268*indD/prctile(indD,99) + (268-1)/268*indQ/prctile(indQ,99);
[XRoc6,YRoc6,TRoc6,AUC6] = perfcurve(y,pred,true);

%% PLOT CURVAS ROC
fig=figure; hold on,
plot(XRocSC,YRocSC,'-.', 'LineWidth', 2);
plot(XRocF1,YRocF1,'-.', 'LineWidth', 2);
plot(XRocF2,YRocF2,'-s', 'LineWidth', 2);
plot(XRoc1,YRoc1,'-.', 'LineWidth', 2);
plot(XRoc2,YRoc2,'-.', 'LineWidth', 2);
plot(XRoc3,YRoc3,'-.', 'LineWidth', 2);
plot(XRoc4,YRoc4,'-o', 'LineWidth', 2);
plot(XRoc5,YRoc5,'-+', 'LineWidth', 2);
plot(XRoc6,YRoc6,'->', 'LineWidth', 2);
legend(sprintf('MSNM completo                                     [AUC=%.2f]',AUCSC), ...
    sprintf('MSNM comprimido en fase I                   [AUC=%.2f]',AUCF1), ...
    sprintf('MSNM comprimido en fase II                  [AUC=%.2f]',AUCF2), ...
    sprintf('MSNM ADICOV en fase I  (Alternativa I)  [AUC=%.2f]',AUC1), ...
    sprintf('MSNM ADICOV en fase I  (Alternativa II) [AUC=%.2f]',AUC2), ...
    sprintf('MSNM ADICOV en fase I  (Alternativa III)[AUC=%.2f]',AUC3), ...
    sprintf('MSNM ADICOV en fase II (Alternativa I)  [AUC=%.2f]',AUC4), ...
    sprintf('MSNM ADICOV en fase II (Alternativa II) [AUC=%.2f]',AUC5), ...
    sprintf('MSNM ADICOV en fase II (Alternativa III)[AUC=%.2f]',AUC6), ...
    'Interpreter','tex', ...
    'Location','southeast', ...
    'NumColumns', 1, ...
    'FontSize', 18);