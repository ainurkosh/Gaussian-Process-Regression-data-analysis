% Gaussian Process Regression
clear
% gprMdl = fitrgp(Tbl,ResponseVarName);
tbl = readtable('test_data.txt','Filetype','text',...
     'ReadVariableNames',false);
tbl.Properties.VariableNames = {'G', 'Temp',	'f_d',	'f_melt',	...
    'S_melt',	'f_k',	'S_k',	'f_Debye',	'S_Debye',	'f_spec_heat',	...
    'S_spec_heat',	'f_rho',	'S_rho',	'f_vl',	'S_vl',	'f_Young', ...	
    'S_Young',	'f_Lattice_const',	'S_Lattice_const',	'f_Molar_mass',	...
    'S_Molar_mass',	'f_molar_volume', 'S_molar_volume'};
tbl(1:70,:)

% 
% gprMdl = fitrgp(tbl,'G', 'KernelFunction','squaredexponential',...               % works
%       'FitMethod','exact','PredictMethod','exact');


gprMdl = fitrgp(tbl,'G', 'KernelFunction','squaredexponential',...               % works
      'FitMethod','sr','PredictMethod','exact');


% gprMdl = fitrgp(tbl,'itr','KernelFunction','ardsquaredexponential',...
%       'FitMethod','sr','PredictMethod','fic','Standardize',1);

ypred = resubPredict(gprMdl);

% Compute the regression loss on the training data (resubstitution loss) for the trained model.
% returns the mean squared error for the Gaussian process regression (GPR) model gpr
RL = resubLoss(gprMdl);

% initial value of the coefficients
beta  = gprMdl.Beta;       

% initial value for the noise standard deviation of the Gaussian process model
sigma  = gprMdl.Sigma;      

% % initial value for the noise standard deviation of the Gaussian process model
% weights  = gprMdl.W;   
% [loores,neff] = postFitStatistics(gprMdl)


test_corr = load('test_data.txt');
R = corrcoef(test_corr);
%R(isnan(R))=min(min(R));
R(isnan(R))=0;


% % =================== Analysis ========================================
figure();
plot(tbl.f_Debye, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_Debye, ypred,'ro', 'LineWidth', 1.5);
xlabel('Film Debye temperature');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
hold off;

% relative TD vs. G
figure();
plot(tbl.f_Debye./ tbl.S_Debye, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_Debye./ tbl.S_Debye, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_D_e_b_y_e/ S_D_e_b_y_e');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative TD vs. relative G
figure();
plot(tbl.f_Debye./ tbl.S_Debye, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
indTD = find(abs((tbl.G - ypred) ./ tbl.G) == max(abs((tbl.G - ypred) ./ tbl.G)));
xlabel('f_D_e_b_y_e/ S_D_e_b_y_e');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')

% relative vl vs. G
figure();
%plot(tbl.f_vl./ tbl.S_vl, tbl.G,'bo', 'LineWidth', 1.5);
plot(tbl.S_vl./ tbl.f_vl, tbl.G,'bo', 'LineWidth', 1.5);
hold on
%plot(tbl.f_vl./ tbl.S_vl, ypred,'ro', 'LineWidth', 1.5);
plot(tbl.S_vl./ tbl.f_vl, ypred,'ro', 'LineWidth', 1.5);
xlabel('S_v_l/ f_v_l');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;


% relative TD vs. relative v
figure();
plot(tbl.S_vl./ tbl.f_vl, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('S_v_l./ f_v_l');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')

% relative melting T vs. G
figure();
plot(tbl.f_melt./ tbl.S_melt, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_melt./ tbl.S_melt, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_m_e_l_t/ S_m_e_l_t');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative melting T vs. relative G
figure();
plot(tbl.f_melt./ tbl.S_melt, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('f_m_e_l_t/ S_m_e_l_t');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')

% relative rho vs. G
figure();
plot(tbl.f_rho./ tbl.S_rho, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_rho./ tbl.S_rho, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_r_h_o/ S_r_h_o');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative rho vs. relative G
figure();
plot(tbl.f_rho./ tbl.S_rho, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('f_r_h_o/ S_r_h_o');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')

% relative Young modulus vs. G
figure();
plot(tbl.f_Young./ tbl.S_Young, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_Young./ tbl.S_Young, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_Y_o_u_n_g/ S_Y_o_u_n_g');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative Young modulus vs. relative G
figure();
plot(tbl.f_Young./ tbl.S_Young, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('f_Y_o_u_n_g/ S_Y_o_u_n_g');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')

% relative molar volume vs. G
figure();
plot(tbl.f_molar_volume./ tbl.S_molar_volume, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_molar_volume./ tbl.S_molar_volume, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_m_o_l_a_r _v_o_l_u_m_e/ S_m_o_l_a_r _v_o_l_u_m_e');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative molar volume vs. relative G
figure();
plot(tbl.f_molar_volume./ tbl.S_molar_volume, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('f_m_o_l_a_r _v_o_l_u_m_e/ S_m_o_l_a_r _v_o_l_u_m_e');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')


% relative thermal conductivity vs. G
figure();
plot(tbl.f_k./ tbl.S_k, tbl.G,'bo', 'LineWidth', 1.5);
hold on
plot(tbl.f_k./ tbl.S_k, ypred,'ro', 'LineWidth', 1.5);
xlabel('f_k/ S_k');
ylabel('G');
legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')
hold off;

% relative thermal conductivity vs. relative G
figure();
plot(tbl.f_k./ tbl.S_k, abs((tbl.G - ypred) ./ tbl.G),'bo', 'LineWidth', 1.5);
xlabel('f_k/ S_k');
ylabel('abs((G - pred) ./ G');
%legend({'data','predictions'},'Location','Best');
%axis([0 4300 0 30]);
set(gca,'FontSize',20)
set(gca,'XMinorTick','on','YMinorTick','on')




% check the similarity of the predicted data to the experimetnal
figure();
plot(tbl.G, ypred, 'bo', 'LineWidth', 1.5);
hold on
x_temp = min(tbl.G) : max(tbl.G);
y_temp = x_temp; 
plot(x_temp, y_temp, 'r--', 'LineWidth', 1.5)
xlabel('G experimental data set');
ylabel('G predicted');
legend({'GPR'},'Location','Best');
set(gca,'FontSize',20)
%axis([19 27 19 27]);
set(gca,'XMinorTick','on','YMinorTick','on')
hold off


% =======================================================================

% correlation of the coefficients
%modify the heatmap
R(2,:) = [];
R(:,2) = [];

figure
ytemp = 1 : 22;

Rf = R([1,2,3,5,7,9, 11, 13, 15, 17, 19, 21],:);
R(4,:) = [];
R(6-1,:) = [];
R(8-2,:) = [];
R(10-3,:) = [];
R(12-4,:) = [];
R(14-5,:) = [];
R(16-6,:) = [];
R(18-7,:) = [];
R(20-8,:) = [];
R(22-9,:) = [];

RS = R(:,[1,2,4,6,8,10, 12, 14, 16, 18, 20, 22]);
R(:,3) = [];
R(:, 5-1) = [];
R(:, 7-2) = [];
R(:, 9-3) = [];
R(:, 11-4) = [];
R(:, 13-5) = [];
R(:, 15-6) = [];
R(:, 17-7) = [];
R(:, 19-8) = [];
R(:,21-9) = [];

%h = heatmap(R([1,2,3,5,7,9, 11, 13, 15, 17, 19, 21],:),R([1,2,4,6,8,10, 12, 14, 16, 18, 20, 22],:), 'Colormap', summer);
%h = heatmap(Rf, RS, 'Colormap', summer);
h = heatmap(R, 'Colormap', summer);
h.Colormap = parula;


%==========================================


colorbar
%legend({'1 G     2 Temp     3 f_ d	    4 f_ melt     5 S_ melt     6 f_ k     7 S_ k	    8 f_ Debye    	9 S_ Debye    	10 f_ spec_ heat    	11 S_ spec_ heat    	12 f_rho    	13 S_ pho    	14 f_ vl    	15 S_ vl     16 f_ Young     17 S_ Young     18 f_ Lattice_ const    	19 S_ Lattice_ const    	20 f_ Molar_ mass    	21 S_ Molar_ mass 	   22 f_ molar_ volume     23 S_ molar_ volume'}, 'numcolumns',2)
%display('1 G     2 Temp     3 f_ d	    4 f_ melt     5 S_ melt     6 f_ k     7 S_ k	    8 f_ Debye    	9 S_ Debye    	10 f_ spec_ heat    	11 S_ spec_ heat    	12 f_rho    	13 S_ pho    	14 f_ vl    	15 S_ vl     16 f_ Young     17 S_ Young     18 f_ Lattice_ const    	19 S_ Lattice_ const    	20 f_ Molar_ mass    	21 S_ Molar_ mass 	   22 f_ molar_ volume     23 S_ molar_ volume')
%axis([0 24 0 24]);

% legend('G', 'Temp',	'f_d',	'f_melt',	...
%     'S_melt',	'f_k',	'S_k',	'f_Debye',	'S_Debye',	'f_spec_heat',	...
%     'S_spec_heat',	'f_rho',	'S_pho',	'f_vl',	'S_vl',	'f_Young', ...	
%     'S_Young',	'f_Lattice_const',	'S_Lattice_const',	'f_Molar_mass',	...
%     'S_Molar_mass',	'f_molar_volume', 'S_molar_volume')
% 

% 5 points of values

% minor value
ind1 = find(tbl.G == min(tbl.G));
param1 = [tbl.f_Debye(ind1)./ tbl.S_Debye(ind1) tbl.f_vl(ind1)./ tbl.S_vl(ind1) tbl.f_melt(ind1)./ tbl.S_melt(ind1) ...
    tbl.f_rho(ind1)./ tbl.S_rho(ind1) tbl.f_Young(ind1)./ tbl.S_Young(ind1) tbl.f_molar_volume(ind1)./ tbl.S_molar_volume(ind1) ...
    tbl.f_k(ind1)./ tbl.S_k(ind1)];

temp = abs((tbl.G - ypred) ./ tbl.G);
indr1 = find(temp == min(temp));
paramr1 = [tbl.f_Debye(indr1)./ tbl.S_Debye(indr1) tbl.f_vl(indr1)./ tbl.S_vl(indr1) tbl.f_melt(indr1)./ tbl.S_melt(indr1) ...
    tbl.f_rho(indr1)./ tbl.S_rho(indr1) tbl.f_Young(indr1)./ tbl.S_Young(indr1) tbl.f_molar_volume(indr1)./ tbl.S_molar_volume(indr1) ...
    tbl.f_k(indr1)./ tbl.S_k(indr1)];

% minor value + 20%
sizeG = 70;
letter = 14;
test = sortrows(tbl.G);
%ind2 = find( tbl.G == test(sizeG/5 + 1) );
ind2 = find(tbl.G == 320);
param2 = [tbl.f_Debye(ind2)./ tbl.S_Debye(ind2) tbl.f_vl(ind2)./ tbl.S_vl(ind2) tbl.f_melt(ind2)./ tbl.S_melt(ind2) ...
    tbl.f_rho(ind2)./ tbl.S_rho(ind2) tbl.f_Young(ind2)./ tbl.S_Young(ind2) tbl.f_molar_volume(ind2)./ tbl.S_molar_volume(ind2) ...
    tbl.f_k(ind2)./ tbl.S_k(ind2)];

temptest = sortrows(temp);
%indr2 = find(temp == temptest(sizeG/5 + 1));
indr2 = find(temp == temptest(57));
paramr2 = [tbl.f_Debye(indr2)./ tbl.S_Debye(indr2) tbl.f_vl(indr2)./ tbl.S_vl(indr2) tbl.f_melt(indr2)./ tbl.S_melt(indr2) ...
    tbl.f_rho(indr2)./ tbl.S_rho(indr2) tbl.f_Young(indr2)./ tbl.S_Young(indr2) tbl.f_molar_volume(indr2)./ tbl.S_molar_volume(indr2) ...
    tbl.f_k(indr2)./ tbl.S_k(indr2)];

% minor value + 40%
%ind3 = find( tbl.G == test(2 * sizeG/5 + 1) );
ind3 = find(tbl.G == 500);
param3 = [tbl.f_Debye(ind3)./ tbl.S_Debye(ind3) tbl.f_vl(ind3)./ tbl.S_vl(ind3) tbl.f_melt(ind3)./ tbl.S_melt(ind3) ...
    tbl.f_rho(ind3)./ tbl.S_rho(ind3) tbl.f_Young(ind3)./ tbl.S_Young(ind3) tbl.f_molar_volume(ind3)./ tbl.S_molar_volume(ind3) ...
    tbl.f_k(ind3)./ tbl.S_k(ind3)];

%indr3 = find(temp == temptest(2 * sizeG/5 + 1));
indr3 = find(temp == temptest(65));
paramr3 = [tbl.f_Debye(indr3)./ tbl.S_Debye(indr3) tbl.f_vl(indr3)./ tbl.S_vl(indr3) tbl.f_melt(indr3)./ tbl.S_melt(indr3) ...
    tbl.f_rho(indr3)./ tbl.S_rho(indr3) tbl.f_Young(indr3)./ tbl.S_Young(indr3) tbl.f_molar_volume(indr3)./ tbl.S_molar_volume(indr3) ...
    tbl.f_k(indr3)./ tbl.S_k(indr3)];

% minor value + 60%
%ind4 = find( tbl.G == test(3 * sizeG/5 + 1) );
ind4 = find(tbl.G == 900);
param4 = [tbl.f_Debye(ind4)./ tbl.S_Debye(ind4) tbl.f_vl(ind4)./ tbl.S_vl(ind4) tbl.f_melt(ind4)./ tbl.S_melt(ind4) ...
    tbl.f_rho(ind4)./ tbl.S_rho(ind4) tbl.f_Young(ind4)./ tbl.S_Young(ind4) tbl.f_molar_volume(ind4)./ tbl.S_molar_volume(ind4) ...
    tbl.f_k(ind4)./ tbl.S_k(ind4)];

%indr4 = find(temp == temptest(3 * sizeG/5 + 1));
indr4 = find(temp == temptest(69));
paramr4 = [tbl.f_Debye(indr4)./ tbl.S_Debye(indr4) tbl.f_vl(indr4)./ tbl.S_vl(indr4) tbl.f_melt(indr4)./ tbl.S_melt(indr4) ...
    tbl.f_rho(indr4)./ tbl.S_rho(indr4) tbl.f_Young(indr4)./ tbl.S_Young(indr4) tbl.f_molar_volume(indr4)./ tbl.S_molar_volume(indr4) ...
    tbl.f_k(indr4)./ tbl.S_k(indr4)];

% % minor value + 80%
% ind5 = find( tbl.G == test(4 * sizeG/5 + 1) );
% param5 = [tbl.f_Debye(ind5)./ tbl.S_Debye(ind5) tbl.f_vl(ind5)./ tbl.S_vl(ind5) tbl.f_melt(ind5)./ tbl.S_melt(ind5) ...
%     tbl.f_rho(ind5)./ tbl.S_rho(ind5) tbl.f_Young(ind5)./ tbl.S_Young(ind5) tbl.f_molar_volume(ind5)./ tbl.S_molar_volume(ind5) ...
%     tbl.f_k(ind5)./ tbl.S_k(ind5)];
% 
% indr5 = find(temp == temptest(4 * sizeG/5 + 1));
% paramr5 = [tbl.f_Debye(indr5)./ tbl.S_Debye(indr5) tbl.f_vl(indr5)./ tbl.S_vl(indr5) tbl.f_melt(indr5)./ tbl.S_melt(indr5) ...
%     tbl.f_rho(indr5)./ tbl.S_rho(indr5) tbl.f_Young(indr5)./ tbl.S_Young(indr5) tbl.f_molar_volume(indr5)./ tbl.S_molar_volume(indr5) ...
%     tbl.f_k(indr5)./ tbl.S_k(indr5)];

% peak values
ind = find(tbl.G == max(tbl.G));
param = [tbl.f_Debye(ind)./ tbl.S_Debye(ind) tbl.f_vl(ind)./ tbl.S_vl(ind) tbl.f_melt(ind)./ tbl.S_melt(ind) ...
    tbl.f_rho(ind)./ tbl.S_rho(ind) tbl.f_Young(ind)./ tbl.S_Young(ind) tbl.f_molar_volume(ind)./ tbl.S_molar_volume(ind) ...
    tbl.f_k(ind)./ tbl.S_k(ind)];

indr = find(temp == max(temp));
paramr = [tbl.f_Debye(indr)./ tbl.S_Debye(indr) tbl.f_vl(indr)./ tbl.S_vl(indr) tbl.f_melt(indr)./ tbl.S_melt(indr) ...
    tbl.f_rho(indr)./ tbl.S_rho(indr) tbl.f_Young(indr)./ tbl.S_Young(indr) tbl.f_molar_volume(indr)./ tbl.S_molar_volume(indr) ...
    tbl.f_k(indr)./ tbl.S_k(indr)];

figure
suptitle('param ratio (f/S) @ various G')
subplot(3,2,1)
plot(param1, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
%ylabel('param ratio (f/S) @ max(G)');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Bi/H/diamond')

subplot(3,2,2)
plot(param2, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
%ylabel('param ratio (f/S) @ max(G)');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Cr/diamond')

subplot(3,2,3)
plot(param3, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
%ylabel('param ratio (f/S) @ max(G)');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Bi/sapphire')

subplot(3,2,4)
plot(param4, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
%ylabel('param ratio (f/S) @ max(G)');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('ZrN/ScN')

% subplot(3,2,5)
% plot(param5, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
% %ylabel('param ratio (f/S) @ max(G)');
% set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
% grid on
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gca,'FontSize',letter)

subplot(3,2,5)
plot(param, 'sk', 'MarkerFaceColor', [0.5 0.5 0.5], 'LineWidth', 2)
%ylabel('param ratio (f/S) @ max(G)');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('TiN/ScN')


figure
suptitle('param ratio (f/S)@ various G')
subplot(3,2,1)
plot(paramr1, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
%ylabel('param ratio (f/S)@ max err G');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Ni/Si')


subplot(3,2,2)
plot(paramr2, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
%ylabel('param ratio (f/S)@ max err G');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Ir/sapphire')


subplot(3,2,3)
plot(paramr3, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
%ylabel('param ratio (f/S)@ max err G');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Ti/diamond')


subplot(3,2,4)
plot(paramr4, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
%ylabel('param ratio (f/S)@ max err G');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Al/H:diamond')


% subplot(3,2,5)
% plot(paramr5, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
% %ylabel('param ratio (f/S)@ max err G');
% set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
% grid on
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gca,'FontSize',letter)

subplot(3,2,5)
plot(paramr, 'sk', 'MarkerFaceColor', [0.5 0.0 0.0], 'LineWidth', 2)
%ylabel('param ratio (f/S)@ max err G');
set(gca,'XTickLabel',{'TDebye','vl','melt','rho','Young','molvol','k'})
grid on
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'FontSize',letter)
legend('Bi/H/diamond')
