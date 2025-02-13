clear all; close all; clc
nElements = 10;

[M,C,K,fnl,f_0,outdof] = build_model(nElements);
n = length(M);
disp(['Number of degrees of freedom = ' num2str(n)])
disp(['Phase space dimensionality = ' num2str(2*n)])

%DS = DynamicalSystem();
%set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
%set(DS.Options,'Emax',5,'Nmax',10,'notation','multiindex')
% set(DS.Options,'Emax',5,'Nmax',10,'notation','tensor')
kappas = [-1; 1];
coeffs = [f_0 f_0]/2;
epsilons = linspace(5e-4, 5e-2, 10);
ii = 1;
for e = epsilons
    disp(e);
    DS = DynamicalSystem();
    set(DS,'M',M,'C',C,'K',K,'fnl',fnl);
    set(DS.Options,'Emax',5,'Nmax',10,'notation','multiindex')
    %set(DS.Options,'Emax',5,'Nmax',10,'notation','tensor');
    DS.add_forcing((e / 5e-4) * coeffs, kappas, 5e-4);

    [V,D,W] = DS.linear_spectral_analysis();
    omega0 = imag(D(1,1));
    omegaRange = omega0*[0.5 2.2];

    figure(ii); hold on
    nCycles = 500;
    coco = cocoWrapper(DS, nCycles, outdof);
    set(coco.Options, 'PtMX', 1000, 'NTST',20, 'dir_name', 'bd_nd');
    set(coco.Options, 'NAdapt', 0, 'h_max', 200, 'MaxRes', 1);
    coco.initialGuess = 'linear';
    start = tic;
    bd_nd = coco.extract_FRC(omegaRange);
    timings.cocoFRCbd_nd = toc(start)
    new_name = ['data_', num2str(ii)];
    movefile('data', new_name);
    ii = ii + 1;
    title(['Eps = ', num2str(e)]);
end
