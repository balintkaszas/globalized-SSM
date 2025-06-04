%% 2D primary mixed-mode SSM for the dynamic buckling of a beam


%% Code is adopted from 
% A. Liu, J. Axås & G. Haller, Data-driven modeling and forecasting 
% of chaotic dynamics on inertial manifolds constructed as 
% spectral submanifolds Chaos 34 (2024) 033140.


% In this example, we apply a sufficiently large axial compression force to 
% a pinned-pinned von Kármán beam [1] and seek to derive an SSM-reduced nonlinear 
% model for the ensuing buckling dynamics.
% 
% 
% 
% [1] S. Jain, P. Tiso, and G. Haller, Exact nonlinear model reduction for a 
% von Kármán beam: slow-fast decomposition and spectral submanifolds, _Journal 
% of Sound and Vibration_ 423 (2018) 195-211. <https://doi.org/10.1016/j.jsv.2018.01.049 
% https://doi.org/10.1016/j.jsv.2018.01.049>


clearvars
close all
load('FE_model_coefficients.mat');
%% Example setup
% The $N$-degree of freedom dynamical system is of the form
% 
% $$\mathbf{M\ddot{q}} + \mathbf{C\dot{q}} + \mathbf{Kq} + \mathbf{f}(\mathbf{q},\mathbf{\dot{q}}) 
% = \mathbf{0}$$
% 
% where $\mathbf{f}=\mathcal{O}(|\mathbf{q}|^2,|\mathbf{\dot{q}}|^2,|\mathbf{q}||\mathbf{\dot{q}}|)$ 
% represents the nonlinearities and $\mathbf{M}$, $\mathbf{C}$, and $\mathbf{K}$ 
% are the $n\times n$ mass, stiffness, and damping matrices, respectively.
% 
% We rewrite the system in first-order form as
% 
% $$\mathbf{\dot{x}} = \mathbf{A}\mathbf{x} + \mathbf{G}(\mathbf{x}) = \mathbf{F}(\mathbf{x})$$
% 
% with
% 
% $\mathbf{x}=\left[\begin{array}{c}\mathbf{q}\\\dot{\mathbf{q}}\end{array}\right],\quad\mathbf{A}=\left[\begin{array}{cc}\mathbf{0}  
% & \mathbf{I}\\-\mathbf{M}^{-1}\mathbf{K} & -\mathbf{M}^{-1}\mathbf{C} \end{array}\right],\quad\mathbf{G}(\mathbf{x})=\left[\begin{array}{c}  
% \mathbf{0} \\ -\mathbf{M}^{-1}\mathbf{f}(\mathbf{x})\end{array}\right]$.

nElements = 12;
l = 2; h = 1e-2; b = 5e-2; % Mesh parameters
E = 190e9;  % Young's modulus
n = size(M,1);    % mechanical dofs (axial def, transverse def, angle)


%% 
% We apply buckling force larger than Euler's critical load
% 
% $$F = \frac{N^2 \pi ^2 EI}{L^2} = \frac{\pi ^2 Ebh^3}{12 L^2}$$

loads = [1.45]*pi^2* E * b*(h)^3 /12 / l^2; 
loadvector = loads.*fExt;

 
%%
obsdof = 35; % select the mid point coordinate in first order form
DS = DynamicalSystem(2);
set(DS,'M',M,'C',C,'K',K_shift,'fnl',fnl_shift);
set(DS.Options,'Emax',5,'Nmax',10,'notation','multiindex')
% set(DS.Options,'Emax',5,'Nmax',10,'notation','tensor')
[V,D,W] = DS.linear_spectral_analysis();
%%
f_periodic  = zeros(size(fExt)); f_periodic(18,1) = 1; % forcing on midpoint
Amp = 21.1154; Frq = 25.30769231;
%%

epsilon = Amp;
kappas = [-1; 1];
coeffs = [f_periodic f_periodic]/2;
DS.add_forcing(coeffs, kappas, epsilon);
%% Autonomous and nonautonomous SSM computation 
S = SSM(DS);
set(S.Options, 'reltol', 100,'notation','multiindex', 'paramStyle', 'graph')
masterModes = [1,10]; 
S.choose_E(masterModes);
[W0, R0] = S.compute_whisker(18);
save('vonkarman_buckled_order18.mat', 'W0', 'R0')
[W1, R1] = S.compute_perturbed_whisker(17,W0,R0,Frq);
save('vonkarman_buckled_order17_nonaut.mat', 'W1', 'R1')


%% Simulate trajectories as in Liu et al:

[F, ~] = functionFromTensors(M, C, K, fnl, loadvector, 0);
z0 = 1e-5;
q0_plus = reduced_to_full_traj(0,[z0; 0],W0);
q0_shifted_plus = q0_plus + unstable_fp;

q0_minus = reduced_to_full_traj(0,[-z0; 0],W0);
q0_shifted_minus = q0_minus + unstable_fp;

timespan = linspace(0,2,2000);
options = odeset('RelTol',1e-8,'AbsTol',1e-10);

[~, sol_plus] = ode45(F, timespan, q0_shifted_plus, options);
[~, sol_minus] = ode45(F, timespan, q0_shifted_minus, options);
DataInfo = struct('nElements', nElements, 'loadvector', loadvector, 'unstable_fp', unstable_fp);
save('data_buckled_unforced.mat', 'DataInfo',  'sol_plus', 'sol_minus', 'unstable_fp');
%%
figure;
hold on; 
plot(timespan, sol_plus(:, obsdof+1))
plot(timespan, sol_minus(:, obsdof+1))
stable_fp = sol_plus(end, :);
xlabel('Time');
ylabel('Midpoint displacement')
%% 
% Add the periodic forcing and simulate chaotic trajectories. 
% This part has a long runtime.
 


options = odeset('RelTol',1e-6,'AbsTol',1e-6);

F_f = Amp.*(M\f_periodic);
F_forced = @(t,x,w) F(t,x) + [zeros(n,1); F_f*cos(w*t)];

x0_ep1 = ic_ + unstable_fp; 
x0_ep2 = ic_ + unstable_fp; 
x0_ep2(18) = x0_ep2(18) + 1e-4; % add a small perturbation to the IC

slowTimeScale = 2*pi/Frq;
numberPeriods = 200;  
numberPointsPerPeriod = 100; 
endTime = numberPeriods*slowTimeScale;
nSamp = numberPeriods*numberPointsPerPeriod+1;
dt = endTime/(nSamp-1);
timespan = [0:dt:endTime];

[~, sol_ep1] = ode15s( @(t,x) F_forced(t,x,Frq), timespan, x0_ep1, options);
[~, sol_ep2] = ode15s( @(t,x) F_forced(t,x,Frq), timespan, x0_ep2, soptions);

save('data_buckled_forced_chaotic.mat', 'Amp', 'reduced', 'Frq', 'F_f', 'M', 'sol_ep1', 'sol_ep2', 'unstable_fp');
figure;
hold on; 
plot(timespan, sol_ep1(:, obsdof+1))
plot(timespan, sol_ep2(:, obsdof+1))
stable_fp = sol_plus(end, :);
xlabel('Time');
ylabel('Midpoint displacement')


%% Compute the projection of the forcing vector: leading order contribution
% to the reduced dynamics

f_periodic  = zeros(size(fExt)); f_periodic(18,1) = 1; % forcing on midpoint
Amp = 21.1;
Frq = 23.957894736842107; % forcing amplitude and frequency in full state space
left_eig = W(:, [1,10]);

F_f = Amp.*(M\f_periodic);
F_forced = @(t,x,w) F(t,x) + [zeros(n,1); F_f*cos(w*t)];
f_ext_vector = [zeros(n,1); F_f];
vstar_fext = left_eig'*DS.B*f_ext_vector
