clear
clc
syms x y vx vy psi r Tf betaf


%%
% rho_aero = 1.225;
% mveh = 1500;
% Cd = 0.389;
% radwhl = 0.2159;
% Af = 4;
% Lxf = 1.2;
% Lxr = 1.4;
% mu = 1;
% Iw = 3.8782;
% I = 4192;
% Calpha = -0.08 * 180 / pi;
% sa_max = 10 / 180 * pi;
% g = 9.8;
% sigma = 0.0;
% small_pos_th = 1e-6;

syms rho_aero mveh Cd radwhl Af Lxf Lxr mu Iw I Calpha sa_max g sigma


%     lx = x0[0, :]
%     vx = x0[1, :]
%     ly = x0[2, :]
%     vy = x0[3, :]
%     psi = x0[4, :]
%     r = x0[5, :]

%     Tf = u0[0, :]
Tr = 0.0;
dmf = 0.0;
dmr = 0.0;
%     betaf = u0[1, :]
betar = 0.0;

%      Define rotation matrix
rotM_00 = cos(betaf);
rotM_01 = -sin(betaf);
rotM_10 = sin(betaf);
rotM_11 = cos(betaf);
rotM_20 = cos(betar);
rotM_21 = -sin(betar);
rotM_30 = sin(betar);
rotM_31 = cos(betar);

%     # Calculating corner and wheel velocity
v_c_00 = vx;                     % v_c_00 = v_xr
v_c_01 = vy + Lxf * r;           % = v_yr
v_c_10 = vx;                     % = v_xf
v_c_11 = vy - Lxr * r;            % = v_yf

v_w_00 = v_c_00 * rotM_00 + v_c_01 * rotM_10;  % \bar{v}_xr
v_w_01 = v_c_00 * rotM_01 + v_c_01 * rotM_11;  % \bar{v}_yr
v_w_10 = v_c_10 * rotM_20 + v_c_11 * rotM_30;  % \bar{v}_xf
v_w_11 = v_c_10 * rotM_21 + v_c_11 * rotM_31;  % \bar{v}_yf

%      Calculating tire force at each wheel
Fz_0 = Lxr * mveh * g / 2 / (Lxf + Lxr);
Fz_1 = Lxf * mveh * g / 2 / (Lxf + Lxr);

F_w00 = Tf / 2 / radwhl;
F_w10 = Tr / 2 / radwhl;

sa0 = atan(v_w_01 / v_w_00);
sa1 = atan(v_w_11 / v_w_10);

F_w01 = Calpha * mu * sa0 * Fz_0;
F_w11 = Calpha * mu * sa1 * Fz_1;

F_c_00 = F_w00 * rotM_00 + F_w01 * rotM_01;
F_c_01 = F_w00 * rotM_10 + F_w01 * rotM_11;
F_c_10 = F_w10 * rotM_20 + F_w11 * rotM_21;
F_c_11 = F_w10 * rotM_30 + F_w11 * rotM_31;

%     calculating the xdot
%     xdot = np.zeros(6)
f1 = vx * cos(psi) - vy * sin(psi);
f2 = vy * r + 2 * (F_c_00 + F_c_10) / mveh - g * sin(sigma) - 0.5 * rho_aero * Cd * Af * vx * vx / mveh;
f3 = vx * sin(psi) + vy * cos(psi);
f4 = -vx * r + 2 * (F_c_01 + F_c_11) / mveh;
f5 = r;
f6 = (2 * F_c_01 * Lxf - 2 * F_c_11 * Lxr + dmf + dmr) / I;


%%

A = [ diff(f1,x) diff(f1,y) diff(f1,vx) diff(f1,vy) diff(f1,psi) diff(f1,r);
    diff(f2,x) diff(f2,y) diff(f2,vx) diff(f2,vy) diff(f2,psi) diff(f2,r);
    diff(f3,x) diff(f3,y) diff(f3,vx) diff(f3,vy) diff(f3,psi) diff(f3,r);
    diff(f4,x) diff(f4,y) diff(f4,vx) diff(f4,vy) diff(f4,psi) diff(f4,r);
    diff(f5,x) diff(f5,y) diff(f5,vx) diff(f5,vy) diff(f5,psi) diff(f5,r);
    diff(f6,x) diff(f6,y) diff(f6,vx) diff(f6,vy) diff(f6,psi) diff(f6,r)];

B = [diff(f1,Tf) diff(f1,betaf);
    diff(f2,Tf) diff(f2,betaf);
    diff(f3,Tf) diff(f3,betaf);
    diff(f4,Tf) diff(f4,betaf);
    diff(f5,Tf) diff(f5,betaf);
    diff(f6,Tf) diff(f6,betaf)];







