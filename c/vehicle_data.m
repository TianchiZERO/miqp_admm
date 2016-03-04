% Data file for hybrid vehicle example.

% Fuel use is given by F(p) = p+ gamma*p^2 (for p>=0)
% We assume that the path is piecewise linear and 
% the slope and length of each piece is stored in
% a, and l, respectively.e

a=[0.5 -0.5 0.2 -0.7 0.6 -0.2 0.7 -0.5 0.8 -0.4]/ 10;
l=[40 20 40 40 20 40 30 40 30 60];

Preq=(a(1): a(1): a(1) * l(1))';
for i=2: length(l)
    Preq =[Preq; (Preq(end) + a(i): a(i): Preq(end) + a(i) * l(i))'];
end


% Model
P_des = Preq(1: 5: end);
T = length(P_des);
tau = 4;
P_eng_max = 1;
alpha = 1;
beta = 1;
gamma = 1;
delta = 3;
eta = 1;
E_max = 40;
E_0 = E_max;
f_tilde = @(P_eng, eng_on) alpha * square(P_eng) + beta * P_eng + gamma * eng_on;


% Build matrices; vector is [P_eng(T); P_batt(T); E_batt(T); turn_on(T-1), eng_on(T)];
% Dynamics
A = [zeros(T - 1, T), ...
     toeplitz([0, zeros(1, T - 2)], [0, 1, zeros(1, T - 2)]), ...
     toeplitz([-1, zeros(1, T - 2)], [-1, 1, zeros(1, T - 2)]) / tau, ...
     zeros(T - 1, T - 1), ...
     zeros(T - 1, T)];
b = zeros(T - 1, 1);

% Initial dynamics
A = [A; [zeros(1, T), tau, zeros(1, T - 1), 1, zeros(1, T - 1), zeros(1, T - 1), zeros(1, T)]]; 
b = [b; E_0];

% Power balance
G = [eye(T), eye(T), zeros(T, 3 * T - 1)];
h = P_des;

% Battery limits
G = [G; [zeros(T, 2 * T), eye(T), zeros(T, 2 * T - 1)]];
h = [h; zeros(T, 1)];
G = [G; [zeros(T, 2 * T), -eye(T), zeros(T, 2 * T - 1)]];
h = [h; -E_max * ones(T, 1)];

% P_eng limits
G = [G; [eye(T), zeros(T, 4 * T - 1)]];
h = [h; zeros(T, 1)];
G = [G; [-eye(T), zeros(T, 3 * T - 1), P_eng_max * eye(T)]];
h = [h; zeros(T, 1)];

% Turn_on
G = [G; [zeros(T, 3 * T), eye(T, T -1 ), zeros(T)]];
h = [h; zeros(T, 1)];
G = [G; [zeros(T - 1, 3 * T), ...
     eye(T - 1, T - 1) ...
     -toeplitz([-1, zeros(1, T - 2)], [-1, 1, zeros(1, T - 2)])]];
h = [h; zeros(T - 1, 1)];

% Fuel cost
Phalf = [ ... 
          sqrt(alpha) * eye(T), zeros(T, 4 * T - 1)
          zeros(1, 3 * T - 1), sqrt(eta), zeros(1, 2 * T - 1)
        ];
P = 2 * (Phalf' * Phalf); % Objective will be (1/2)x^TPx, not x^TPx
q = [beta * ones(T, 1); zeros(4 * T - 1, 1)];
q = q + [zeros(3 * T - 1, 1); -2 * eta * E_max; zeros(2 * T - 1, 1)];
q = q + [zeros(4 * T - 1, 1); gamma * ones(T, 1)];
r = eta * E_max^2;

% turn-on cost
q = q + delta * [zeros(3 * T, 1); ones(T - 1, 1); zeros(T, 1)];

% put problem in standard form with slack variables
l = length(h);
k = length(P);
P = [P, zeros(k, l); zeros(l, k), zeros(l, l)];
q = [q; zeros(l, 1)];
A = [A, zeros(size(A, 1), l); G, -eye(l)];
b = [b; h];

% permute vectors to the standard form
l1 = T;
l2 = 0;
l3 = l;
l5 = 4*T-1;
n = l1 + l3 + 4 * T - 1;
Perm = [ ...
          zeros(l5, l1), zeros(l5, l3), eye(l5),
          eye(l1),       zeros(l1, l3), zeros(l1, l5),
          zeros(l3, l1), eye(l3),       zeros(l3, l5),
       ];
   
A = A * Perm;
P = Perm' * P * Perm;
q = Perm' * q;
