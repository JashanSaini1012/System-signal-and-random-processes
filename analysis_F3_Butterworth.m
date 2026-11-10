%% analysis_F3_Butterworth.m
% Project 14 (SSRP): analysis of ONE smoothing filter on ALL subjects.
%   Filter: Butterworth low-pass, order 2, -3 dB at 4 Hz
%   Proposed by: Bhavya Chaudhary (25110075)
%
% The processing is identical to bioradio_emg.py (the program that drives the Unity game):
%   raw -> Butterworth band-pass 20-450 Hz (order 4) -> 50 Hz notch (Q = 30)   [same in all 4 files]
%       -> full-wave rectify -> THIS SMOOTHING FILTER -> (env - rest)/(MVC - rest), clipped 0..1.5
% The four files (analysis_F3_Butterworth, analysis_F5_Kalman, analysis_F6_ChebyshevI,
% analysis_F7_ChebyshevII) share every setting and measurement except the smoothing filter,
% so their results can be compared directly.
%
% Put this file next to the 'data' folder (S*_calibration.json and S*_*_game_*.csv).
% Needs the Signal Processing Toolbox. Results are saved in ./results_F3_Butterworth

clear; clc; close all;
set(0, 'DefaultFigureVisible', 'off');   % draw figures off-screen so no window can be closed mid-run
cleanupVis = onCleanup(@() set(0, 'DefaultFigureVisible', 'on'));
set(groot, 'defaultAxesToolbarVisible', 'off');   % keep the axes toolbar out of saved figures

%% ================= Settings (same in all four files) =================
fs      = 1000;
dataDir = 'data';
% Signal analysis uses only each subject's data-collection recording. S2 also played
% the game rounds; those are analysed in game_analysis.m only.
ONLY_FILES = struct('S2', 'S2_F2_game_20261109_174535.csv');
skip    = round(1.0*fs);       % samples skipped at the start of every file (filter start-up)
N       = 100;                 % 100 ms window for the MAV / RMS features
ZOOM       = 10;               % seconds shown in the zoomed figure (Fig2)
ZOOM_START = [];               % [] = pick the most active part automatically,
                               % or a number, e.g. 120, to start the zoom 120 s into the recording

% Game thresholds, same as UdpEmgControl.cs (fractions of MVC)
G.runOn = 0.10;  G.runFull = 0.60;
G.kickOn = 0.30; G.kickOff = 0.20; G.kickWin = 0.30; G.lockout = 0.70;

% Band-pass + notch (same for every filter)
[bBP, aBP] = butter(4, [20 450]/(fs/2), 'bandpass');
[bN,  aN ] = notch50(fs);
[bRef, aRef] = butter(2, 4/(fs/2));         % reference envelope (zero-phase), used to measure delay
[bh, ah]   = butter(2, 3/(fs/2), 'high');   % used only to measure ripple

%% ================= THE SMOOTHING FILTER OF THIS FILE =================
F.id   = 'F3';
F.name = 'Butterworth';
F.desc = 'Butterworth low-pass, order 2, -3 dB at 4 Hz';
F.type = 'iir';
[F.b, F.a] = butter(2, 4/(fs/2));
F.designB = F.b;  F.designA = F.a;

outDir = fullfile(pwd, ['results_' F.id '_' F.name]);
if ~exist(outDir, 'dir'), mkdir(outDir); end
col = [0.00 0.45 0.74];

%% ================= PART 1: the filter on its own =================
f = logspace(-1, log10(450), 3000)';
H = freqz(F.designB, F.designA, f, fs);
gd = grpdelay(F.designB, F.designA, f, fs)/fs*1000;

% Step test: muscle level jumps 0 -> 0.5 MVC for 1 s, then back to 0
t0 = 1*fs;  t1 = 2*fs;
u  = [zeros(t0,1); 0.5*ones(t1 - t0,1); zeros(fs,1)];
y  = applySmooth(u, F);
y  = y / (y(t1)/0.5);                        % remove the DC gain, like the calibration does
tt = ((0:numel(u)-1)' - t0)/fs*1000;
startDelay = (find(y(t0+1:end) >= G.runOn, 1) - 1)/fs*1000;
stopDelay  = (find(y(t1+1:end) <= G.runOn, 1) - 1)/fs*1000;
overshoot  = 100*(max(y(t0+1:t1))/0.5 - 1);

h = figure('Name', 'Filter design', 'Position', [80 80 1150 750]);
subplot(2,2,1);
plot(f, 20*log10(abs(H) + eps), 'LineWidth', 1.5, 'Color', col);
set(gca, 'XScale', 'log'); grid on; xlim([0.1 450]); ylim([-100 5]);
xline(4, 'k:'); xlabel('Frequency (Hz)'); ylabel('Gain (dB)'); title('Magnitude response');
subplot(2,2,2);
plot(f, gd, 'LineWidth', 1.5, 'Color', col);
set(gca, 'XScale', 'log'); grid on; xlim([0.1 20]);
xlabel('Frequency (Hz)'); ylabel('Group delay (ms)'); title('Delay vs frequency');
subplot(2,2,[3 4]); hold on;
plot(tt, u, 'k:'); plot(tt, y, 'LineWidth', 1.5, 'Color', col); yline(G.runOn, 'r--', 'run threshold');
grid on; xlim([-200 1600]); xlabel('Time from step (ms)'); ylabel('Fraction of MVC');
title(sprintf('Step response: start %.0f ms, stop %.0f ms, overshoot %.1f%%', ...
    startDelay, stopDelay, overshoot));
sgtitle([F.id ' ' F.name ': ' F.desc]);
saveFig(h, fullfile(outDir, 'Fig1_filter_design.png'), 'Resolution', 200);
fprintf('%s %s  step test: start %.0f ms, stop %.0f ms, overshoot %.1f%%\n', ...
    F.id, F.name, startDelay, stopDelay, overshoot);

%% ================= PART 2: all subjects =================
files = dir(fullfile(dataDir, 'S*_*_game_*.csv'));
if isempty(files), error('No recordings found in folder "%s".', dataDir); end
tok  = regexp({files.name}, '^(S\d+)_', 'tokens', 'once');
tok  = tok(~cellfun(@isempty, tok));
subs = unique(cellfun(@(c) c{1}, tok, 'UniformOutput', false));
nS   = numel(subs);
mus  = {'biceps','triceps'};

rows = {};  matchLog = {};
hPSD = figure('Name', 'Spectra', 'Position', [60 60 1000 220*nS]);
hBur = figure('Name', 'Before vs after', 'Position', [60 60 1100 240*nS]);

for s = 1:nS
    sub = subs{s};
    calFile = fullfile(dataDir, [sub '_calibration.json']);
    cal = [];
    if exist(calFile, 'file'), cal = cleanCal(jsondecode(fileread(calFile))); end
    fl = dir(fullfile(dataDir, [sub '_*_game_*.csv']));
    if isfield(ONLY_FILES, sub), fl = fl(strcmp({fl.name}, ONLY_FILES.(sub))); end
    if isempty(fl), warning('No data-collection file found for %s.', sub); continue; end
    [~, o] = sort({fl.name});  fl = fl(o);

    D = struct();
    for m = 1:2
        D.(mus{m}) = struct('raw',[],'clean',[],'cz',[],'rect',[],'env',[],'mav',[],'rms',[],'ref',[],'rest',false(0,1));
    end

    for k = 1:numel(fl)
        T = readtable(fullfile(dataDir, fl(k).name));
        if height(T) <= skip + fs, continue; end
        fid  = char(string(T.filter(1)));
        keep = (skip+1):height(T);
        for m = 1:2
            mu  = mus{m};
            x   = T.([mu '_raw_mV']);  x = x - mean(x);
            c   = filter(bN, aN, filter(bBP, aBP, x));          % causal, as in the game
            rc  = abs(c);
            env = applySmooth(rc, F);                           % <-- this file's filter
            mav = filter(ones(N,1)/N, 1, rc);                   % MAV feature
            rmsv = sqrt(filter(ones(N,1)/N, 1, c.^2));          % RMS feature
            cz  = filtfilt(bN, aN, filtfilt(bBP, aBP, x));      % zero-phase, reference only
            ref = filtfilt(bRef, aRef, abs(cz));
            rm  = restMask(cz, fs);

            % If this recording was made with this filter, check MATLAB matches what the game got
            if strcmp(fid, F.id) && ~isempty(cal) && isfield(cal.levels.(mu), F.id)
                L  = cal.levels.(mu).(F.id);
                en = min(max((env - L.rest)/(L.mvc - L.rest), 0), 1.5);
                rr = corrcoef(en(keep), T.([mu '_env'])(keep));
                matchLog(end+1, :) = {fl(k).name, mu, rr(1,2)}; %#ok<AGROW>
            end

            d = D.(mu);
            d.raw = [d.raw; x(keep)];    d.clean = [d.clean; c(keep)];  d.cz  = [d.cz; cz(keep)];
            d.rect = [d.rect; rc(keep)]; d.env = [d.env; env(keep)];
            d.mav = [d.mav; mav(keep)];  d.rms = [d.rms; rmsv(keep)];
            d.ref = [d.ref; ref(keep)];  d.rest = [d.rest; rm(keep)];
            D.(mu) = d;
        end
    end
    if isempty(D.biceps.ref), warning('%s: no usable files', sub); continue; end

    t    = (0:numel(D.biceps.ref)-1)'/fs;
    mins = numel(t)/fs/60;
    nrm  = struct();  raw0 = struct();
    for m = 1:2
        mu = mus{m};  d = D.(mu);
        [rest, mvc, src]    = calLevels(cal, mu, F.id, d.env, d.rest);
        nrm.(mu)  = min(max((d.env - rest)/(mvc - rest), 0), 1.5);
        [rest0, mvc0, ~]    = calLevels(cal, mu, 'F0', d.rect, d.rest);
        raw0.(mu) = min(max((d.rect - rest0)/(mvc0 - rest0), 0), 1.5);

        [cc, lags] = xcorr(d.env - mean(d.env), d.ref - mean(d.ref), 500);
        [~, i] = max(cc);
        on  = d.ref > 0.25*pct(d.ref, 99);
        hp  = filtfilt(bh, ah, d.env);
        rip = 100*std(hp(on))/mean(d.env(on));
        thr = G.runOn;  if m == 2, thr = G.kickOn; end
        fr  = 100*mean(nrm.(mu)(d.rest) > thr);
        [snr0, snr1] = snrPair(d.raw, d.cz, fs);
        % RMS / MAV ratio during activity (the 25% most active samples); about 1.25 for Gaussian-like EMG
        act = d.rms > pct(d.rms, 75);
        rmsMav = median(d.rms(act) ./ max(d.mav(act), eps));
        rows(end+1, :) = {sub, mu, [F.id ' ' F.name], lags(i), rip, fr, snr0, snr1, rmsMav, src}; %#ok<AGROW>

        % Spectra: raw vs band-pass + notch
        figure(hPSD); subplot(nS, 2, 2*(s-1) + m);
        [P1, fp] = pwelch(d.raw, hamming(1000), 500, 1000, fs);
        P2 = pwelch(d.clean, hamming(1000), 500, 1000, fs);
        semilogy(fp, P1, fp, P2); grid on; xlim([0 500]); title([sub ' ' mu]);
        if m == 1, ylabel('Power (mV^2/Hz)'); end
        if s == nS, xlabel('Frequency (Hz)'); end
        if s == 1 && m == 1, legend('Raw', 'Band-pass + notch'); end

        % Before (raw, F0) vs after (this filter) on the strongest burst
        [~, im] = max(d.ref);
        idx = max(1, im - 1500):min(numel(t), im + 1500);
        figure(hBur); subplot(nS, 2, 2*(s-1) + m); hold on;
        plot(t(idx) - t(idx(1)), raw0.(mu)(idx), 'Color', [.6 .6 .6]);
        plot(t(idx) - t(idx(1)), nrm.(mu)(idx), 'LineWidth', 1.5, 'Color', col);
        yline(thr, 'r--'); hold off; grid on; ylim([0 1.5]); title([sub ' ' mu]);
        if m == 1, ylabel('Fraction of MVC'); end
        if s == nS, xlabel('Time (s)'); end
        if s == 1 && m == 1, legend('Before (raw, F0)', ['After (' F.name ')']); end
    end

    % ---- Pick the part of the recording to show (ZOOM seconds) ----
    % (the zoom is only for the figures; all numbers above use the whole recording)
    w   = pickWindow(D.biceps.ref, D.triceps.ref, fs, ZOOM, ZOOM_START);
    tw  = t(w) - t(w(1));
    t0w = t(w(1));

    % ---- Per-subject figure: signal chain, zoomed ----
    h1 = figure('Name', [sub ' signal chain'], 'Position', [60 60 1150 1000]);
    for m = 1:2
        d = D.(mus{m});
        thr = G.runOn;  tlab = 'run threshold';
        if m == 2, thr = G.kickOn;  tlab = 'kick threshold'; end
        subplot(5,2,m);   plot(tw, d.raw(w), 'Color', [.3 .3 .3]); grid on; ylabel('mV');
        title(sprintf('%s %s: 1 Raw', sub, mus{m}));
        subplot(5,2,m+2); plot(tw, d.clean(w), 'b'); grid on; ylabel('mV');
        title('2 Band-pass 20-450 Hz + 50 Hz notch');
        subplot(5,2,m+4); plot(tw, d.rect(w), 'Color', [.45 .45 .45]); grid on; ylabel('mV');
        title('3 Rectified |x|');
        subplot(5,2,m+6); plot(tw, d.mav(w), tw, d.rms(w), 'LineWidth', 1.2); grid on; ylabel('mV');
        title('4 Features: MAV and RMS (100 ms)'); legend('MAV', 'RMS', 'Location', 'northeast');
        subplot(5,2,m+8); plot(tw, nrm.(mus{m})(w), 'Color', col, 'LineWidth', 1.5); hold on;
        yline(thr, 'r--', tlab); hold off; grid on; ylim([0 1.5]); ylabel('Fraction of MVC');
        title(['5 Envelope: ' F.name ' (normalised)']);
        xlabel(sprintf('Time (s), starting %.0f s into the recording', t0w));
    end
    sgtitle(sprintf('%s: signal after each step, %d s of the recording', sub, ZOOM));
    saveFig(h1, fullfile(outDir, ['Fig2_signal_chain_' sub '.png']), 'Resolution', 200);

    % ---- Per-subject figure: whole recording (overview), with the zoomed part marked ----
    h1b = figure('Name', [sub ' overview'], 'Position', [60 60 1150 500]);
    for m = 1:2
        subplot(2,1,m);
        plot(t, nrm.(mus{m}), 'Color', col); hold on;
        xline(t(w(1)), 'r-', 'zoom');  xline(t(w(end)), 'r-');  hold off;
        grid on; ylim([0 1.5]); ylabel('Fraction of MVC');
        title(sprintf('%s %s: %s envelope, whole recording', sub, mus{m}, F.name));
    end
    xlabel('Time (s)');
    saveFig(h1b, fullfile(outDir, ['Fig2b_overview_' sub '.png']), 'Resolution', 200);
end
saveFig(hPSD, fullfile(outDir, 'Fig4_spectra_all.png'), 'Resolution', 200);
saveFig(hBur, fullfile(outDir, 'Fig5_before_vs_after_all.png'), 'Resolution', 200);

%% ================= Tables and summary =================
Tres = cell2table(rows, 'VariableNames', {'Subject','Muscle','Filter','Delay_ms','Ripple_pct', ...
    'RestFalse_pct','SNR_raw_dB','SNR_filtered_dB','RMS_MAV_ratio','Normalisation'});
Tstep = table({[F.id ' ' F.name]}, startDelay, stopDelay, overshoot, 'VariableNames', ...
    {'Filter','StartDelay_ms','StopDelay_ms','Overshoot_pct'});
writetable(Tres,  fullfile(outDir, 'table_per_recording.csv'));
writetable(Tstep, fullfile(outDir, 'table_step_test.csv'));

lab = strcat(Tres.Subject, {' '}, Tres.Muscle);
h3 = figure('Name', 'Per-subject results', 'Position', [60 60 1100 750]);
subplot(3,1,1); bar(categorical(lab, lab), Tres.Delay_ms, 'FaceColor', col); grid on;
ylabel('Delay (ms)'); title([F.id ' ' F.name ': results for every subject']);
subplot(3,1,2); bar(categorical(lab, lab), Tres.Ripple_pct, 'FaceColor', col); grid on;
ylabel('Ripple (%)');
subplot(3,1,3); bar(categorical(lab, lab), Tres.RestFalse_pct, 'FaceColor', col); grid on;
ylabel('False triggers at rest (%)');
saveFig(h3, fullfile(outDir, 'Fig6_results_per_subject.png'), 'Resolution', 200);

fprintf('\n===== %s %s: per recording =====\n', F.id, F.name);  disp(Tres);
fprintf('Mean over all recordings: delay %.0f ms, ripple %.1f%%, false triggers at rest %.1f%%\n', ...
    mean(Tres.Delay_ms), mean(Tres.Ripple_pct), mean(Tres.RestFalse_pct));
if ~isempty(matchLog)
    fprintf('\n===== MATLAB vs envelope the game received (should be close to 1) =====\n');
    for i = 1:size(matchLog, 1), fprintf('%-40s %-8s r = %.4f\n', matchLog{i,:}); end
end
fprintf('\nFigures and tables saved in %s\n', outDir);

%% ================= Local functions (same in all four files) =================
function y = applySmooth(rc, F)
if strcmp(F.type, 'kalman_rms')
    y = kalmanRms(rc, F.N, F.q, F.r);
else
    y = filter(F.b, F.a, rc);
end
end

function y = kalmanRms(rc, N, q, r)
% F5, team member's design (same as class KalmanRms in bioradio_emg.py):
% 100 ms moving RMS, then a random-walk Kalman filter starting at the first value with P0 = R.
z = sqrt(filter(ones(N,1)/N, 1, rc.^2));
y = zeros(size(z));  x = z(1);  p = r;  y(1) = x;
for i = 2:numel(z)
    p = p + q;  k = p/(p + r);  x = x + k*(z(i) - x);  p = (1 - k)*p;  y(i) = x;
end
end

function cal = cleanCal(cal)
% Calibration files made before version v3 have old F5/F6/F7 values: drop them (they are then
% estimated from the data instead).
if ~isfield(cal, 'version') || ~strcmp(cal.version, 'v3')
    for mu = {'biceps', 'triceps'}
        for id = {'F5', 'F6', 'F7'}
            if isfield(cal.levels.(mu{1}), id{1})
                cal.levels.(mu{1}) = rmfield(cal.levels.(mu{1}), id{1});
            end
        end
    end
end
end

function [b, a] = notch50(fs)
% Same as scipy.signal.iirnotch(50, 30, fs)
w0 = 50/(fs/2);  bw = w0/30;
beta = tan(bw*pi/2);  g = 1/(1 + beta);
b = g*[1, -2*cos(w0*pi), 1];
a = [1, -2*g*cos(w0*pi), 2*g - 1];
end

function m = restMask(cz, fs)
% Rest = the quietest 10% of 500 ms windows
W = round(0.5*fs);  nW = floor(numel(cz)/W);
r = sqrt(mean(reshape(cz(1:nW*W).^2, W, nW), 1));
m = false(numel(cz), 1);
for j = find(r <= pct(r, 10))
    m((j-1)*W + 1 : j*W) = true;
end
end

function [rest, mvc, src] = calLevels(cal, mu, id, e, rm)
% Rest and MVC from the calibration file; estimated from the data if the file lacks this filter
if ~isempty(cal) && isfield(cal, 'levels') && isfield(cal.levels, mu) && isfield(cal.levels.(mu), id)
    rest = cal.levels.(mu).(id).rest;  mvc = cal.levels.(mu).(id).mvc;  src = 'calibration';
else
    rest = median(e(rm));  mvc = pct(e, 95);  src = 'estimated';
end
if mvc <= rest, mvc = rest + eps; end
end

function [s0, s1] = snrPair(x, y, fs)
% Activity-to-rest power ratio, raw (x) vs filtered (y), 500 ms windows
W  = round(0.5*fs);  nW = floor(numel(x)/W);
X  = reshape(x(1:nW*W), W, nW);  Y = reshape(y(1:nW*W), W, nW);
r  = sqrt(mean(Y.^2, 1));
rs = r <= pct(r, 10);  ac = r >= pct(r, 90);
s0 = 10*log10(mean(mean(X(:,ac).^2)) / mean(mean(X(:,rs).^2)));
s1 = 10*log10(mean(mean(Y(:,ac).^2)) / mean(mean(Y(:,rs).^2)));
end

function w = pickWindow(rb, rt, fs, Z, zStart)
% Samples to show: Z seconds, either from zStart (s) or the most active part of the recording
% (biceps + triceps), avoiding the relax period at the start and the last 15 s.
n = numel(rb);  L = min(n, round(Z*fs));
if ~isempty(zStart)
    i0 = round(zStart*fs) + 1;
else
    a = rb/max(pct(rb, 99), eps) + rt/max(pct(rt, 99), eps);
    m = filter(ones(L,1)/L, 1, a);               % activity over the last Z seconds
    m(1:min(n, L + 25*fs)) = -inf;
    m(max(1, n - 15*fs):end) = -inf;
    [mx, iEnd] = max(m);
    if isinf(mx), iEnd = L; end
    i0 = iEnd - L + 1;
end
i0 = min(max(1, i0), max(1, n - L + 1));
w  = i0:(i0 + L - 1);
end

function v = pct(x, p)
% Percentile without the Statistics Toolbox
x = sort(x(~isnan(x(:))));
if isempty(x), v = NaN; return; end
v = x(max(1, min(numel(x), round(p/100*numel(x)))));
end

function saveFig(h, f, varargin)
% Save a figure with a white (light) background, whatever MATLAB's own theme is
if ~isgraphics(h)
    warning('Figure for %s was closed before saving; skipped.', f); return;
end
try
    theme(h, 'light');          % MATLAB R2025a and newer
catch
end
try
    exportgraphics(h, f, varargin{:});
catch err
    warning('exportgraphics failed for %s (%s); saving with print instead.', f, err.message);
    set(h, 'Color', 'w', 'InvertHardcopy', 'off');
    print(h, f, '-dpng', '-r200');
end
end
