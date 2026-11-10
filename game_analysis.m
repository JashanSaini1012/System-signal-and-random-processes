%% game_analysis.m
% Project 14 (SSRP): game results for every smoothing filter (the game member's analysis).
%
% Reads the Unity game logs (GameLogs/) and the Python EMG recordings (data/) and makes the
% tables and figures for the game part of the report.
%
%   1. Labels     which filter Python was REALLY running during each game round
%                 (the game's Filter box is only a label and can be wrong)
%   2. Football   goals, shots, tackles, ... per filter                         -> FigG1
%   3. Smoothness how jittery the player's run command was, per filter           -> FigG2
%   4. Response   each filter's own delay (step test) + the reaction test;
%                 reaction times of filters that were not tested are ESTIMATED   -> FigG3, FigG5
%   5. Simulation every filter, including F0 Raw, on the SAME recorded EMG of the tester,
%                 passed through the game's run/kick rules (raw vs processed)    -> FigG4, FigG6
%
% Folder layout:  game_analysis.m,  data/ (S*_*_game_*.csv, S*_calibration.json),  GameLogs/
% Needs the Signal Processing Toolbox. Results are saved in ./results_game

clear; clc; close all;
set(groot, 'defaultAxesToolbarVisible', 'off');   % keep the axes toolbar out of saved figures

%% ================= Settings =================
dataDir = 'data';
logDir  = 'GameLogs';
GAME_SUBJECT = 'S2';   % the game tester: only this subject's rounds and recordings are analysed
EXCLUDE_ROUNDS = {'F6'};  % F6 (a Chebyshev I low-pass made by the team) is no member's design: its round
                          % is listed in the label table but left out of the results. The member's
                          % Chebyshev I design is the pipeline 'DN', evaluated offline (section 5).
outDir  = fullfile(pwd, 'results_game');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fs = 1000;

% Game rules (UdpEmgControl.cs), fractions of MVC
G.runOn = 0.10;  G.runFull = 0.60;
G.kickOn = 0.30; G.kickOff = 0.20; G.kickWin = 0.30; G.lockout = 0.70;

ids = {'F0','F2','F3','F5','DN','F7'};                       % filters in the simulation
nameMap = containers.Map({'F0','F1','F2','F3','F4','F5','F6','F7','DN'}, ...
    {'F0 Raw','F1 MAV','F2 Moving RMS','F3 Butterworth','F4 EMA','F5 Kalman (RMS)', ...
     'F6 Chebyshev I','F7 Chebyshev II','Chebyshev I pipeline'});
colMap = containers.Map(ids, {[.62 .62 .62], [.55 .34 .29], [.12 .47 .71], ...
    [.17 .63 .17], [.84 .15 .16], [.58 .40 .74]});

% Filters, identical to bioradio_emg.py
[flt.bBP, flt.aBP] = butter(4, [20 450]/(fs/2), 'bandpass');
[flt.bN,  flt.aN ] = notch50(fs);
[flt.b3,  flt.a3 ] = butter(2, 4/(fs/2));            % F3 Butterworth
[flt.b6,  flt.a6 ] = cheby1(2, 1, 3.285/(fs/2));     % F6 Chebyshev I
[flt.b7,  flt.a7 ] = cheby2(2, 40, 28.35/(fs/2));    % F7 Chebyshev II
flt.N = 100;  flt.kq = 1e-2;  flt.kr = 1;            % F2 RMS window, F5 Kalman on RMS
% 'DN': Dhruval Naik's Chebyshev I pipeline, exactly as in his code: Chebyshev I band-pass
% (order 4, 0.5 dB, 20-450 Hz), comb notch at 50/150/250 Hz, 120 ms Hamming RMS, tanh soft clip
[flt.bC1, flt.aC1] = cheby1(4, 0.5, [20 450]/(fs/2), 'bandpass');
fc = [50 150 250];  flt.bC = zeros(3,3);  flt.aC = zeros(3,3);
for m = 1:3
    w0 = 2*pi*fc(m)/fs;  rr = 1 - (w0/35)/2;
    bn = [1, -2*cos(w0), 1];  an = [1, -2*rr*cos(w0), rr^2];
    flt.bC(m,:) = bn*(sum(an)/sum(bn));  flt.aC(m,:) = an;
end
NH = round(0.120*fs);  wH = 0.54 - 0.46*cos(2*pi*(0:NH-1)'/(NH-1));  flt.hH = wH/sum(wH);

%% ================= 1. Python recordings: which filter was really running =================
pf = dir(fullfile(dataDir, [GAME_SUBJECT '_*_game_*.csv']));
if isempty(pf), error('No recordings in "%s".', dataDir); end
PY = struct('file', {}, 'subject', {}, 'start', {}, 'dur', {}, 'filt', {}, 'b', {}, 't', {});
for k = 1:numel(pf)
    p = fullfile(dataDir, pf(k).name);
    opts = detectImportOptions(p);
    opts.SelectedVariableNames = {'t_s', 'filter', 'biceps_raw_mV', 'triceps_raw_mV'};
    T = readtable(p, opts);
    PY(end+1) = struct('file', pf(k).name, 'subject', strtok(pf(k).name, '_'), ...
        'start', stamp(pf(k).name), 'dur', T.t_s(end) - T.t_s(1), 'filt', {string(T.filter)}, ...
        'b', T.biceps_raw_mV, 't', T.triceps_raw_mV); %#ok<SAGROW>
    fprintf('Read %s\n', pf(k).name);
end

%% ================= 2-3. Game logs: labels, football, smoothness, reaction test =================
gl = dir(fullfile(logDir, [GAME_SUBJECT '_*.csv']));
labRows = {};  allStarts = {};  footRows = {};  smRows = {};  SM = struct('subject', {}, 'filter', {}, 'start', {}, 't', {}, 'run', {});
RT = [];  rtCues = [];  rtFilter = '';
for k = 1:numel(gl)
    name = gl(k).name;  p = fullfile(logDir, name);
    S = readSummary(p);
    if isempty(S), continue; end
    isRT = contains(name, '_RT_');
    st = stamp(name);
    len = 300;  if isRT, len = 110; elseif isfield(S, 'round_s'), len = str2double(S.round_s); end
    [act, pyfile] = actualFilter(PY, S.subject, st, len, fs);
    logged = S.filter;
    if isempty(act), act = logged; end
    fixed = name;
    if ~strcmp(act, logged), fixed = regexprep(name, ['_' logged '_'], ['_' act '_'], 'once'); end
    kind = 'football';  if isRT, kind = 'reaction test'; end
    labRows(end+1, :) = {name, kind, S.subject, logged, act, pyfile, fixed}; %#ok<AGROW>

    if isRT
        RT = S;  rtFilter = act;
        rtCues = readtable(p, 'CommentStyle', '#');
        continue;
    end
    allStarts(end+1, :) = {S.subject, st}; %#ok<AGROW>      % every football round, for the play order
    if ismember(act, EXCLUDE_ROUNDS), continue; end   % listed in the label table only

    T = readtable(p, 'CommentStyle', '#');
    ev = string(T.event);  ev(ismissing(ev)) = "";
    kp = str2double(extractAfter(ev(startsWith(ev, "kick:")), "kick:"));
    goals = str2double(S.goals);  shots = str2double(S.shots);
    footRows(end+1, :) = {S.subject, act, str2double(S.played_s), goals, shots, 100*goals/max(shots,1), ...
        str2double(S.saves), str2double(S.misses), str2double(S.tackles), str2double(S.attacks), ...
        str2double(S.kicks_without_ball), str2double(S.avg_time_to_goal_s), mean(kp), st, name}; %#ok<AGROW>

    % Smoothness of the run command, resampled to 100 Hz, only while the player has the ball
    [tt0, iu] = unique(T.t_s);
    run0  = T.run(iu);
    poss0 = double(strcmp(T.state(iu), 'Possession'));
    tt = (tt0(1):0.01:tt0(end))';
    runI  = interp1(tt0, run0, tt);
    possI = interp1(tt0, poss0, tt) > 0.5;
    on = runI > 0 & possI;
    [starts, flick] = runStats(on, 100, (tt(end) - tt(1))/60);
    smRows(end+1, :) = {S.subject, act, starts, flick, 100*mean(on), ripplePct(runI, on, 100), name}; %#ok<AGROW>
    SM(end+1) = struct('subject', S.subject, 'filter', act, 'start', st, 't', tt, 'run', runI); %#ok<SAGROW>
end

Tlab  = cell2table(labRows, 'VariableNames', {'GameLog','Kind','Subject','LoggedFilter','ActualFilter', ...
    'PythonRecording','CorrectedName'});
Tfoot = cell2table(footRows, 'VariableNames', {'Subject','Filter','Played_s','Goals','Shots','Shooting_pct', ...
    'Saves','Misses','Tackles','Attacks','KicksWithoutBall','AvgTimeToGoal_s','MeanKickPower','Start','File'});
Tfoot = sortrows(Tfoot, 'Start');
Tfoot.OrderPlayed = zeros(height(Tfoot), 1);
subsF = unique(Tfoot.Subject);
for i = 1:height(Tfoot)          % order among ALL football rounds played, excluded ones included
    mine = allStarts(strcmp(allStarts(:,1), Tfoot.Subject{i}), 2);
    Tfoot.OrderPlayed(i) = nnz([mine{:}] <= Tfoot.Start(i));
end
Tsm = cell2table(smRows, 'VariableNames', {'Subject','Filter','RunStarts_per_min','Flickers_per_min', ...
    'Running_pct','RunRipple_pct','File'});

%% ================= 4. Response time =================
t0 = 3*fs;  t1 = 4*fs;
u = [zeros(t0,1); 0.5*ones(t1 - t0,1); zeros(2*fs,1)];
nF = numel(ids);
runStart = zeros(nF,1); runStop = runStart; kickDel = runStart; overS = runStart;
for i = 1:nF
    y = envelope1(u, ids{i}, flt);
    y = y / (y(t1)/0.5);
    runStart(i) = find(y(t0+1:end) >= G.runOn, 1) - 1;
    runStop(i)  = find(y(t1+1:end) <= G.runOn, 1) - 1;
    kickDel(i)  = (find(y(t0+1:end) >= G.kickOn, 1) - 1) + 1000*G.kickWin;
    overS(i)    = 100*(max(y(t0+1:t1))/0.5 - 1);
end
Tresp = table(ids', runStart, runStop, kickDel, overS, 'VariableNames', ...
    {'Filter','RunStartDelay_ms','RunStopDelay_ms','KickDelay_ms','Overshoot_pct'});
human = [];
if ~isempty(RT)
    j = find(strcmp(ids, rtFilter));
    human.start = str2double(RT.run_onset_median_ms)   - runStart(j);
    human.stop  = str2double(RT.run_release_median_ms) - runStop(j);
    human.kick  = str2double(RT.kick_median_ms)        - kickDel(j);
    Tresp.StartReaction_ms = human.start + runStart;
    Tresp.StopReaction_ms  = human.stop  + runStop;
    Tresp.KickReaction_ms  = human.kick  + kickDel;
    Tresp.Measured = strcmp(ids', rtFilter);
end

%% ================= 5. Offline simulation: every filter on the same recorded EMG =================
subs = unique({PY.subject});
simRows = {};  example = [];
for s = 1:numel(subs)
    sub = subs{s};
    calFile = fullfile(dataDir, [sub '_calibration.json']);
    if ~exist(calFile, 'file'), continue; end
    cal = jsondecode(fileread(calFile));
    if ~isfield(cal, 'version') || ~strcmp(cal.version, 'v3')
        fprintf('%s: calibration file is older than v3, skipped in the simulation\n', sub);
        continue;
    end
    acc = struct();
    for i = 1:nF, acc.(ids{i}).b = [];  acc.(ids{i}).t = []; end
    for r = find(strcmp({PY.subject}, sub))
        for mm = 1:2
            if mm == 1, x = PY(r).b; mu = 'biceps'; fld = 'b'; else, x = PY(r).t; mu = 'triceps'; fld = 't'; end
            c  = filter(flt.bN, flt.aN, filter(flt.bBP, flt.aBP, x - x(1)));
            rc = abs(c);
            for i = 1:nF
                if strcmp(ids{i}, 'DN')
                    % his chain works on the raw signal and normalises with the F2 levels + tanh
                    e = dnEnvelope(x, flt);
                    Lv = cal.levels.(mu).F2;
                    n = 1.5*tanh(max((e - Lv.rest)/max(Lv.mvc - Lv.rest, 1e-9), 0)/1.5);
                else
                    e = envelope1(rc, ids{i}, flt);
                    Lv = cal.levels.(mu).(ids{i});
                    n = min(max((e - Lv.rest)/max(Lv.mvc - Lv.rest, 1e-9), 0), 1.5);
                end
                acc.(ids{i}).(fld) = [acc.(ids{i}).(fld); n(fs+1:end)];
            end
        end
    end
    for i = 1:nF
        b = acc.(ids{i}).b;  tr = acc.(ids{i}).t;
        mins = numel(b)/fs/60;
        on = b > G.runOn;
        [starts, flick] = runStats(on, fs, mins);
        kk = kickSim(tr, fs, G);
        simRows(end+1, :) = {sub, ids{i}, mins, starts, flick, 100*mean(on), numel(kk)/mins, ...
            ripplePct(b, on, fs)}; %#ok<AGROW>
    end
    if strcmp(sub, 'S2')
        for i = 1:nF, example.(ids{i}) = acc.(ids{i}).b; end
    end
    fprintf('Simulated %s\n', sub);
end
Tsim = cell2table(simRows, 'VariableNames', {'Subject','Filter','Minutes','RunStarts_per_min', ...
    'Flickers_per_min','Running_pct','Kicks_per_min','BicepsRipple_pct'});

% Run rule with and without hysteresis, for Butterworth and the Chebyshev I pipeline (S2)
ruleRows = {};
if ~isempty(example)
    rules = {'game: on/off at 0.10', 0.10, 0.10; 'hysteresis: on 0.10, off 0.072', 0.10, 0.072; ...
             'Schmitt: on 0.25, off 0.18', 0.25, 0.18};
    for f = {'F3', 'DN'}
        b = example.(f{1});  mins = numel(b)/fs/60;
        for q = 1:size(rules, 1)
            on = hysteresisOn(b, rules{q,2}, rules{q,3});
            [starts, flick] = runStats(on, fs, mins);
            ruleRows(end+1, :) = {nameMap(f{1}), rules{q,1}, flick, starts}; %#ok<AGROW>
        end
    end
end
Trule = cell2table(ruleRows, 'VariableNames', {'Envelope','RunRule','Flickers_per_min','RunStarts_per_min'});

%% ================= Save tables =================
writetable(Tlab,  fullfile(outDir, 'labels_corrected.csv'));
writetable(removevars(Tfoot, 'Start'), fullfile(outDir, 'table_football.csv'));
writetable(Tsm,   fullfile(outDir, 'table_smoothness_game.csv'));
writetable(Tresp, fullfile(outDir, 'table_response_time.csv'));
writetable(Tsim,  fullfile(outDir, 'table_offline_simulation.csv'));
writetable(Trule, fullfile(outDir, 'table_run_rules.csv'));

%% ================= Figures =================
% FigG1: football results of the tester (the subject with the most rounds)
[~, ib] = max(cellfun(@(s) nnz(strcmp(Tfoot.Subject, s)), subsF));
tester = subsF{ib};
F2 = Tfoot(strcmp(Tfoot.Subject, tester), :);
ordTxt = {'1st','2nd','3rd','4th','5th','6th','7th','8th'};
lab1 = cell(height(F2), 1);  col1 = zeros(height(F2), 3);
for i = 1:height(F2)
    lab1{i} = sprintf('%s (%s)', nameMap(F2.Filter{i}), ordTxt{F2.OrderPlayed(i)});
    col1(i,:) = colorOf(colMap, F2.Filter{i});
end
h = figure('Name', 'Football', 'Position', [40 40 1250 780]);
subplot(2,2,1); barLabel(F2.Goals, lab1, col1, '%.0f'); title('Goals in 5 minutes');
subplot(2,2,2); barLabel(F2.Shooting_pct, lab1, col1, '%.0f'); title('Shooting accuracy (goals / shots, %)');
subplot(2,2,3); barLabel(F2.Tackles, lab1, col1, '%.0f'); title('Tackles (player ran into a defender)');
subplot(2,2,4); barLabel(F2.AvgTimeToGoal_s, lab1, col1, '%.1f'); title('Average time to score (s)');
sgtitle(sprintf('%s football results per smoothing filter (order played in brackets)', tester));
saveFig(h, fullfile(outDir, 'FigG1_football_results.png'), 'Resolution', 200);

% FigG2: smoothness of the run command (tester)
sm = SM(strcmp({SM.subject}, tester));
[~, o] = sort([sm.start]);  sm = sm(o);
n2 = numel(sm);
h = figure('Name', 'Smoothness', 'Position', [40 40 1150 260*(n2+1)/1.6]);
subplot(n2+1, 1, 1);
tsel = Tsm(strcmp(Tsm.Subject, tester), :);
[~, o2] = ismember({sm.filter}, tsel.Filter);
bar([tsel.RunStarts_per_min(o2), tsel.Flickers_per_min(o2)]);
set(gca, 'XTick', 1:n2, 'XTickLabel', cellfun(@(f) nameMap(f), {sm.filter}, 'UniformOutput', false));
legend('run starts / min', 'flickers / min (stops < 0.2 s)', 'Location', 'northeast'); grid on;
title(sprintf('Smoothness of the run command (%s, from the game logs)', tester));
for i = 1:n2
    subplot(n2+1, 1, i+1);
    m = sm(i).t >= 60 & sm(i).t < 80;
    plot(sm(i).t(m) - 60, sm(i).run(m), 'Color', colorOf(colMap, sm(i).filter), 'LineWidth', 1.1);
    ylim([0 1.05]); grid on; ylabel(nameMap(sm(i).filter), 'FontSize', 8);
end
xlabel('Time (s), 20 s of each round');
saveFig(h, fullfile(outDir, 'FigG2_smoothness.png'), 'Resolution', 200);

% FigG3: response time
labR = cellfun(@(f) nameMap(f), ids, 'UniformOutput', false);
h = figure('Name', 'Response time', 'Position', [40 40 1300 480]);
subplot(1,2,1);
bar([Tresp.RunStartDelay_ms, Tresp.RunStopDelay_ms]); grid on;
set(gca, 'XTick', 1:nF, 'XTickLabel', labR, 'XTickLabelRotation', 15);
ylabel('ms'); legend('start running', 'stop running', 'Location', 'northwest');
title('Delay added by the filter alone (step test)');
if ~isempty(human)
    subplot(1,2,2);
    bar([Tresp.StartReaction_ms, Tresp.StopReaction_ms, Tresp.KickReaction_ms]); grid on;
    set(gca, 'XTick', 1:nF, 'XTickLabel', labR, 'XTickLabelRotation', 15);
    ylabel('ms'); legend('start', 'stop', 'kick', 'Location', 'northwest');
    ylim([0 1.2*max(Tresp.KickReaction_ms)]);
    for i = 1:nF
        if Tresp.Measured(i), tx = 'measured'; else, tx = 'estimated'; end
        text(i, 1.12*max(Tresp.KickReaction_ms), tx, 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    title('Reaction time in the game: cue to action (person + filter)');
end
saveFig(h, fullfile(outDir, 'FigG3_response_time.png'), 'Resolution', 200);

% FigG5: reaction test, every cue
if ~isempty(rtCues)
    h = figure('Name', 'Reaction test', 'Position', [40 40 1000 420]);
    isRun = strcmp(rtCues.type, 'run');  isKick = strcmp(rtCues.type, 'kick');
    plot(rtCues.cue(isRun), rtCues.onset_ms(isRun), 'o', 'LineWidth', 1.3); hold on;
    plot(rtCues.cue(isRun), rtCues.release_ms(isRun), 's', 'LineWidth', 1.3);
    plot(rtCues.cue(isKick), rtCues.onset_ms(isKick), '^', 'LineWidth', 1.3); hold off;
    grid on; xlabel('Cue number'); ylabel('ms');
    legend('start running (FLEX cue)', 'stop running (RELAX cue)', 'kick (KICK cue)', 'Location', 'best');
    title(sprintf('Reaction test, %s with %s: every cue', RT.subject, nameMap(rtFilter)));
    saveFig(h, fullfile(outDir, 'FigG5_reaction_test_cues.png'), 'Resolution', 200);
end

% FigG4: offline simulation, mean over subjects
if ~isempty(Tsim)
    M = zeros(nF, 4);
    for i = 1:nF
        r = strcmp(Tsim.Filter, ids{i});
        M(i,:) = [mean(Tsim.Flickers_per_min(r)), mean(Tsim.RunStarts_per_min(r)), ...
                  mean(Tsim.Kicks_per_min(r)), mean(Tsim.BicepsRipple_pct(r))];
    end
    colS = cell2mat(cellfun(@(f) colorOf(colMap, f), ids', 'UniformOutput', false));
    h = figure('Name', 'Simulation', 'Position', [40 40 1250 780]);
    subplot(2,2,1); barLabel(M(:,1), labR, colS, '%.1f', true);
    title('Flickers per minute (run stops < 0.2 s), log scale');
    subplot(2,2,2); barLabel(M(:,2), labR, colS, '%.1f', true);
    title('Run starts per minute, log scale');
    subplot(2,2,3); barLabel(M(:,3), labR, colS, '%.1f'); title('Kicks per minute');
    subplot(2,2,4); barLabel(M(:,4), labR, colS, '%.0f'); title('Ripple of the biceps level (%)');
    sgtitle(sprintf('Offline game simulation: every filter on the SAME recorded EMG (%s, mean)', ...
        strjoin(unique(Tsim.Subject)', ', ')));
    saveFig(h, fullfile(outDir, 'FigG4_simulation_all_filters.png'), 'Resolution', 200);
end

% FigG6: the same 8 s of EMG through each filter
if ~isempty(example)
    b3 = example.F3;
    act4 = filter(ones(4*fs,1)/(4*fs), 1, b3);          % trailing 4 s mean
    [~, iEnd] = max(act4);
    i0 = max(1, iEnd - 6*fs);  w = i0:min(numel(b3), i0 + 8*fs - 1);  tw = (0:numel(w)-1)'/fs;
    h = figure('Name', 'Raw vs filtered', 'Position', [40 40 1150 1000]);
    for i = 1:nF
        subplot(nF, 1, i);
        lw = 1.4;  if i == 1, lw = 0.6; end
        plot(tw, example.(ids{i})(w), 'Color', colorOf(colMap, ids{i}), 'LineWidth', lw); hold on;
        yline(G.runOn, 'r--'); hold off; ylim([0 1.5]); grid on; ylabel(nameMap(ids{i}), 'FontSize', 8);
        if i == 1, title('S2 biceps level with each filter, same 8 s of EMG (red line = run threshold)'); end
    end
    xlabel('Time (s)');
    saveFig(h, fullfile(outDir, 'FigG6_raw_vs_filtered_example.png'), 'Resolution', 200);
end

%% ================= Print =================
fprintf('\n== Labels (logged in the game vs filter Python was really running) ==\n');  disp(Tlab(:, [1 4 5 7]));
fprintf('== Football ==\n');  disp(Tfoot(:, {'Subject','OrderPlayed','Filter','Goals','Shots','Shooting_pct', ...
    'Tackles','KicksWithoutBall','AvgTimeToGoal_s','MeanKickPower'}));
fprintf('== Smoothness of the run command in the game ==\n');  disp(Tsm(:, 1:6));
fprintf('== Response time ==\n');  disp(Tresp);
if ~isempty(human)
    fprintf('Person''s own reaction (from the %s reaction test): start %.0f ms, stop %.0f ms, kick %.0f ms\n', ...
        rtFilter, human.start, human.stop, human.kick);
end
fprintf('\n== Offline simulation ==\n');  disp(Tsim);
fprintf('== Run rule with and without hysteresis (S2) ==\n');  disp(Trule);
fprintf('Saved in %s\n', outDir);

%% ================= Local functions =================
function d = stamp(name)
tok = regexp(name, '(\d{8}_\d{6})', 'tokens', 'once');
if isempty(tok), d = NaT; else, d = datetime(tok{1}, 'InputFormat', 'yyyyMMdd_HHmmss'); end
end

function S = readSummary(p)
% The game writes one summary line at the end: "# subject=S2 filter=F3 goals=21 ..."
S = [];
tok = regexp(fileread(p), '# subject=[^\r\n]*', 'match', 'once');
if isempty(tok), return; end
S = struct();
parts = strsplit(strtrim(tok(3:end)), ' ');
for i = 1:numel(parts)
    kv = strsplit(parts{i}, '=');
    if numel(kv) == 2, S.(matlab.lang.makeValidName(kv{1})) = kv{2}; end
end
end

function [act, pyfile] = actualFilter(PY, sub, st, len, fs)
% Filter that Python was running during [st, st + len] (wall clock)
act = '';  pyfile = '';
for r = 1:numel(PY)
    if ~strcmp(PY(r).subject, sub) || isnat(PY(r).start), continue; end
    off = seconds(st - PY(r).start);
    if off >= -5 && off <= PY(r).dur
        a = floor(max(off, 0)*fs) + 1;
        b = min(numel(PY(r).filt), floor(min(off + len, PY(r).dur)*fs));
        seg = PY(r).filt(a:max(a, b));
        [uq, ~, ic] = unique(seg);
        [~, im] = max(accumarray(ic, 1));
        act = char(uq(im));  pyfile = PY(r).file;
        return;
    end
end
end

function e = envelope1(rc, id, flt)
% Smoothing filters of bioradio_emg.py, applied to the rectified signal
switch id
    case 'F0', e = rc;
    case 'F2', e = sqrt(filter(ones(flt.N,1)/flt.N, 1, rc.^2));
    case 'F3', e = filter(flt.b3, flt.a3, rc);
    case 'F5', e = kalmanRms(rc, flt.N, flt.kq, flt.kr);
    case 'F6', e = filter(flt.b6, flt.a6, rc);
    case 'F7', e = filter(flt.b7, flt.a7, rc);
    case 'DN', e = sqrt(max(filter(flt.hH, 1, rc.^2), 0));   % step test of his Hamming RMS
    otherwise, error('Unknown filter %s', id);
end
end

function e = dnEnvelope(x, flt)
% Dhruval Naik's chain on the raw signal: Chebyshev I band-pass, comb notch, 120 ms Hamming RMS
y = filter(flt.bC1, flt.aC1, x);
for m = 1:3, y = filter(flt.bC(m,:), flt.aC(m,:), y); end
e = sqrt(max(filter(flt.hH, 1, y.^2), 0));
end

function on = hysteresisOn(x, onT, offT)
% on above onT, off again only below offT (offT = onT gives a plain threshold)
on = false(size(x));  s = false;
for i = 1:numel(x)
    if ~s && x(i) > onT, s = true; elseif s && x(i) <= offT, s = false; end
    on(i) = s;
end
end

function y = kalmanRms(rc, N, q, r)
z = sqrt(filter(ones(N,1)/N, 1, rc.^2));
y = zeros(size(z));  x = z(1);  p = r;  y(1) = x;
for i = 2:numel(z)
    p = p + q;  k = p/(p + r);  x = x + k*(z(i) - x);  p = (1 - k)*p;  y(i) = x;
end
end

function [b, a] = notch50(fs)
w0 = 50/(fs/2);  bw = w0/30;
beta = tan(bw*pi/2);  g = 1/(1 + beta);
b = g*[1, -2*cos(w0*pi), 1];
a = [1, -2*g*cos(w0*pi), 2*g - 1];
end

function [starts, flick] = runStats(on, fs, minutes)
% Run starts per minute, and 'flickers' (stops shorter than 0.2 s) per minute
d  = diff([0; on(:); 0]);
st = find(d == 1);  en = find(d == -1);
gaps = st(2:end) - en(1:end-1);
starts = numel(st)/minutes;
flick  = nnz(gaps < 0.2*fs)/minutes;
end

function v = ripplePct(x, on, fs)
% Fast wobble (above 3 Hz) of a signal while it is on, as % of its mean
if nnz(on) < fs, v = NaN; return; end
[bh, ah] = butter(2, 3/(fs/2), 'high');
hp = filtfilt(bh, ah, x);
v  = 100*std(hp(on))/max(mean(x(on)), 1e-9);
end

function k = kickSim(tri, fs, G)
k = [];  st = 0;  tmr = 0;  dt = 1/fs;
for i = 1:numel(tri)
    switch st
        case 0
            if tri(i) >= G.kickOn, st = 1; tmr = 0; end
        case 1
            tmr = tmr + dt;
            if tmr >= G.kickWin || tri(i) < G.kickOff
                k(end+1) = i; st = 2; tmr = 0; %#ok<AGROW>
            end
        case 2
            tmr = tmr + dt;
            if tmr >= G.lockout && tri(i) < G.kickOff, st = 0; end
    end
end
end

function c = colorOf(colMap, f)
if isKey(colMap, f), c = colMap(f); else, c = [.4 .4 .4]; end
end

function barLabel(vals, labels, cols, fmt, useLog)
% Bar chart with one colour per bar and the value written on top (optionally log scale)
if nargin < 5, useLog = false; end
vals = vals(:);
if useLog
    bh = bar(vals, 'FaceColor', 'flat', 'BaseValue', min([vals(vals > 0); 1])/2);
    set(gca, 'YScale', 'log');
else
    bh = bar(vals, 'FaceColor', 'flat');
end
bh.CData = cols;
for i = 1:numel(vals)
    text(i, vals(i), sprintf(fmt, vals(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 8);
end
set(gca, 'XTick', 1:numel(vals), 'XTickLabel', labels, 'XTickLabelRotation', 15, 'FontSize', 8);
grid on;
end

function saveFig(h, f, varargin)
% Save a figure with a white (light) background, whatever MATLAB's own theme is
try
    theme(h, 'light');          % MATLAB R2025a and newer
catch
end
exportgraphics(h, f, varargin{:});
end
