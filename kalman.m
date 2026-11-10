
%% filter_with_graphs.m
% EMG filtering + Kalman comparison for S1-S4
% Original F0-F4 preserved, F5 Kalman added

clear; clc; close all;

fs = 1000;

[b, a]   = butter(4, [20 450]/(fs/2), 'bandpass');
[bl, al] = butter(2, 4/(fs/2));
[bh, ah] = butter(2, 3/(fs/2), 'high');

N = 100;  % 100 ms window

subs = {'S1','S2','S3','S4'};
mus  = {'biceps','triceps'};

names = {'F0 raw','F1 MAV','F2 RMS', ...
         'F3 LP 4Hz','F4 EMA','F5 Kalman'};

figDir = fullfile(pwd,'figures');

if ~exist(figDir,'dir')
    mkdir(figDir);
end

R = [];
labels = {};

% Kalman tuning factors
% These are scaled according to each RMS signal's variance
Qfactor = 1e-4;
Rfactor = 1e-2;

for s = 1:numel(subs)

    % ---- Load all CSV parts ----

    files = dir(fullfile('data', ...
                    [subs{s} '_*_game_*.csv']));

    if isempty(files)
        error('No game CSV files for %s',subs{s});
    end

    T = table();

    for k = 1:numel(files)
        T = [T; readtable(fullfile('data',files(k).name))];
    end

    t = (0:height(T)-1)'/fs;

    % ---- Process both muscles ----

    for m = 1:2

        x = T.([mus{m} '_raw_mV']);
        x = x - mean(x);

        % ====================================
        % ZERO-PHASE FILTERING
        % ====================================

        xf = filtfilt(b,a,x);

        env = sqrt(movmean(xf.^2,N));

        % ====================================
        % SNR BEFORE AND AFTER
        % ====================================

        W = 500;
        nW = floor(numel(x)/W);

        X = reshape(x(1:nW*W),W,nW);
        Y = reshape(xf(1:nW*W),W,nW);

        r = sqrt(mean(Y.^2));
        sr = sort(r);

        rest = r <= sr(max(1,round(0.1*nW)));
        act  = r >= sr(max(1,round(0.9*nW)));

        snr0 = 10*log10( ...
            mean(X(:,act).^2,'all') / ...
            mean(X(:,rest).^2,'all'));

        snr1 = 10*log10( ...
            mean(Y(:,act).^2,'all') / ...
            mean(Y(:,rest).^2,'all'));

        % ====================================
        % CAUSAL FILTERING F0-F4
        % ====================================

        yc = filter(b,a,x);

        rc = abs(yc);

        % F0: Rectified raw
        F0 = rc;

        % F1: MAV 100 ms
        F1 = filter(ones(N,1)/N,1,rc);

        % F2: RMS 100 ms
        F2 = sqrt(filter(ones(N,1)/N,1,yc.^2));

        % F3: 4 Hz low-pass
        F3 = filter(bl,al,rc);

        % F4: EMA
        F4 = filter(0.01,[1 -0.99],rc);

        % ====================================
        % F5: KALMAN FILTER ON RMS
        % ====================================

        z = F2;

        signalVar = var(z,1);

        % Scale noise parameters to signal amplitude
        Qk = max(Qfactor*signalVar,1e-12);
        Rk = max(Rfactor*signalVar,1e-12);

        F5 = zeros(size(z));

        % Initial state
        xhat = z(1);
        P = Rk;

        F5(1) = xhat;

        for i = 2:length(z)

            % Prediction
            xPred = xhat;
            PPred = P + Qk;

            % Kalman gain
            K = PPred/(PPred + Rk);

            % Measurement update
            xhat = xPred + K*(z(i)-xPred);

            % Covariance update
            P = (1-K)*PPred;

            F5(i) = xhat;

        end

        % ====================================
        % COMBINE ALL FILTER OUTPUTS
        % ====================================

        E = [F0 F1 F2 F3 F4 F5];

        % Zero-phase reference
        ref = filtfilt(bl,al,abs(xf));

        on = ref > 0.25*max(ref);

        % ====================================
        % DELAY AND RIPPLE
        % ====================================

        d = zeros(1,6);
        rip = zeros(1,6);

        for k = 1:6

            e = E(:,k);

            [c,lags] = xcorr( ...
                e-mean(e), ...
                ref-mean(ref),500);

            [~,i] = max(c);

            d(k) = lags(i);

            hp = filtfilt(bh,ah,e);

            if any(on) && mean(e(on)) > eps
                rip(k) = 100*std(hp(on))/mean(e(on));
            else
                rip(k) = NaN;
            end

        end

        R = [R; snr0 snr1 d rip];

        labels{end+1} = [subs{s} ' ' mus{m}];

        fprintf(['%s %-7s SNR %5.1f -> %5.1f dB | ' ...
            'Delay F0-F5: %s ms | Ripple: %s %%\n'], ...
            subs{s},mus{m},snr0,snr1, ...
            mat2str(d),mat2str(round(rip,1)));

        % ====================================
        % FIGURE 1: ORIGINAL SIGNAL GRAPHS
        % ====================================

        figure(10+s);

        subplot(3,2,m);
        plot(t,x);
        title([subs{s} ' ' mus{m} ' raw']);
        ylabel('mV');

        subplot(3,2,m+2);
        plot(t,xf);
        title('Filtered 20-450 Hz');
        ylabel('mV');

        subplot(3,2,m+4);
        plot(t,env);
        title('Reference RMS 100 ms');
        ylabel('mV');
        xlabel('Time (s)');

        % ====================================
        % FIGURE 2: FREQUENCY SPECTRUM
        % ====================================

        [P1,f] = pwelch(x,hamming(1000),500,1000,fs);

        P2 = pwelch(xf,hamming(1000),500,1000,fs);

        figure(2);

        subplot(4,2,2*(s-1)+m);

        semilogy(f,P1,f,P2);
        grid on;
        xlim([0 500]);

        title([subs{s} ' ' mus{m}]);

        if s == 4
            xlabel('Frequency (Hz)');
        end

        if m == 1
            ylabel('Power (mV^2/Hz)');
        end

        if s == 1 && m == 1
            legend('Raw','Filtered');
        end

        % ====================================
        % FIGURE 3: ALL FILTERS COMPARISON
        % ====================================

        [~,im] = max(ref);

        idx = max(1,im-1500):min(numel(ref),im+1500);

        figure(3);

        subplot(4,2,2*(s-1)+m);

        plot(t(idx),E(idx,1),'Color',[.85 .85 .85]);
        hold on;

        plot(t(idx),E(idx,2:6),'LineWidth',1.1);

        plot(t(idx),ref(idx),'k','LineWidth',1.8);

        hold off;

        title([subs{s} ' ' mus{m}]);
        ylabel('mV');

        if s == 4
            xlabel('Time (s)');
        end

        if s == 1 && m == 1
            legend([names,{'Reference'}],'FontSize',7);
        end

        % ====================================
        % FIGURE 6: BEFORE vs AFTER KALMAN
        % ====================================

        % One separate figure for each subject
        % Two subplots: biceps and triceps

        figure(20+s);

        subplot(2,1,m);

        plot(t(idx),F2(idx),'Color',[0.65 0.65 0.65], ...
             'LineWidth',1.1);

        hold on;

        plot(t(idx),F5(idx),'b','LineWidth',1.5);

        hold off;

        grid on;

        title([subs{s} ' ' mus{m} ...
               ' - Before vs After Kalman']);

        ylabel('EMG RMS (mV)');

        legend('Before Kalman (F2 RMS)', ...
               'After Kalman (F5)','Location','best');

        if m == 2
            xlabel('Time (s)');
        end

    end

    % Save original signal figure
    exportgraphics(figure(10+s), ...
        fullfile(figDir,['Fig1_signals_' subs{s} '.png']), ...
        'Resolution',200);

    % Save individual Kalman comparison
    exportgraphics(figure(20+s), ...
        fullfile(figDir,['Fig6_Kalman_' subs{s} '.png']), ...
        'Resolution',200);

end

% ==========================================
% SAVE FIGURE 2: SPECTRA
% ==========================================

exportgraphics(figure(2), ...
    fullfile(figDir,'Fig2_spectra_all.png'), ...
    'Resolution',200);

% ==========================================
% SAVE FIGURE 3: FILTER COMPARISON
% ==========================================

exportgraphics(figure(3), ...
    fullfile(figDir,'Fig3_filters_on_burst_all.png'), ...
    'Resolution',200);

% ==========================================
% FIGURE 4: SNR
% ==========================================

figure(4);

bar(categorical(labels,labels),R(:,1:2));

grid on;

ylabel('SNR (dB)');
legend('Raw','Filtered 20-450 Hz', ...
       'Location','northwest');

title('Activity-to-Rest Power Ratio');

exportgraphics(figure(4), ...
    fullfile(figDir,'Fig4_SNR_all.png'), ...
    'Resolution',200);

% ==========================================
% FIGURE 5: DELAY VS RIPPLE
% ==========================================

md = mean(R(:,3:8),'omitnan');
mr = mean(R(:,9:14),'omitnan');

figure(5);

scatter(md,mr,90,'filled');

grid on;
hold on;

text(md+2,mr,names,'FontSize',10);

hold off;

xlabel('Delay (ms)');
ylabel('Ripple (%)');

title('Envelope Smoothness vs Delay');

exportgraphics(figure(5), ...
    fullfile(figDir,'Fig5_delay_vs_ripple.png'), ...
    'Resolution',200);

% ==========================================
% FINAL SUMMARY
% ==========================================

fprintf('\nMEAN of all %d recordings\n',size(R,1));

fprintf('SNR %.1f -> %.1f dB\n', ...
        mean(R(:,1)),mean(R(:,2)));

fprintf('Delay F0-F5 = %s ms\n', ...
        mat2str(round(md)));

fprintf('Ripple F0-F5 = %s %%\n', ...
        mat2str(round(mr,1)));

fprintf('\nKalman parameters:\n');
fprintf('Q factor = %g\n',Qfactor);
fprintf('R factor = %g\n',Rfactor);

fprintf('\nFigures saved in %s\n',figDir);
