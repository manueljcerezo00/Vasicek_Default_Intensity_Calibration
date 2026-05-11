%% G2++ Brigo Swaption Calibration - Live Script Skeleton
% Purpose:
%   1. Build a GaussianTwoFactors model from OIS curves already available in the workspace.
%   2. Load Brigo Table 4.2 ATM Euro swaption volatilities by hand.
%   3. Calibrate constant-volatility G2++ parameters (a,b,sigma,eta,rho).
%   4. Compare calibrated implied volatilities with Brigo Tables 4.3 and 4.4.
%
% Required in the workspace before running this script:
%   pricingDate     : char/string/datenum, e.g. '13/02/2001'
%   discountCurve   : ZCCurve object, preferably OIS/discount curve
%   forecastCurve   : ZCCurve object, or same object if single-curve setup
%
% Required methods in GaussianTwoFactors:
%   priceSwaptionG2PP
%   getSwaptionInstrument
%   getSwaptionBondBasket
%   getInitialMarketZC
%   varI, muTForward, varTForward, covTForward

clearvars -except pricingDate discountCurve forecastCurve oisCurve oisDiscountCurve oisForecastCurve
clc;
clear classes;

%% 0. Curve object convention
% Adapt these names to the production library workspace.
% If you have a single OIS curve available, use it for both discount and forecast
% for the first Brigo benchmark.

if ~exist('discountCurve','var')
    if exist('oisDiscountCurve','var')
        discountCurve = oisDiscountCurve;
    elseif exist('oisCurve','var')
        discountCurve = oisCurve;
    else
        error('Define discountCurve, oisDiscountCurve, or oisCurve before running.');
    end
end

if ~exist('forecastCurve','var')
    if exist('oisForecastCurve','var')
        forecastCurve = oisForecastCurve;
    else
        forecastCurve = discountCurve; % first benchmark: single-curve approximation
    end
end

if ~exist('pricingDate','var')
    pricingDate = '13/02/2001'; % Brigo example date
end

%% 1. Brigo Table 4.2 market ATM swaption volatilities
% Rows: option maturities 1y,2y,3y,4y,5y,7y,10y.
% Columns: underlying swap tenors 1y,...,10y.
% Quote type: Black/lognormal volatility, as presented in Brigo's swaption-volatility table.

brigoExpiries = [1 2 3 4 5 7 10];
brigoTenors   = 1:10;

brigoMarketVol = [
    0.1640 0.1550 0.1430 0.1310 0.1240 0.1190 0.1160 0.1120 0.1100 0.1070;
    0.1600 0.1500 0.1390 0.1290 0.1220 0.1190 0.1160 0.1130 0.1100 0.1080;
    0.1570 0.1450 0.1340 0.1240 0.1190 0.1150 0.1130 0.1100 0.1080 0.1060;
    0.1480 0.1360 0.1260 0.1190 0.1140 0.1120 0.1090 0.1070 0.1050 0.1030;
    0.1400 0.1280 0.1210 0.1140 0.1100 0.1070 0.1050 0.1030 0.1020 0.1000;
    0.1300 0.1190 0.1130 0.1050 0.1010 0.0990 0.0970 0.0960 0.0950 0.0930;
    0.1160 0.1070 0.1000 0.0930 0.0900 0.0890 0.0870 0.0860 0.0850 0.0840
];

% Brigo reported constant-vol G2++ calibration parameters for Table 4.2.
brigoReported.a     = 0.773511777;
brigoReported.b     = 0.082013014;
brigoReported.sigma = 0.022284644;
brigoReported.eta   = 0.010382461;
brigoReported.rho   = -0.701985206;

%% 2. Build the model
% Constant-volatility G2++ can be represented using one effective bucket.
% If your getBucketBounds uses [0,timeSteps] and [timeSteps,Inf], then the
% volatility vectors must have length numel(timeSteps)+1.

basis = 3; % ACT/365' in your current library convention

model = GaussianTwoFactors(pricingDate, ...
    brigoReported.a, ...
    brigoReported.b, ...
    discountCurve, ...
    forecastCurve, ...
    'basis', basis, ...
    'displayLog', false);

% One long time step creates two buckets: [0,100] and [100,Inf].
% Both have the same constant vol value.
model.volatilityTimeStructure = 100;
model.volatilityTimeStructureDt = [];
model.volatility_sigma = brigoReported.sigma .* [1 1];
model.volatility_eta   = brigoReported.eta   .* [1 1];
model.correlation = brigoReported.rho;

%% 3. Method availability test
requiredMethods = [
    "priceSwaptionG2PP"
    "getSwaptionInstrument"
    "getSwaptionBondBasket"
    "getInitialMarketZC"
    "varI"
    "muTForward"
    "varTForward"
    "covTForward"
];

availableMethods = string(methods(model));
missingMethods = setdiff(requiredMethods, availableMethods);
assert(isempty(missingMethods), "Missing methods: " + strjoin(missingMethods, ", "));
disp('Method availability test passed.');

%% 4. Build Brigo calibration set
% Brigo table uses ATM swaptions by expiry/tenor grid.
% For first benchmark, use annual fixed-leg payments.

calibSet = buildBrigoSwaptionCalibrationSetLocal( ...
    brigoExpiries, brigoTenors, brigoMarketVol, ...
    'quoteType', 'black', ...
    'direction', 'PAY', ...
    'notional', 1.0, ...
    'strikeConvention', 'ATM', ...
    'paymentFrequency', 1.0);

fprintf('Calibration instruments: %d\n', numel(calibSet));

%% 5. Sanity check: one Brigo-style swaption price
instr0 = calibSet(1);
K0 = getCalibrationStrikeG2PPLocal(model, instr0);

price0 = model.priceSwaptionG2PP(K0, instr0.expiry, instr0.paymentDates, ...
    'direction', instr0.direction, ...
    'notional', instr0.notional);

[F0, A0] = getSwaptionForwardAndAnnuityG2PPLocal(model, instr0.expiry, instr0.paymentDates);
marketPrice0 = swaptionPriceFromVolLocal(F0, K0, instr0.expiry, A0, instr0.quote, instr0.quoteType, instr0.direction, instr0.notional);

fprintf('First instrument: expiry %.0fy, tenor %.0fy\n', instr0.expiry, instr0.tenor);
fprintf('ATM strike: %.8f\n', K0);
fprintf('Model price:  %.10f\n', price0);
fprintf('Market price: %.10f\n', marketPrice0);
assert(isfinite(price0) && price0 >= 0, 'First model price must be finite and nonnegative.');

%% 6. Price all instruments at Brigo reported parameters
[resReported, diagReported] = g2ppBrigoCalibrationResidualsLocal( ...
    g2ppInverseTransformBrigo5Local( ...
        brigoReported.a, brigoReported.b, brigoReported.sigma, brigoReported.eta, brigoReported.rho), ...
    model, calibSet, ...
    'objectiveType', 'normalizedPrice');

fprintf('RMSE at Brigo reported parameters: %.8e\n', sqrt(mean(resReported.^2)));
fprintf('Max abs residual at Brigo reported parameters: %.8e\n', max(abs(resReported)));

%% 7. Calibrate from an initial guess
% Brigo used global optimization followed by local search. Here we start with
% lsqnonlin; later use multi-start if the local result is unstable.

a0 = 0.50;
b0 = 0.10;
sigma0 = 0.015;
eta0 = 0.010;
rho0 = -0.50;

theta0Raw = g2ppInverseTransformBrigo5Local(a0, b0, sigma0, eta0, rho0);

result = calibrateG2PPToBrigoSurfaceLocal( ...
    model, calibSet, theta0Raw, ...
    'objectiveType', 'normalizedPrice', ...
    'useLsqnonlin', true, ...
    'maxIterations', 200, ...
    'display', 'iter');

disp(result.params);
fprintf('Calibration RMSE: %.8e\n', result.rmse);
fprintf('Calibration max abs residual: %.8e\n', result.maxAbsError);

%% 8. Compute calibrated model implied volatilities
[modelVolMatrix, priceMatrix, marketPriceMatrix] = computeModelVolMatrixLocal(model, calibSet, brigoExpiries, brigoTenors);

volErrorPct = 100 .* (modelVolMatrix - brigoMarketVol) ./ brigoMarketVol;

modelVolTable = array2table(modelVolMatrix, ...
    'RowNames', compose('%dy', brigoExpiries), ...
    'VariableNames', compose('%dy', brigoTenors));

errorPctTable = array2table(volErrorPct, ...
    'RowNames', compose('%dy', brigoExpiries), ...
    'VariableNames', compose('%dy', brigoTenors));

disp('Model implied volatility matrix:');
disp(modelVolTable);

disp('Percentage error matrix:');
disp(errorPctTable);

%% 9. Compare with Brigo Table 4.3 and 4.4, if desired
brigoTable43_ModelVol = [
    0.1870 0.1529 0.1395 0.1327 0.1276 0.1231 0.1190 0.1154 0.1120 0.1085;
    0.1603 0.1427 0.1348 0.1295 0.1251 0.1210 0.1174 0.1139 0.1103 0.1068;
    0.1509 0.1376 0.1307 0.1258 0.1216 0.1180 0.1146 0.1109 0.1073 0.1041;
    0.1422 0.1311 0.1252 0.1210 0.1175 0.1142 0.1106 0.1069 0.1037 0.1005;
    0.1335 0.1247 0.1199 0.1165 0.1134 0.1098 0.1062 0.1030 0.0998 0.0968;
    0.1218 0.1161 0.1124 0.1087 0.1050 0.1018 0.0986 0.0954 0.0928 0.0902;
    0.1090 0.1029 0.0997 0.0965 0.0934 0.0909 0.0884 0.0858 0.0833 0.0809
];

brigoVolDiff = modelVolMatrix - brigoTable43_ModelVol;
fprintf('Max abs difference vs Brigo Table 4.3 vols: %.8e\n', max(abs(brigoVolDiff(:))));

%% 10. Robustness checks
% 10.1 residuals must be finite at initial parameters
res0 = g2ppBrigoCalibrationResidualsLocal(theta0Raw, model, calibSet, 'objectiveType', 'normalizedPrice');
assert(all(isfinite(res0)), 'Initial residual vector contains non-finite values.');

% 10.2 calibrated params must be admissible
p = result.params;
assert(p.a > 0 && p.b > 0 && p.sigma > 0 && p.eta > 0 && abs(p.rho) < 1);

% 10.3 payer-receiver parity on one instrument
instrP = calibSet(10);
KP = getCalibrationStrikeG2PPLocal(model, instrP);
pricePAY = model.priceSwaptionG2PP(KP, instrP.expiry, instrP.paymentDates, 'direction','PAY','notional',instrP.notional);
priceREC = model.priceSwaptionG2PP(KP, instrP.expiry, instrP.paymentDates, 'direction','REC','notional',instrP.notional);
basketP = model.getSwaptionBondBasket(instrP.expiry, instrP.paymentDates);
instrObjP = model.getSwaptionInstrument(KP, instrP.expiry, instrP.paymentDates, 'direction','PAY','notional',instrP.notional);
parityRHS = instrP.notional * (basketP.P0T0 - sum(instrObjP.coeffs(:).*basketP.P0Ti(:)));
parityErr = (pricePAY - priceREC) - parityRHS;
assert(abs(parityErr) / max(1, abs(pricePAY)+abs(priceREC)+abs(parityRHS)) < 1e-5, 'Payer-receiver parity failed.');

disp('Robustness checks passed.');

%% Local helper functions
function calibSet = buildBrigoSwaptionCalibrationSetLocal(expiries, tenors, quotes, args)
    arguments
        expiries (1,:) double
        tenors (1,:) double
        quotes (:,:) double
        args.quoteType (1,:) char {mustBeMember(args.quoteType, {'price','black','normal'})} = 'black'
        args.direction (1,:) char {mustBeMember(args.direction, {'PAY','REC'})} = 'PAY'
        args.notional (1,1) double = 1.0
        args.strikeConvention (1,:) char {mustBeMember(args.strikeConvention, {'ATM','fixed'})} = 'ATM'
        args.fixedStrike (1,1) double = NaN
        args.paymentFrequency (1,1) double {mustBePositive} = 1
        args.weight (:,:) double = []
    end

    nExp = numel(expiries);
    nTenor = numel(tenors);

    if size(quotes,1) ~= nExp || size(quotes,2) ~= nTenor
        error('quotes must have size numel(expiries) x numel(tenors).');
    end

    if isempty(args.weight)
        weights = ones(nExp, nTenor);
    else
        weights = args.weight;
    end

    k = 0;
    calibSet = struct([]);

    for i = 1:nExp
        expiry = expiries(i);
        for j = 1:nTenor
            tenor = tenors(j);
            quote = quotes(i,j);

            if ~isfinite(quote)
                continue;
            end

            dt = 1 / args.paymentFrequency;
            nPayments = round(tenor * args.paymentFrequency);
            paymentDates = expiry + (1:nPayments).*dt;

            k = k + 1;
            calibSet(k).expiry = expiry;
            calibSet(k).tenor = tenor;
            calibSet(k).paymentDates = paymentDates;
            calibSet(k).quote = quote;
            calibSet(k).quoteType = args.quoteType;
            calibSet(k).direction = args.direction;
            calibSet(k).notional = args.notional;
            calibSet(k).strikeConvention = args.strikeConvention;
            calibSet(k).fixedStrike = args.fixedStrike;
            calibSet(k).weight = weights(i,j);
            calibSet(k).row = i;
            calibSet(k).col = j;
        end
    end
end

function [F, A, delta, P0T0, P0Ti] = getSwaptionForwardAndAnnuityG2PPLocal(model, expiry, paymentDates)
    paymentDates = paymentDates(:);
    zc = model.getInitialMarketZC([expiry, paymentDates.']);
    zc = zc(:);

    P0T0 = zc(1);
    P0Ti = zc(2:end);
    delta = diff([expiry; paymentDates]);
    A = sum(delta .* P0Ti);
    F = (P0T0 - P0Ti(end)) / A;
end

function K = getCalibrationStrikeG2PPLocal(model, instr)
    [F, ~] = getSwaptionForwardAndAnnuityG2PPLocal(model, instr.expiry, instr.paymentDates);
    if strcmpi(instr.strikeConvention, 'ATM')
        K = F;
    else
        K = instr.fixedStrike;
    end
end

function price = swaptionPriceFromVolLocal(F, K, T, A, vol, volType, direction, notional)
    omega = 1;
    if strcmpi(direction, 'REC')
        omega = -1;
    end

    if T == 0 || vol == 0
        price = notional * A * max(omega*(F-K),0);
        return;
    end

    sqrtT = sqrt(T);

    switch lower(volType)
        case 'normal'
            std = vol * sqrtT;
            d = (F-K) / std;
            price = notional * A * (omega*(F-K)*normcdf(omega*d) + std*normpdf(d));
        case 'black'
            if F <= 0 || K <= 0
                error('Black volatility requires positive F and K.');
            end
            std = vol * sqrtT;
            d1 = (log(F/K) + 0.5*std^2) / std;
            d2 = d1 - std;
            price = notional * A * omega * (F*normcdf(omega*d1) - K*normcdf(omega*d2));
        otherwise
            error('Unsupported vol type.');
    end
end

function impVol = swaptionImpliedVolFromPriceLocal(price, F, K, T, A, volType, direction, notional)
    omega = 1;
    if strcmpi(direction, 'REC')
        omega = -1;
    end

    intrinsic = notional * A * max(omega*(F-K),0);
    if price < intrinsic - 1e-12
        impVol = NaN;
        return;
    end
    if abs(price - intrinsic) < 1e-14
        impVol = 0;
        return;
    end

    f = @(v) swaptionPriceFromVolLocal(F,K,T,A,v,volType,direction,notional) - price;

    if strcmpi(volType,'black') && (F <= 0 || K <= 0)
        impVol = NaN;
        return;
    end

    lo = 1e-10;
    hi = 5.0;
    while f(hi) < 0
        hi = 2*hi;
        if hi > 100
            impVol = NaN;
            return;
        end
    end
    impVol = fzero(f, [lo, hi]);
end

function marketPrice = swaptionMarketPriceFromQuoteLocal(model, instr)
    K = getCalibrationStrikeG2PPLocal(model, instr);
    [F, A] = getSwaptionForwardAndAnnuityG2PPLocal(model, instr.expiry, instr.paymentDates);

    switch lower(instr.quoteType)
        case 'price'
            marketPrice = instr.notional * instr.quote;
        case {'black','normal'}
            marketPrice = swaptionPriceFromVolLocal(F,K,instr.expiry,A,instr.quote,instr.quoteType,instr.direction,instr.notional);
        otherwise
            error('Unsupported quote type.');
    end
end

function thetaRaw = g2ppInverseTransformBrigo5Local(a,b,sigma,eta,rho)
    if abs(rho) >= 1
        error('rho must satisfy abs(rho)<1.');
    end
    thetaRaw = [log(a); log(b); log(sigma); log(eta); atanh(rho)];
end

function params = g2ppApplyTransformBrigo5Local(model, thetaRaw)
    params.a = exp(thetaRaw(1));
    params.b = exp(thetaRaw(2));
    params.sigma = exp(thetaRaw(3));
    params.eta = exp(thetaRaw(4));
    params.rho = tanh(thetaRaw(5));

    model.meanReversion_x = params.a;
    model.meanReversion_y = params.b;
    model.correlation = params.rho;

    nSigma = numel(model.volatility_sigma);
    nEta = numel(model.volatility_eta);
    if nSigma == 0 || nEta == 0
        error('Volatility vectors must be initialized before calibration.');
    end
    model.volatility_sigma = params.sigma .* ones(1,nSigma);
    model.volatility_eta = params.eta .* ones(1,nEta);
end

function [residuals, diagnostics] = g2ppBrigoCalibrationResidualsLocal(thetaRaw, model, calibSet, args)
    arguments
        thetaRaw (5,1) double
        model (1,1) GaussianTwoFactors
        calibSet (1,:) struct
        args.objectiveType (1,:) char {mustBeMember(args.objectiveType, {'price','normalizedPrice'})} = 'normalizedPrice'
        args.badValuePenalty (1,1) double = 1e6
    end

    params = g2ppApplyTransformBrigo5Local(model, thetaRaw);
    n = numel(calibSet);
    residuals = zeros(n,1);

    diagnostics.modelPrice = NaN(n,1);
    diagnostics.marketPrice = NaN(n,1);
    diagnostics.strike = NaN(n,1);
    diagnostics.forward = NaN(n,1);
    diagnostics.annuity = NaN(n,1);
    diagnostics.expiry = NaN(n,1);
    diagnostics.tenor = NaN(n,1);

    for k = 1:n
        instr = calibSet(k);
        try
            K = getCalibrationStrikeG2PPLocal(model, instr);
            [F, A] = getSwaptionForwardAndAnnuityG2PPLocal(model, instr.expiry, instr.paymentDates);
            marketPrice = swaptionMarketPriceFromQuoteLocal(model, instr);
            modelPrice = model.priceSwaptionG2PP(K, instr.expiry, instr.paymentDates, ...
                'direction', instr.direction, 'notional', instr.notional);

            diagnostics.modelPrice(k) = modelPrice;
            diagnostics.marketPrice(k) = marketPrice;
            diagnostics.strike(k) = K;
            diagnostics.forward(k) = F;
            diagnostics.annuity(k) = A;
            diagnostics.expiry(k) = instr.expiry;
            diagnostics.tenor(k) = instr.tenor;

            switch args.objectiveType
                case 'price'
                    residuals(k) = sqrt(instr.weight) * (modelPrice - marketPrice);
                case 'normalizedPrice'
                    scale = max(instr.notional * A, 1e-12);
                    residuals(k) = sqrt(instr.weight) * (modelPrice - marketPrice) / scale;
            end

            if ~isfinite(residuals(k))
                residuals(k) = args.badValuePenalty;
            end
        catch
            residuals(k) = args.badValuePenalty;
        end
    end
    diagnostics.params = params;
end

function result = calibrateG2PPToBrigoSurfaceLocal(model, calibSet, theta0Raw, args)
    arguments
        model (1,1) GaussianTwoFactors
        calibSet (1,:) struct
        theta0Raw (5,1) double
        args.objectiveType (1,:) char {mustBeMember(args.objectiveType, {'price','normalizedPrice'})} = 'normalizedPrice'
        args.maxIterations (1,1) double = 200
        args.display (1,:) char = 'iter'
        args.useLsqnonlin (1,1) logical = true
    end

    residualFun = @(theta) g2ppBrigoCalibrationResidualsLocal(theta, model, calibSet, 'objectiveType', args.objectiveType);

    if args.useLsqnonlin
        opts = optimoptions('lsqnonlin', 'Display', args.display, ...
            'MaxIterations', args.maxIterations, ...
            'StepTolerance', 1e-8, 'FunctionTolerance', 1e-8);
        [thetaStar, resnorm, residuals, exitflag, output] = lsqnonlin(residualFun, theta0Raw, [], [], opts);
    else
        objFun = @(theta) sum(residualFun(theta).^2);
        opts = optimset('Display', args.display, 'MaxIter', args.maxIterations);
        [thetaStar, resnorm, exitflag, output] = fminsearch(objFun, theta0Raw, opts);
        residuals = residualFun(thetaStar);
    end

    params = g2ppApplyTransformBrigo5Local(model, thetaStar);
    [finalResiduals, diagnostics] = g2ppBrigoCalibrationResidualsLocal(thetaStar, model, calibSet, 'objectiveType', args.objectiveType);

    result.thetaRaw = thetaStar;
    result.params = params;
    result.resnorm = resnorm;
    result.residuals = residuals;
    result.finalResiduals = finalResiduals;
    result.exitflag = exitflag;
    result.output = output;
    result.diagnostics = diagnostics;
    result.rmse = sqrt(mean(finalResiduals.^2));
    result.maxAbsError = max(abs(finalResiduals));
end

function [volMatrix, priceMatrix, marketPriceMatrix] = computeModelVolMatrixLocal(model, calibSet, expiries, tenors)
    volMatrix = NaN(numel(expiries), numel(tenors));
    priceMatrix = NaN(numel(expiries), numel(tenors));
    marketPriceMatrix = NaN(numel(expiries), numel(tenors));

    for k = 1:numel(calibSet)
        instr = calibSet(k);
        i = instr.row;
        j = instr.col;
        K = getCalibrationStrikeG2PPLocal(model, instr);
        [F,A] = getSwaptionForwardAndAnnuityG2PPLocal(model, instr.expiry, instr.paymentDates);

        modelPrice = model.priceSwaptionG2PP(K, instr.expiry, instr.paymentDates, ...
            'direction', instr.direction, 'notional', instr.notional);
        marketPrice = swaptionMarketPriceFromQuoteLocal(model, instr);
        modelVol = swaptionImpliedVolFromPriceLocal(modelPrice, F, K, instr.expiry, A, instr.quoteType, instr.direction, instr.notional);

        volMatrix(i,j) = modelVol;
        priceMatrix(i,j) = modelPrice;
        marketPriceMatrix(i,j) = marketPrice;
    end
end
