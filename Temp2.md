# Calibration and Model Choices


```matlab
function [irs, npv, settleType, irsOpt] = buildCalibrationIRS(obj, args, calibGrid, step, pricers)
    % G2++ Calibration Helper: Builds the IRS object used at one calibration
    % step, using the same production conventions as HW1F and Cheyette.
    % The function also computes the deterministic IRS result through
    % PricingDCF, because the fair swap rate is needed for strike and
    % payer/receiver direction.
    %
    % INPUTS:
    % obj       | GaussianTwoFactors | scalar : G2++ model
    % args      | struct             | scalar : calibration arguments
    % calibGrid | struct             | scalar : calibration dates and option data
    % step      | double             | scalar : current calibration bucket index
    % pricers   | struct             | scalar : pricing engines
    %
    % OUTPUTS:
    % irs       | IRSVanille         | scalar : IRS used for calibration
    % npv       | struct             | scalar : IRS pricing measures
    % settleType| char               | vector : swaption settlement type
    % irsOpt    | IRSVanille/[]      | scalar : option IRS when available

    option = calibGrid.option;
    settleType = calibGrid.settleType;
    irsOpt = [];

    isLongTemplate = numel(args.irs) == 1 && ...
        yearfrac(args.irs.startDate, args.irs.maturityDate, 0) > 90;

    if isLongTemplate

        if isempty(option)

            irs = copy(args.irs);
            irs.setNotional(args.notional);
            irs.setDates(calibGrid.swapStartDates(step), calibGrid.swapsMaturities(step));

        elseif isa(option, 'Swaption')

            irs = copy(args.irs);
            irs.setNotional(args.notional);

            if isprop(option, 'irs') && ~isempty(option.irs)
                irs.setDates(option.irs.startDate, option.irs.maturityDate);
                irsOpt = copy(option.irs);
            else
                irs.setDates(calibGrid.swapStartDates(step), calibGrid.swapsMaturities(step));
            end

        elseif isa(option, 'BermudanOptionable')

            idxOpt = step + calibGrid.pastNbExerciseDates;

            if isprop(option, 'irsForHWCalibration') && ~isempty(option.irsForHWCalibration)
                irs = copy(option.irsForHWCalibration(idxOpt));
            else
                irs = copy(args.irs);
                irs.setDates(calibGrid.swapStartDates(step), calibGrid.swapsMaturities(step));
            end

            irs.setNotional(args.notional);

            if isprop(option, 'irs') && numel(option.irs) >= idxOpt
                irsOpt = copy(option.irs(idxOpt));
            end

        else
            error("Unsupported option type in buildCalibrationIRS.");
        end

    elseif numel(args.irs) ~= calibGrid.nbTimeSteps

        error("Wrong number of IRS objects for G2++ calibration.");

    else

        irs = copy(args.irs(step));
        irs.setNotional(args.notional);

    end

    [npv, ~, ~] = pricers.swap.IRSVanille(irs, ...
        'measures', true, ...
        'details', true);
end
```

```matlab
function K = computeCalibrationStrike(obj, args, calibGrid, step, npv)
    % G2++ Calibration Helper: Computes the calibration strike for one
    % swaption. First version deliberately excludes strike adjustment.
    %
    % INPUTS:
    % obj       | GaussianTwoFactors | scalar : G2++ model
    % args      | struct             | scalar : calibration arguments
    % calibGrid | struct             | scalar : calibration dates and option data
    % step      | double             | scalar : current calibration bucket index
    % npv       | struct             | scalar : IRS pricing measures
    %
    % OUTPUT:
    % K         | double             | scalar : calibration strike

    %#ok<INUSD>

    option = calibGrid.option;

    if isempty(args.strike)

        if isempty(option)

            K = npv.FairSwapRate;

        else

            if ~isprop(option, 'strike') || isempty(option.strike)
                K = npv.FairSwapRate;

            elseif isscalar(option.strike)
                K = option.strike;

            else
                idxOpt = step + calibGrid.pastNbExerciseDates;
                K = option.strike(idxOpt);
            end

        end

    elseif isscalar(args.strike)

        K = args.strike;

    else

        K = args.strike(step);

    end

    if ~isfinite(K)
        error("Calibration strike is not finite.");
    end
end
```

```matlab
function direction = inferSwaptionDirection(K, fairSwapRate)
    % G2++ Calibration Helper: Infers payer/receiver convention from strike
    % versus fair swap rate, following the HW1F and Cheyette calibration style.
    %
    % INPUTS:
    % K            | double | scalar : calibration strike
    % fairSwapRate | double | scalar : fair swap rate
    %
    % OUTPUT:
    % direction    | char   | vector : 'PAY' or 'REC'

    if K > fairSwapRate
        direction = 'PAY';
    else
        direction = 'REC';
    end
end
```

```matlab
function swaptionPack = buildCalibrationSwaption(obj, args, calibGrid, step, irs, K, direction, settleType)
    % G2++ Calibration Helper: Builds the Swaption object and extracts the
    % yearfrac payment dates required by the G2++ Brigo swaption pricer.
    %
    % INPUTS:
    % obj       | GaussianTwoFactors | scalar : G2++ model
    % args      | struct             | scalar : calibration arguments
    % calibGrid | struct             | scalar : calibration dates and grid
    % step      | double             | scalar : current calibration bucket index
    % irs       | IRSVanille         | scalar : underlying IRS
    % K         | double             | scalar : calibration strike
    % direction | char               | vector : 'PAY' or 'REC'
    % settleType| char               | vector : settlement type
    %
    % OUTPUT:
    % swaptionPack | struct : swaption object and G2++ pricing inputs

    %#ok<INUSD>

    irs.direction = direction;

    swaption = Swaption(irs, ...
        'exerciseDate', calibGrid.swaptionsExpiries(step), ...
        'strike', K, ...
        'settleType', settleType);

    [~, fixedLegPDates, ~] = irs.fixedLeg01(calibGrid.swapStartDates(step));

    fixedLegPDates = fixedLegPDates(:).';
    nDates = numel(fixedLegPDates);

    if nDates < 2
        error("Fixed leg payment date vector must contain at least start date and one payment date.");
    end

    pricingDateNum = getPricingDateNum(obj, obj.pricingDate);

    % G2++ Brigo pricer expects payment dates T1,...,Tn, excluding swap start.
    paymentDates = yearfracExtend(pricingDateNum, fixedLegPDates(2:nDates), obj.basis);
    paymentDates = paymentDates(:).';

    tenorsDt = yearfracExtend(fixedLegPDates(1:nDates-1), fixedLegPDates(2:nDates), ...
        swaption.irs.fixedLeg.interestDCC);
    tenorsDt = tenorsDt(:).';

    expiry = calibGrid.timeStepsStartDateFrac(step);

    swaptionPack = struct();
    swaptionPack.swaption = swaption;
    swaptionPack.irs = irs;
    swaptionPack.strike = K;
    swaptionPack.direction = direction;
    swaptionPack.settleType = settleType;
    swaptionPack.expiry = expiry;
    swaptionPack.paymentDates = paymentDates;
    swaptionPack.tenorsDt = tenorsDt;
    swaptionPack.fixedLegPDates = fixedLegPDates;
end
```

```matlab
function [price, bachelierVol] = computeMarketPrice(obj, args, step, swaptionPack, pricers, npv)
    % G2++ Calibration Helper: Computes the market target price using the
    % production Bachelier pricer. This keeps the G2++ calibration aligned
    % with the current library quote convention.
    %
    % INPUTS:
    % obj          | GaussianTwoFactors | scalar : G2++ model
    % args         | struct             | scalar : calibration arguments
    % step         | double             | scalar : current calibration bucket index
    % swaptionPack | struct             | scalar : swaption package
    % pricers      | struct             | scalar : pricing engines
    % npv          | struct             | scalar : IRS pricing measures
    %
    % OUTPUTS:
    % price        | double             | scalar : Bachelier market price
    % bachelierVol | double             | scalar : Bachelier implied volatility used

    %#ok<INUSD>

    if isempty(args.vol) || numel(args.vol) == 1

        [price, ~, ~, bachelierVol] = pricers.bachelier.Swaption( ...
            swaptionPack.swaption, ...
            'measures', false, ...
            'vol', args.vol, ...
            'irsRes', npv);

    else

        [price, ~, ~, bachelierVol] = pricers.bachelier.Swaption( ...
            swaptionPack.swaption, ...
            'measures', false, ...
            'vol', args.vol(step), ...
            'irsRes', npv);

    end

    if ~isfinite(price)
        error("Bachelier market price is not finite.");
    end
end
```

```matlab
function [sigmaCalibTrial, etaCalibTrial] = assembleTrialCalibrationVols(obj, thetaLocal, step, calibState)
    % G2++ Calibration Helper: Builds trial calibration volatility vectors
    % from one local parameter block. Future buckets are filled forward with
    % the current trial value. This avoids unknown future buckets while still
    % allowing the G2++ core pricer to see a complete volatility structure.
    %
    % INPUTS:
    % obj        | GaussianTwoFactors | scalar : G2++ model
    % thetaLocal | double             | vector : [sigma_k ; eta_k]
    % step       | double             | scalar : current calibration bucket index
    % calibState | struct             | scalar : current calibration state
    %
    % OUTPUTS:
    % sigmaCalibTrial | double | vector : trial calibration sigma buckets
    % etaCalibTrial   | double | vector : trial calibration eta buckets

    %#ok<INUSD>

    thetaLocal = thetaLocal(:);

    if numel(thetaLocal) ~= 2
        error("thetaLocal must be [sigma_k ; eta_k] in the current helper version.");
    end

    sigmaK = thetaLocal(1);
    etaK   = thetaLocal(2);

    if sigmaK < 0 || etaK < 0 || ~isfinite(sigmaK) || ~isfinite(etaK)
        sigmaCalibTrial = NaN(size(calibState.sigmaCalib));
        etaCalibTrial   = NaN(size(calibState.etaCalib));
        return;
    end

    sigmaCalibTrial = calibState.sigmaCalib;
    etaCalibTrial   = calibState.etaCalib;

    sigmaCalibTrial(step:end) = sigmaK;
    etaCalibTrial(step:end)   = etaK;
end
```

```matlab
function previousState = setModelVolatilityState(obj, sigmaModel, etaModel)
    % G2++ Calibration Helper: Applies one trial volatility state to the
    % model while keeping the previous state available for restoration.
    %
    % INPUTS:
    % obj        | GaussianTwoFactors | scalar : G2++ model
    % sigmaModel | double             | vector : internal model sigma buckets
    % etaModel   | double             | vector : internal model eta buckets
    %
    % OUTPUT:
    % previousState | struct : previous model volatility state

    previousState = struct();
    previousState.volatility_sigma = obj.volatility_sigma;
    previousState.volatility_eta = obj.volatility_eta;

    obj.volatility_sigma = sigmaModel(:).';
    obj.volatility_eta   = etaModel(:).';
end
```

```matlab
function restoreModelVolatilityState(obj, previousState)
    % G2++ Calibration Helper: Restores the model volatility state after a
    % trial pricing evaluation.

    obj.volatility_sigma = previousState.volatility_sigma;
    obj.volatility_eta   = previousState.volatility_eta;
end
```

```matlab
function [residual, modelPrice, trialState] = g2ppBucketPriceResidual(obj, thetaLocal, args, step, calibGrid, calibState, swaptionPack, marketPrice)
    % G2++ Calibration Helper: Evaluates the one-bucket G2++ calibration
    % residual for a trial local parameter block. This helper does not decide
    % the final calibration policy; it only maps trial volatilities into the
    % current G2++ model, calls the core swaption pricer, and returns the
    % model-minus-market price error.
    %
    % INPUTS:
    % obj          | GaussianTwoFactors | scalar : G2++ model
    % thetaLocal   | double             | vector : [sigma_k ; eta_k]
    % args         | struct             | scalar : calibration arguments
    % step         | double             | scalar : current calibration bucket index
    % calibGrid    | struct             | scalar : calibration grid
    % calibState   | struct             | scalar : calibration state
    % swaptionPack | struct             | scalar : swaption and payment dates
    % marketPrice  | double             | scalar : Bachelier market price
    %
    % OUTPUTS:
    % residual     | double             | scalar : model price minus market price
    % modelPrice   | double             | scalar : G2++ model price
    % trialState   | struct             | scalar : trial vectors used

    %#ok<INUSD>

    [sigmaCalibTrial, etaCalibTrial] = assembleTrialCalibrationVols( ...
        obj, thetaLocal, step, calibState);

    if any(~isfinite(sigmaCalibTrial)) || any(~isfinite(etaCalibTrial))
        residual = 1e10;
        modelPrice = NaN;
        trialState = struct();
        return;
    end

    [sigmaModel, etaModel] = adjustCalibrationVolBuckets( ...
        obj, sigmaCalibTrial, etaCalibTrial);

    previousState = setModelVolatilityState(obj, sigmaModel, etaModel);

    try

        modelPrice = obj.getSwaptionPriceG2PP( ...
            swaptionPack.strike, ...
            swaptionPack.expiry, ...
            swaptionPack.paymentDates, ...
            'direction', swaptionPack.direction, ...
            'notional', args.notional);

        if ~isfinite(modelPrice)
            residual = 1e10;
        else
            residual = modelPrice - marketPrice;
        end

    catch

        modelPrice = NaN;
        residual = 1e10;

    end

    restoreModelVolatilityState(obj, previousState);

    trialState = struct();
    trialState.sigmaCalib = sigmaCalibTrial;
    trialState.etaCalib = etaCalibTrial;
    trialState.sigmaModel = sigmaModel;
    trialState.etaModel = etaModel;
end
```

Minimal loop skeleton these helpers are designed for:

```matlab
for step = calibState.startStep:calibGrid.nbTimeSteps

    [irs, npv, settleType, ~] = buildCalibrationIRS(obj, args, calibGrid, step, pricers);

    K = computeCalibrationStrike(obj, args, calibGrid, step, npv);
    obj.calibrationStrike(step) = K;

    direction = inferSwaptionDirection(K, npv.FairSwapRate);

    swaptionPack = buildCalibrationSwaption(obj, args, calibGrid, step, irs, K, direction, settleType);

    [mktPrice(step), bachelierVol(step)] = computeMarketPrice(obj, args, step, swaptionPack, pricers, npv);

    % Solver policy still to be chosen:
    % residualFun = @(theta) g2ppBucketPriceResidual(...);

end
```

The key design point stays unchanged: the calibration helper only mutates trial $\sigma,\eta$; the actual price still goes through `obj.getSwaptionPriceG2PP`.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
