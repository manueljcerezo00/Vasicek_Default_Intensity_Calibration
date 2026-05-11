# Calibration and Model Choices


> **Providing MATLAB skeleton code**
>
> The user is asking for the first part of a MATLAB file, focusing on a significant, testable section. I'll provide a MATLAB code skeleton with method signatures, arguments, setup, calibration grid, pricers, initialization, and method availability. I'll include an orchestrator with an optional "stopAfterSetup" and local helper functions like `buildCalibrationGridLocal` and `initializeCalibStateLocal`. The answer seems to not need citations unless using file search explicitly.
>
> **Outlining first section of MATLAB code**
>
> I need to focus on writing the first testable section, which includes parsing arguments, extracting the swaption vector or option, setting up the volatility structure, initializing calibration arrays, and building pricers. I'll avoid using the "adjust strike" feature and ensure this piece fits into the `@GaussianTwoFactors/swaptionPiecewiseCalibrationG2PP.m` file. I'll also ensure compatibility with the IRSVanille library and use existing models like `BachelierModel`. For forecast/discount arguments, I'll ensure they're robust but avoid unnecessary dependencies.
>
> **Defining options for calibration setup**
>
> For the first testable section, I'll focus on supporting extraction from a swaptionVolCube vector and a Bermudan-like option. I'll use a setup similar to the HW model with argument definitions like `args.irs`, `args.strike`, `args.option`, etc. The default for `args.stopAfterSetup` will be `true` to avoid unimplemented functionality for now, but I'll change that later. This preserves general calibration flexibility.
>
> Thought for 42s

Object: **first testable section of `swaptionPiecewiseCalibrationG2PP.m`**.

Scope of this section:

$$
\boxed{
\text{setup only: calibration grid + model bucket adapter + pricers + initialized state}
}
$$

No strike adjustment.  
No local solve yet.  
No pricing loop yet.

Paste this as the first version of:

```matlab
@GaussianTwoFactors/swaptionPiecewiseCalibrationG2PP.m
```

```matlab
function [calibratedSigma, calibratedEta, bachelierVol, mktPrice, modelPrice, arraySwaption, diagnostics] = ...
    swaptionPiecewiseCalibrationG2PP(obj, args)
%SWAPTIONPIECEWISECALIBRATIONG2PP
% First G2++ swaption calibration orchestrator.
%
% Section 1 objective:
%   - build the calibration time grid from swaption cube or option;
%   - set G2++ volatilityTimeStructure consistently;
%   - reconcile calibration buckets with current G2++ internal bucket logic;
%   - initialize sigma/eta vectors;
%   - initialize production pricers;
%   - return diagnostics before the pricing/calibration loop.
%
% Current convention:
%   Calibration grid has n dates.
%   Current G2++ core interprets n dates as n+1 model buckets.
%   Therefore calibration vectors are expanded by a terminal policy:
%
%       sigmaModel = [sigmaCalib; sigmaCalib(end)]
%       etaModel   = [etaCalib; etaCalib(end)]
%
%   under terminalPolicy = 'flatLast'.

arguments
    obj (1,1) GaussianTwoFactors

    % Instrument source
    args.irs (:,1) IRSVanille = obj.irsDefault
    args.option = []
    args.swaptionVector = []

    % Swaption cube extraction
    args.vectorType (1,:) char {mustBeMember(args.vectorType, {'CoTerm','CoIni'})} = 'CoTerm'
    args.extractValue (1,1) cell = obj.defaultCalibrationMaturity
    args.toleranceCoTerm (1,1) logical = false

    % Basic calibration controls
    args.strike double = []
    args.vol = []
    args.settleType (1,:) char = 'CashSettled'
    args.notional (1,1) double = 1e6

    % Restart / Greeks acceleration
    args.alreadyComputedSigma (:,1) double = []
    args.alreadyComputedEta (:,1) double = []

    % First version: no strike adjustment
    args.adjustStrike (1,1) logical = false

    % G2++ local parametrization controls, used later
    args.localMode (1,:) char {mustBeMember(args.localMode, {'joint','ratio'})} = 'joint'
    args.terminalPolicy (1,:) char {mustBeMember(args.terminalPolicy, {'flatLast','zero'})} = 'flatLast'

    % Initial values
    args.initialSigma (1,1) double {mustBePositive} = 0.01
    args.initialEta (1,1) double {mustBePositive} = 0.01

    % Solver controls, used later
    args.display (1,:) char = 'none'
    args.maxIterations (1,1) double = 200

    % Development switch
    args.stopAfterSetup (1,1) logical = true
end

%--------------------------------------------------------
% 0) Guardrails for this first version
%--------------------------------------------------------
if args.adjustStrike
    error("adjustStrike is intentionally disabled in this first G2++ calibration version.");
end

%--------------------------------------------------------
% 1) Build calibration grid
%--------------------------------------------------------
calibGrid = buildCalibrationGridLocal(obj, args);

obj.volatilityTimeStructure = calibGrid.volatilityTimeStructure;
obj.volatilityTimeStructureDt = calibGrid.volatilityTimeStructureDt;

nbTimeSteps = calibGrid.nbTimeSteps;

%--------------------------------------------------------
% 2) Initialize calibration state and adapt buckets to G2++ core
%--------------------------------------------------------
calibState = initializeCalibrationStateLocal(obj, args, calibGrid);

obj.volatility_sigma = calibState.sigmaModel;
obj.volatility_eta   = calibState.etaModel;

obj.calibrationStrike = NaN(nbTimeSteps, 1);

%--------------------------------------------------------
% 3) Initialize pricers
%--------------------------------------------------------
pricers = buildPricersLocal(obj);

%--------------------------------------------------------
% 4) Initialize outputs
%--------------------------------------------------------
calibratedSigma = calibState.sigmaCalib;
calibratedEta   = calibState.etaCalib;

bachelierVol = NaN(nbTimeSteps, 1);
mktPrice     = NaN(nbTimeSteps, 1);
modelPrice   = NaN(nbTimeSteps, 1);

arraySwaption = [];

diagnostics = struct();
diagnostics.status = "setup_completed";
diagnostics.calibGrid = calibGrid;
diagnostics.calibState = calibState;
diagnostics.pricersReady = true;
diagnostics.nbTimeSteps = nbTimeSteps;
diagnostics.startStep = calibState.startStep;
diagnostics.internalModelBucketCount = numel(obj.volatility_sigma);
diagnostics.calibrationBucketCount = numel(calibratedSigma);

%--------------------------------------------------------
% 5) Development stop
%--------------------------------------------------------
if args.stopAfterSetup
    return;
end

error("Calibration loop not implemented yet. Section 1 setup passed.");

end
```

Local helpers below the same file:

```matlab
%========================================================
% LOCAL HELPER 1: build calibration grid
%========================================================
function calibGrid = buildCalibrationGridLocal(obj, args)

pricingDateNum = pricingDateNumLocal(obj.pricingDate);

if isempty(args.option)

    if isempty(args.swaptionVector)
        swaptionVector = obj.swaptionVolCube.extractVector( ...
            'vectorType', args.vectorType, ...
            'value', args.extractValue, ...
            'toleranceCoTerm', args.toleranceCoTerm);
    else
        swaptionVector = args.swaptionVector;
    end

    switch args.vectorType
        case 'CoTerm'
            volatilityTimeStructure = swaptionVector.expiriesDates(:).';
            swaptionsExpiries = swaptionVector.expiriesDates(:).';
            swapStartDates = swaptionVector.startSwapDates(:).';
            swapsMaturities = swaptionVector.matSwapDates(:).';

        case 'CoIni'
            error("CoIni G2++ calibration is not implemented in the first version.");
    end

    pastNbExerciseDates = 0;
    option = [];
    settleType = args.settleType;

else

    option = args.option;

    if isa(option, 'Swaption')
        exerciseDates = option.exerciseDate;
        deliveryDates = option.exerciseSettleDate;

    elseif isa(option, 'BermudanOptionable')
        exerciseDates = option.exerciseDates;
        deliveryDates = option.deliveryDates;

    else
        error("args.option must be a Swaption or BermudanOptionable object.");
    end

    exerciseDates = exerciseDates(:).';
    deliveryDates = deliveryDates(:).';

    futureMask = exerciseDates > pricingDateNum;

    volatilityTimeStructure = exerciseDates(futureMask);
    swaptionsExpiries = exerciseDates(futureMask);
    swapStartDates = deliveryDates(futureMask);

    swapsMaturities = option.maturityDate .* ones(1, numel(volatilityTimeStructure));

    pastNbExerciseDates = sum(~futureMask);
    settleType = option.settleType;
    swaptionVector = [];
end

if isempty(volatilityTimeStructure)
    error("No future swaption expiry/exercise date found for calibration.");
end

volatilityTimeStructure = volatilityTimeStructure(:).';
volatilityTimeStructureDt = yearfracExtend(pricingDateNum, volatilityTimeStructure, obj.basis);
volatilityTimeStructureDt = volatilityTimeStructureDt(:).';

timeStepsStartDateFrac = yearfracExtend(pricingDateNum, swapStartDates, obj.basis);
timeStepsStartDateFrac = timeStepsStartDateFrac(:).';

calibGrid = struct();
calibGrid.option = option;
calibGrid.swaptionVector = swaptionVector;
calibGrid.settleType = settleType;

calibGrid.volatilityTimeStructure = volatilityTimeStructure;
calibGrid.volatilityTimeStructureDt = volatilityTimeStructureDt;

calibGrid.swaptionsExpiries = swaptionsExpiries(:).';
calibGrid.swapStartDates = swapStartDates(:).';
calibGrid.swapsMaturities = swapsMaturities(:).';

calibGrid.timeStepsStartDateFrac = timeStepsStartDateFrac;
calibGrid.pastNbExerciseDates = pastNbExerciseDates;
calibGrid.nbTimeSteps = numel(volatilityTimeStructure);

end
```

```matlab
%========================================================
% LOCAL HELPER 2: initialize calibration vectors
%========================================================
function calibState = initializeCalibrationStateLocal(obj, args, calibGrid)

nbTimeSteps = calibGrid.nbTimeSteps;

nSigmaPrev = numel(args.alreadyComputedSigma);
nEtaPrev   = numel(args.alreadyComputedEta);

if nSigmaPrev ~= nEtaPrev
    error("alreadyComputedSigma and alreadyComputedEta must have the same length.");
end

if nSigmaPrev > nbTimeSteps
    error("alreadyComputedSigma/alreadyComputedEta cannot exceed the number of calibration buckets.");
end

startStep = nSigmaPrev + 1;

sigmaCalib = args.initialSigma .* ones(nbTimeSteps, 1);
etaCalib   = args.initialEta   .* ones(nbTimeSteps, 1);

if nSigmaPrev > 0
    sigmaCalib(1:nSigmaPrev) = args.alreadyComputedSigma(:);
    etaCalib(1:nEtaPrev)     = args.alreadyComputedEta(:);
end

[sigmaModel, etaModel] = expandCalibrationBucketsLocal( ...
    sigmaCalib, etaCalib, args.terminalPolicy);

validateExpandedBucketsLocal(obj, calibGrid, sigmaModel, etaModel);

calibState = struct();
calibState.startStep = startStep;

calibState.sigmaCalib = sigmaCalib;
calibState.etaCalib = etaCalib;

calibState.sigmaModel = sigmaModel;
calibState.etaModel = etaModel;

calibState.terminalPolicy = args.terminalPolicy;
calibState.localMode = args.localMode;

end
```

```matlab
%========================================================
% LOCAL HELPER 3: expand calibration buckets to current G2++ core
%========================================================
function [sigmaModel, etaModel] = expandCalibrationBucketsLocal(sigmaCalib, etaCalib, terminalPolicy)

sigmaCalib = sigmaCalib(:);
etaCalib   = etaCalib(:);

if numel(sigmaCalib) ~= numel(etaCalib)
    error("sigmaCalib and etaCalib must have the same length.");
end

switch terminalPolicy
    case 'flatLast'
        sigmaModel = [sigmaCalib; sigmaCalib(end)];
        etaModel   = [etaCalib; etaCalib(end)];

    case 'zero'
        sigmaModel = [sigmaCalib; 0.0];
        etaModel   = [etaCalib; 0.0];

    otherwise
        error("Unsupported terminalPolicy.");
end

end
```

```matlab
%========================================================
% LOCAL HELPER 4: validate current G2++ bucket convention
%========================================================
function validateExpandedBucketsLocal(obj, calibGrid, sigmaModel, etaModel)

nKnots = numel(calibGrid.volatilityTimeStructure);
expectedModelBuckets = nKnots + 1;

if numel(sigmaModel) ~= expectedModelBuckets
    error("sigmaModel must have length numel(volatilityTimeStructure)+1 under current G2++ bucket logic.");
end

if numel(etaModel) ~= expectedModelBuckets
    error("etaModel must have length numel(volatilityTimeStructure)+1 under current G2++ bucket logic.");
end

if any(~isfinite(sigmaModel)) || any(sigmaModel < 0)
    error("sigmaModel must be finite and nonnegative.");
end

if any(~isfinite(etaModel)) || any(etaModel < 0)
    error("etaModel must be finite and nonnegative.");
end

% This also checks that the object has a usable time grid.
if isempty(obj.volatilityTimeStructure) && isempty(obj.volatilityTimeStructureDt)
    error("G2++ volatility time structure is empty.");
end

end
```

```matlab
%========================================================
% LOCAL HELPER 5: initialize production pricers
%========================================================
function pricers = buildPricersLocal(obj)

pricers = struct();

pricers.swap = PricingDCF(obj.DCF);

bachModel = BachelierModel(obj.swaptionVolCube, obj.DCF.discount, obj.DCF.forecast);
pricers.bachelier = PricingBachelierModel(bachModel);

end
```

```matlab
%========================================================
% LOCAL HELPER 6: robust pricing date conversion
%========================================================
function pricingDateNum = pricingDateNumLocal(pricingDate)

if isnumeric(pricingDate)
    pricingDateNum = pricingDate;

elseif isdatetime(pricingDate)
    pricingDateNum = datenum(pricingDate);

elseif isstring(pricingDate) || ischar(pricingDate)
    pricingDateNum = datenum(char(pricingDate), 'dd/mm/yyyy');

else
    error("Unsupported pricingDate format.");
end

end
```

Minimal test after pasting:

```matlab
[calSigma, calEta, bachVol, mktPrice, modelPrice, arraySwaption, diagnostics] = ...
    G2PP.swaptionPiecewiseCalibrationG2PP( ...
        'stopAfterSetup', true);

assert(diagnostics.nbTimeSteps > 0);
assert(numel(G2PP.volatility_sigma) == diagnostics.nbTimeSteps + 1);
assert(numel(G2PP.volatility_eta) == diagnostics.nbTimeSteps + 1);
assert(all(isfinite(G2PP.volatility_sigma)));
assert(all(isfinite(G2PP.volatility_eta)));

disp(diagnostics.status);
```

This first section is useful because it tests the hardest structural point before pricing: **the calibration grid is compatible with the current G2++ internal bucket convention without modifying the core model**.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
