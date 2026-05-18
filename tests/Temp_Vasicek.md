# Calibration Procedure Review

Bonino-Casalini’s calibration has **three steps**.

## Correct reading of the paper

### Step 1 - global constant G1++ calibration

They first calibrate a **constant-parameter G1++** model to the **whole swaption surface**.

The model is:

$$
dx(t)=-a x(t)dt+\sigma dW(t).
$$

They use initial values:

$$
a_0=0.05,\qquad \sigma_0=0.01,
$$

and minimize the sum of squared differences between market and model swaption prices. The purpose is not final pricing; it is to get a robust global mean-reversion parameter for the following step. `ssrn-3598285 (1).pdf`

### Step 2 - two piecewise G1++ slice bootstraps

Then they calibrate **two piecewise-constant G1++ models**.

Not co-terminal. Not diagonal.

They use two **tenor slices** of the swaption matrix:

$$
\tau=\tau_1
$$

the shortest swap tenor, and

$$
\tau=\tau_m
$$

the longest swap tenor. `ssrn-3598285 (1).pdf`

For the first slice, they bootstrap:

$$
\sigma^s_0,\ldots,\sigma^s_{n-1}.
$$

For the second slice, they bootstrap:

$$
\eta^s_0,\ldots,\eta^s_{n-1}.
$$

The incremental logic is:

$$
\sigma^s_0
\quad\text{fits}\quad
s_{11},
$$

$$
\sigma^s_1
\quad\text{fits}\quad
s_{11},s_{21}
\quad
\text{with }\sigma^s_0\text{ fixed},
$$

and so on until all expiries on that slice are fitted. `ssrn-3598285 (1).pdf`

So this is where the two independent volatility shapes come from:

$$
\sigma^s
\quad\text{from the short-tenor slice},
$$

$$
\eta^s
\quad\text{from the long-tenor slice}.
$$

### Step 3 - global scaled G2++ calibration

Then comes the actual XG2++ calibration.

They do **not** calibrate every G2++ bucket freely. They reduce dimension by assuming:

$$
(\sigma_0,\ldots,\sigma_{n-1})
=
\alpha
(\sigma^s_0,\ldots,\sigma^s_{n-1}),
$$

$$
(\eta_0,\ldots,\eta_{n-1})
=
\beta
(\eta^s_0,\ldots,\eta^s_{n-1}).
$$

So the full G2++ volatility structures are scalar multiples of the two G1++ slice structures. Then they globally calibrate:

$$
a,\ b,\ \alpha,\ \beta,\ \rho.
$$

Initial point:

$$
a=b=a_{G1++},\qquad \alpha=1,\qquad \beta=1,\qquad \rho=0.
$$

They minimize squared price errors over the whole swaption surface. `ssrn-3598285 (1).pdf`

## Correct architectural consequence

Our previous loop:

$$
\text{one G2++ bucket}
\rightarrow
\text{one residual}
\rightarrow
\text{one solve}
$$

is **not the Bonino final step**.

It should be moved to a different role.

The Bonino architecture should be:

```text
BoninoCalibrationG2PP
│
├─ Step 1: calibrate constant G1++ globally
│      → aG1
│
├─ Step 2: bootstrap G1++ short-tenor slice
│      → sigmaShape
│
├─ Step 2: bootstrap G1++ long-tenor slice
│      → etaShape
│
└─ Step 3: calibrate scaled G2++ globally
       σ = α sigmaShape
       η = β etaShape
       parameters = [a,b,α,β,ρ]
```


$$
a,b \text{ fixed},
$$

and Step 3 calibrates only:

$$
\alpha,\beta,\rho,
$$

or even only

$$
\alpha,\beta
$$

if $\rho$ is also fixed.

## What this changes in our module

We should not call the current function:

```matlab
swaptionPiecewiseCalibrationG2PP
```

a Bonino implementation yet.

A better naming split is:

```matlab
initializeBoninoShapesG2PP
```

for Step 1 and Step 2, and:

```matlab
calibrateScaledG2PPToSurface
```

for Step 3.

The current per-bucket helper design is still useful, but mainly for **G1++ slice bootstrapping**, not for the final G2++ stage.

## Revised minimal implementation target

Do this next:

1. Extract full ATM swaption matrix:

$$
s_{ij},\quad i=1,\ldots,n,\quad j=1,\ldots,m.
$$

2. Extract two slices:

$$
s_{\cdot,1}
\quad\text{and}\quad
s_{\cdot,m}.
$$

3. Bootstrap one-factor piecewise volatility on each slice:

$$
\sigmaShape = \text{G1++Bootstrap}(s_{\cdot,1}),
$$

$$
\etaShape = \text{G1++Bootstrap}(s_{\cdot,m}).
$$

4. Set G2++ volatility vectors:

$$
\sigma = \alpha\sigmaShape,
\qquad
\eta = \beta\etaShape.
$$

5. Calibrate reduced global parameters to the whole surface.

## Clean correction

The scalar ratio idea

$$
\eta_k=q\sigma_k
$$

Bonino’s dimensional reduction is instead:

$$
\sigma_k=\alpha\sigma^s_k,
\qquad
\eta_k=\beta\eta^s_k.
$$

That is the key proposition we must now encode.

## Response:
5/12/2026, 11:54:54 PM



Bonino’s multiplicative structure is:

$$
\sigma_k=\alpha\,\sigma_k^s,
\qquad
\eta_k=\beta\,\eta_k^s.
$$

This is a **model structure**: the G2++ volatility vector keeps the shape found by the G1++ slice bootstrap, and only rescales its level. Bonino-Casalini explicitly reduce the G2++ volatility vectors this way before the final global calibration. `ssrn-3598285 (1).pdf`

The exponential transform is an **optimizer parametrization**:

$$
\alpha=\exp(u),
\qquad
\beta=\exp(v),
$$

so that the solver can search over unconstrained variables $u,v\in\mathbb R$, while guaranteeing

$$
\alpha>0,\qquad \beta>0.
$$

So the combined implementation is:

$$
\sigma_k=\exp(u)\sigma_k^s,
\qquad
\eta_k=\exp(v)\eta_k^s.
$$

In log form:

$$
\log \sigma_k
=
u+\log \sigma_k^s,
\qquad
\log \eta_k
=
v+\log \eta_k^s.
$$



$$
\boxed{
\text{multiplication = model restriction,}
\qquad
\text{exponential = positivity transform.}
}
$$

In MATLAB, the final calibration parameter vector could be:

```matlab
theta = [aRaw; bRaw; alphaRaw; betaRaw; rhoRaw];

a     = exp(aRaw);
b     = exp(bRaw);
alpha = exp(alphaRaw);
beta  = exp(betaRaw);
rho   = tanh(rhoRaw);

sigma = alpha .* sigmaShape;
eta   = beta  .* etaShape;
```



$$
\boxed{
\text{reuse your pricing/instrument/temporary-state machinery, but replace local bucket solve by global scaled-shape solve.}
}
$$

## What we reuse

| Existing piece | Keep? | Role |
|---|---:|---|
| `buildPricersLocal` | yes | market and swap pricing |
| IRS/swaption package logic | yes | build calibration instruments |
| `setModelVolatilityState` / `restoreModelVolatilityState` | yes | safe trial model state |
| `adjustCalibrationVolBuckets` | yes | current model bucket convention |
| `getSwaptionPriceG2PP` | yes | model price |
| `g2ppBucketPriceResidual` | no for Bonino final step | keep as test / alternative local bootstrap |

## Up to what level to go now

Go to **Bonino Step 3**, assuming Step 2 shapes are already available:

$$
\sigmaShape=(\sigma^s_0,\ldots,\sigma^s_{n-1}),
\qquad
\etaShape=(\eta^s_0,\ldots,\eta^s_{n-1}).
$$

Then calibrate:

$$
\sigma=\alpha\sigmaShape,
\qquad
\eta=\beta\etaShape.
$$

Given your current project convention that $a,b$ are fixed, the first production version should calibrate:

$$
\theta=(\alpha,\beta,\rho),
$$

or, if $\rho$ is also fixed:

$$
\theta=(\alpha,\beta).
$$

Use raw optimizer variables:

$$
\alpha=e^{u},\qquad \beta=e^{v},\qquad \rho=\tanh(w).
$$

---

# 1. Orchestrator section

Put this **after** setup/pricers and after you have built or received `boninoCalibSet`, `sigmaShape`, `etaShape`.

```matlab
%--------------------------------------------------------
% Bonino Step 3: scaled G2++ global calibration
%--------------------------------------------------------
boninoState = buildBoninoShapeStateG2PP( ...
    obj, args, sigmaShape, etaShape);

[thetaStar, solverInfo] = solveBoninoScaledG2PP( ...
    obj, args, boninoCalibSet, boninoState);

[params, sigmaCalib, etaCalib] = decodeBoninoThetaG2PP( ...
    thetaStar, args, boninoState);

[sigmaModel, etaModel] = adjustCalibrationVolBuckets( ...
    obj, sigmaCalib, etaCalib);

obj.volatility_sigma = sigmaModel(:).';
obj.volatility_eta   = etaModel(:).';

if getStructFieldG2PP(args, 'calibrateRho', true)
    obj.correlation = params.rho;
end

calibratedSigma = sigmaCalib(:);
calibratedEta   = etaCalib(:);

[finalResidual, modelPrice] = boninoScaledSurfaceResidualG2PP( ...
    obj, thetaStar, args, boninoCalibSet, boninoState);

mktPrice = [boninoCalibSet.marketPrice].';
bachelierVol = [boninoCalibSet.bachelierVol].';
arraySwaption = {boninoCalibSet.swaption}.';

diagnostics.bonino = makeBoninoDiagnosticsG2PP( ...
    params, solverInfo, finalResidual, modelPrice, mktPrice, boninoState);

diagnostics.status = "bonino_scaled_calibration_completed";
```

---

# 2. Build Bonino shape state

```matlab
function boninoState = buildBoninoShapeStateG2PP(obj, args, sigmaShape, etaShape)
    % G2++ Bonino Helper: Stores the two volatility shapes obtained from
    % the G1++ slice bootstraps and validates their compatibility with the
    % current model bucket convention.

    arguments
        obj (1,1) GaussianTwoFactors
        args (1,1) struct
        sigmaShape (:,1) double
        etaShape (:,1) double
    end

    %#ok<INUSD>

    sigmaShape = sigmaShape(:);
    etaShape   = etaShape(:);

    if numel(sigmaShape) ~= numel(etaShape)
        error("sigmaShape and etaShape must have the same length.");
    end

    if any(~isfinite(sigmaShape)) || any(sigmaShape < 0)
        error("sigmaShape must be finite and nonnegative.");
    end

    if any(~isfinite(etaShape)) || any(etaShape < 0)
        error("etaShape must be finite and nonnegative.");
    end

    if all(sigmaShape == 0) || all(etaShape == 0)
        error("Bonino volatility shapes cannot be identically zero.");
    end

    boninoState = struct();
    boninoState.sigmaShape = sigmaShape;
    boninoState.etaShape = etaShape;
    boninoState.nbBuckets = numel(sigmaShape);
    boninoState.terminalPolicy = getStructFieldG2PP(args, 'terminalPolicy', 'flatLast');
end
```

---

# 3. Decode Bonino parameters

```matlab
function [params, sigmaCalib, etaCalib] = decodeBoninoThetaG2PP(thetaRaw, args, boninoState)
    % G2++ Bonino Helper: Decodes raw optimizer variables into model
    % parameters and builds the scaled volatility vectors.

    arguments
        thetaRaw (:,1) double
        args (1,1) struct
        boninoState (1,1) struct
    end

    thetaRaw = thetaRaw(:);

    calibrateRho = getStructFieldG2PP(args, 'calibrateRho', true);

    if calibrateRho
        if numel(thetaRaw) ~= 3
            error("thetaRaw must be [logAlpha; logBeta; rhoRaw] when calibrateRho = true.");
        end

        alpha = exp(thetaRaw(1));
        beta  = exp(thetaRaw(2));
        rho   = tanh(thetaRaw(3));

    else
        if numel(thetaRaw) ~= 2
            error("thetaRaw must be [logAlpha; logBeta] when calibrateRho = false.");
        end

        alpha = exp(thetaRaw(1));
        beta  = exp(thetaRaw(2));
        rho   = getStructFieldG2PP(args, 'fixedRho', NaN);

        if ~isfinite(rho) || abs(rho) >= 1
            error("args.fixedRho must be finite and strictly inside (-1,1).");
        end
    end

    sigmaCalib = alpha .* boninoState.sigmaShape;
    etaCalib   = beta  .* boninoState.etaShape;

    params = struct();
    params.alpha = alpha;
    params.beta = beta;
    params.rho = rho;
end
```

---

# 4. Bonino scaled surface residual

```matlab
function [residual, modelPrice, trialState] = boninoScaledSurfaceResidualG2PP(obj, thetaRaw, args, boninoCalibSet, boninoState)
    % G2++ Bonino Helper: Computes the full-surface residual under the
    % scaled-shape Bonino parametrization.

    arguments
        obj (1,1) GaussianTwoFactors
        thetaRaw (:,1) double
        args (1,1) struct
        boninoCalibSet (:,1) struct
        boninoState (1,1) struct
    end

    penalty = getStructFieldG2PP(args, 'residualPenalty', 1e10);

    n = numel(boninoCalibSet);
    residual = penalty .* ones(n, 1);
    modelPrice = NaN(n, 1);

    [params, sigmaCalib, etaCalib] = decodeBoninoThetaG2PP( ...
        thetaRaw, args, boninoState);

    [sigmaModel, etaModel] = adjustCalibrationVolBuckets( ...
        obj, sigmaCalib, etaCalib);

    trialState = struct();
    trialState.params = params;
    trialState.sigmaCalib = sigmaCalib;
    trialState.etaCalib = etaCalib;
    trialState.sigmaModel = sigmaModel;
    trialState.etaModel = etaModel;

    previousState = struct();
    previousState.volatility_sigma = obj.volatility_sigma;
    previousState.volatility_eta = obj.volatility_eta;
    previousState.correlation = obj.correlation;

    obj.volatility_sigma = sigmaModel(:).';
    obj.volatility_eta   = etaModel(:).';
    obj.correlation      = params.rho;

    for i = 1:n

        try
            modelPrice(i) = obj.getSwaptionPriceG2PP( ...
                boninoCalibSet(i).strike, ...
                boninoCalibSet(i).expiry, ...
                boninoCalibSet(i).paymentDates, ...
                'direction', boninoCalibSet(i).direction, ...
                'notional', boninoCalibSet(i).notional);

            if isfinite(modelPrice(i))
                weight = getBoninoWeightG2PP(boninoCalibSet(i));
                residual(i) = weight .* (modelPrice(i) - boninoCalibSet(i).marketPrice);
            end

        catch
            modelPrice(i) = NaN;
            residual(i) = penalty;
        end
    end

    obj.volatility_sigma = previousState.volatility_sigma;
    obj.volatility_eta   = previousState.volatility_eta;
    obj.correlation      = previousState.correlation;
end
```

---

# 5. Solve Bonino scaled G2++

```matlab
function [thetaStar, solverInfo] = solveBoninoScaledG2PP(obj, args, boninoCalibSet, boninoState)
    % G2++ Bonino Helper: Solves the reduced global calibration problem
    % using the Bonino scaled volatility shapes.

    arguments
        obj (1,1) GaussianTwoFactors
        args (1,1) struct
        boninoCalibSet (:,1) struct
        boninoState (1,1) struct
    end

    calibrateRho = getStructFieldG2PP(args, 'calibrateRho', true);

    alpha0 = getStructFieldG2PP(args, 'initialAlpha', 1.0);
    beta0  = getStructFieldG2PP(args, 'initialBeta', 1.0);

    if calibrateRho
        rho0 = getStructFieldG2PP(args, 'initialRho', 0.0);
        theta0 = [log(alpha0); log(beta0); safeAtanhG2PP(rho0)];

        lb = [
            log(getStructFieldG2PP(args, 'alphaLowerBound', 1e-4))
            log(getStructFieldG2PP(args, 'betaLowerBound',  1e-4))
            safeAtanhG2PP(getStructFieldG2PP(args, 'rhoLowerBound', -0.999))
        ];

        ub = [
            log(getStructFieldG2PP(args, 'alphaUpperBound', 100.0))
            log(getStructFieldG2PP(args, 'betaUpperBound',  100.0))
            safeAtanhG2PP(getStructFieldG2PP(args, 'rhoUpperBound', 0.999))
        ];
    else
        theta0 = [log(alpha0); log(beta0)];

        lb = [
            log(getStructFieldG2PP(args, 'alphaLowerBound', 1e-4))
            log(getStructFieldG2PP(args, 'betaLowerBound',  1e-4))
        ];

        ub = [
            log(getStructFieldG2PP(args, 'alphaUpperBound', 100.0))
            log(getStructFieldG2PP(args, 'betaUpperBound',  100.0))
        ];
    end

    objective = @(theta) boninoScaledSurfaceResidualG2PP( ...
        obj, theta, args, boninoCalibSet, boninoState);

    opts = optimoptions('lsqnonlin', ...
        'Display', getStructFieldG2PP(args, 'display', 'none'), ...
        'MaxIterations', getStructFieldG2PP(args, 'maxIterations', 200), ...
        'FunctionTolerance', getStructFieldG2PP(args, 'functionTolerance', 1e-10), ...
        'StepTolerance', getStructFieldG2PP(args, 'stepTolerance', 1e-10));

    [thetaStar, resnorm, residual, exitflag, output] = lsqnonlin( ...
        @(theta) objective(theta), theta0, lb, ub, opts);

    thetaStar = thetaStar(:);

    solverInfo = struct();
    solverInfo.resnorm = resnorm;
    solverInfo.residual = residual;
    solverInfo.exitflag = exitflag;
    solverInfo.output = output;
    solverInfo.theta0 = theta0;
    solverInfo.lowerBound = lb;
    solverInfo.upperBound = ub;
end
```

---

# 6. Build Bonino calibration set from a normalized surface pack

This function assumes you already converted your `SwaptionCube` into a simple structure. I would not hard-code the cube API here.

Expected fields:

```matlab
surfacePack.expiryDates      % n x 1
surfacePack.swapStartDates   % n x m
surfacePack.maturityDates    % n x m
surfacePack.volMatrix        % n x m
```

Optional:

```matlab
surfacePack.strikeMatrix
surfacePack.weightMatrix
surfacePack.directionMatrix
```

```matlab
function boninoCalibSet = buildBoninoCalibrationSetG2PP(obj, args, surfacePack, pricers)
    % G2++ Bonino Helper: Builds all swaption instruments and market prices
    % from a normalized ATM swaption surface pack.

    arguments
        obj (1,1) GaussianTwoFactors
        args (1,1) struct
        surfacePack (1,1) struct
        pricers (1,1) struct
    end

    expiryDates = surfacePack.expiryDates(:);
    swapStartDates = surfacePack.swapStartDates;
    maturityDates = surfacePack.maturityDates;
    volMatrix = surfacePack.volMatrix;

    [nExp, nTenor] = size(volMatrix);

    if numel(expiryDates) ~= nExp
        error("surfacePack.expiryDates must match the number of volMatrix rows.");
    end

    if any(size(swapStartDates) ~= [nExp, nTenor])
        error("surfacePack.swapStartDates must have the same size as volMatrix.");
    end

    if any(size(maturityDates) ~= [nExp, nTenor])
        error("surfacePack.maturityDates must have the same size as volMatrix.");
    end

    k = 0;
    boninoCalibSet = struct([]);

    for i = 1:nExp
        for j = 1:nTenor

            if ~isfinite(volMatrix(i,j))
                continue;
            end

            k = k + 1;

            boninoCalibSet(k,1) = buildBoninoCalibrationPointG2PP( ...
                obj, args, surfacePack, pricers, i, j);
        end
    end

    if isempty(boninoCalibSet)
        error("No valid swaption quote found in surfacePack.");
    end
end
```

---

# 7. Build one Bonino calibration point

```matlab
function point = buildBoninoCalibrationPointG2PP(obj, args, surfacePack, pricers, rowIdx, colIdx)
    % G2++ Bonino Helper: Builds one swaption instrument and its Bachelier
    % market price from a surface node.

    arguments
        obj (1,1) GaussianTwoFactors
        args (1,1) struct
        surfacePack (1,1) struct
        pricers (1,1) struct
        rowIdx (1,1) double {mustBeInteger, mustBePositive}
        colIdx (1,1) double {mustBeInteger, mustBePositive}
    end

    expiryDate = surfacePack.expiryDates(rowIdx);
    swapStartDate = surfacePack.swapStartDates(rowIdx, colIdx);
    maturityDate = surfacePack.maturityDates(rowIdx, colIdx);
    volInput = surfacePack.volMatrix(rowIdx, colIdx);

    irs = copy(args.irs);
    irs.setNotional(getStructFieldG2PP(args, 'notional', 1e6));
    irs.setDates(swapStartDate, maturityDate);

    [npv, ~, ~] = pricers.swap.IRSVanille(irs, ...
        'measures', true, ...
        'details', true);

    if isfield(surfacePack, 'strikeMatrix') && isfinite(surfacePack.strikeMatrix(rowIdx, colIdx))
        K = surfacePack.strikeMatrix(rowIdx, colIdx);
    else
        K = npv.FairSwapRate;
    end

    if isfield(surfacePack, 'directionMatrix')
        direction = char(surfacePack.directionMatrix(rowIdx, colIdx));
    else
        if K > npv.FairSwapRate
            direction = 'PAY';
        else
            direction = 'REC';
        end
    end

    irs.direction = direction;

    swaption = Swaption(irs, ...
        'exerciseDate', expiryDate, ...
        'strike', K, ...
        'settleType', getStructFieldG2PP(args, 'settleType', 'CashSettled'));

    [~, fixedLegPDates, ~] = irs.fixedLeg01(swapStartDate);

    fixedLegPDates = fixedLegPDates(:).';

    if numel(fixedLegPDates) < 2
        error("Fixed leg payment dates are not sufficient.");
    end

    pricingDateNum = pricingDateNumLocal(obj.pricingDate);

    paymentDates = yearfracExtend( ...
        pricingDateNum, fixedLegPDates(2:end), obj.basis);
    paymentDates = paymentDates(:).';

    expiry = yearfracExtend(pricingDateNum, expiryDate, obj.basis);

    try
        [marketPrice, ~, ~, bachelierVol] = pricers.bachelier.Swaption( ...
            swaption, ...
            'measures', false, ...
            'vol', volInput, ...
            'irsRes', npv);
    catch
        [marketPrice, ~, ~] = pricers.bachelier.Swaption( ...
            swaption, ...
            'measures', false, ...
            'vol', volInput, ...
            'irsRes', npv);

        bachelierVol = volInput;
    end

    if ~isfinite(marketPrice)
        error("Market price is not finite at surface node (%d,%d).", rowIdx, colIdx);
    end

    if isfield(surfacePack, 'weightMatrix') && isfinite(surfacePack.weightMatrix(rowIdx, colIdx))
        weight = surfacePack.weightMatrix(rowIdx, colIdx);
    else
        weight = 1.0;
    end

    point = struct();
    point.row = rowIdx;
    point.col = colIdx;
    point.irs = irs;
    point.swaption = swaption;
    point.strike = K;
    point.direction = direction;
    point.expiry = expiry;
    point.paymentDates = paymentDates;
    point.marketPrice = marketPrice;
    point.bachelierVol = bachelierVol;
    point.notional = getStructFieldG2PP(args, 'notional', 1e6);
    point.weight = weight;
end
```

---

# 8. Diagnostics

```matlab
function diagnostics = makeBoninoDiagnosticsG2PP(params, solverInfo, residual, modelPrice, marketPrice, boninoState)
    % G2++ Bonino Helper: Stores final global scaled calibration diagnostics.

    arguments
        params (1,1) struct
        solverInfo (1,1) struct
        residual (:,1) double
        modelPrice (:,1) double
        marketPrice (:,1) double
        boninoState (1,1) struct
    end

    valid = isfinite(residual);

    diagnostics = struct();
    diagnostics.params = params;
    diagnostics.solverInfo = solverInfo;
    diagnostics.nbInstruments = numel(residual);
    diagnostics.nbValidResiduals = sum(valid);
    diagnostics.rmsePrice = sqrt(mean(residual(valid).^2));
    diagnostics.maxAbsPriceError = max(abs(residual(valid)));
    diagnostics.meanAbsPriceError = mean(abs(residual(valid)));
    diagnostics.modelPrice = modelPrice;
    diagnostics.marketPrice = marketPrice;
    diagnostics.residual = residual;
    diagnostics.sigmaShape = boninoState.sigmaShape;
    diagnostics.etaShape = boninoState.etaShape;
end
```

---

# 9. Weight getter

```matlab
function weight = getBoninoWeightG2PP(calibPoint)
    % G2++ Bonino Utility: Returns the residual weight of one surface point.

    if isfield(calibPoint, 'weight') && ~isempty(calibPoint.weight)
        weight = calibPoint.weight;
    else
        weight = 1.0;
    end

    if ~isfinite(weight) || weight < 0
        error("Calibration weight must be finite and nonnegative.");
    end
end
```

---

# 10. Safe inverse hyperbolic tangent

```matlab
function x = safeAtanhG2PP(rho)
    % G2++ Bonino Utility: Stable inverse transform for correlation.

    if ~isfinite(rho)
        error("rho must be finite.");
    end

    epsRho = 1e-10;
    rho = min(max(rho, -1 + epsRho), 1 - epsRho);

    x = 0.5 .* log((1 + rho) ./ (1 - rho));
end
```

---

# 11. Struct getter

```matlab
function value = getStructFieldG2PP(s, fieldName, defaultValue)
    % G2++ Calibration Utility: Struct field getter with default.

    if isfield(s, fieldName) && ~isempty(s.(fieldName))
        value = s.(fieldName);
    else
        value = defaultValue;
    end
end
```

---

# 12. Robust pricing date conversion

```matlab
function pricingDateNum = pricingDateNumLocal(pricingDate)
    % G2++ Calibration Utility: Converts pricing date to datenum.

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

## Final placement

Your current file should now have two different calibration paths:

```text
A. g2ppBucketPriceResidual
   → local G2++ bucket experiments
   → not the Bonino final step

B. boninoScaledSurfaceResidualG2PP
   → Bonino Step 3
   → global scaled-shape calibration
```

The next missing block is **not** more G2++ local residual code. It is the Step 2 adapter:

```matlab
[sigmaShape, etaShape] = runTwoG1PPSliceBootstraps(...)
```

For now, the clean production move is to pass these shapes as inputs:

```matlab
args.sigmaShape = sigmaShapeFromShortTenorG1PP;
args.etaShape   = etaShapeFromLongTenorG1PP;
```

Then validate the Bonino Step 3 G2++ layer independently.


# Calibration Pipeline Review

> **Clarifying setup for helper architecture**
>
> I'll focus on setting up the architecture for the helpers. The user mentioned a reference to a PDF ("Bonino style"), which may need to be cited if relevant. I don't need to use web searches, and there's no need for personal context. I'll set up the names and define their responsibilities, maybe using skeleton function signatures but without full code yet, as it's a first iteration.
>
> **Clarifying citation process for pipeline and G2++ kernel**
>
> I need to mention the staged pipeline and G2++ kernel after shapes are prepared. Regarding the citation, I’m unsure about including the PDF lines. The search result didn’t give line markers. Usually, file search citations should include line ranges, but since we don’t have those, I’ll need to fetch them manually.
>
> **Clarifying citation pointer**
>
> Oops, I think I need to correct the pointer to "turn" for the citation. The results in [56] might include "turn6file0." It could be a good idea to try using "6:0" as the pointer or perhaps click through manually to confirm the exact line references.
>
> Thought for 13s

Good. We freeze the orchestrator. We define only the **SETUP helper contracts**.

The XG2++ setup must prepare clean structures; it must not calibrate, price G2++, or mutate model volatility except where explicitly stated. This matches the staged architecture in the brief: market preparation first, then one-factor initialization, then two-factor reduced calibration; the G2++ pricing kernel comes only after volatility shapes are prepared. `README2 (1).pdf`

```text
XG2++ SETUP HELPERS
│
├─ buildPricersG2PP
│     → production pricing handles
│
├─ buildVolSurfacePackXG2PP
│     → normalized market surface
│
└─ initCalibXG2PP
      → calibration control/state before G1++ and G2++
```

# 1. `buildPricersG2PP`

## Responsibility

Build the pricing engines required by calibration.

```text
buildPricersG2PP
│
├─ PricingDCF(obj.DCF)
│     → IRS deterministic pricing
│     → fair swap rate
│     → annuity / leg details if needed
│
├─ BachelierModel(obj.swaptionVolCube, obj.DCF.discount, obj.DCF.forecast)
│     → market quote convention
│
└─ PricingBachelierModel
      → market swaption target price
```

## Output contract

```matlab
pricers.swap
pricers.bachelierModel
pricers.bachelier
```

Optional later:

```matlab
pricers.swapAlt
```

for strike-adjustment cases with alternative forecast/discount curves.

## Boundary

This helper should not build IRS, swaptions, or calibration instruments.

It only returns engines.

---

# 2. `buildVolSurfacePackXG2PP`

## Responsibility

Normalize the swaption cube/surface into a simple calibration data object.

This corresponds to the market-normalization block in the brief: cube/surface $\rightarrow$ expiry dates, tenors, maturities, vol matrix, optional strikes, optional weights. `README2 (1).pdf`

```text
buildVolSurfacePackXG2PP
│
├─ input
│     → obj.swaptionVolCube
│     → moneyness
│     → optional extraction policy
│
├─ extract
│     → vol surface
│     → expiries
│     → tenors
│     → startSwapDates
│     → matSwapDates
│
├─ normalize dates
│     → pricingDateNum
│     → expiryDatesYr
│     → startSwapDatesYr
│     → matSwapDatesYr
│     → tenorsYr
│
└─ validate
      → finite vol entries
      → dimension consistency
      → maturity > start > expiry
```

## Output contract

```matlab
surfacePack.moneyness
surfacePack.volMatrix

surfacePack.expiryDates
surfacePack.expiryDatesYr

surfacePack.tenors
surfacePack.tenorsYr

surfacePack.startSwapDates
surfacePack.startSwapDatesYr

surfacePack.matSwapDates
surfacePack.matSwapDatesYr

surfacePack.nbExpiries
surfacePack.nbTenors
surfacePack.nbQuotes

surfacePack.validMask
surfacePack.quoteRows
surfacePack.quoteCols
```

Optional, not mandatory in first version:

```matlab
surfacePack.weights
surfacePack.strikes
surfacePack.volType
surfacePack.shift
```

## Boundary

This helper should not construct `IRSVanille`.

It should not construct `Swaption`.

It should not call the Bachelier pricer.

It should only produce market geometry.

---

# 3. `initCalibXG2PP`

## Responsibility

Build the control state for the XG2++ procedure.

```text
initCalibXG2PP
│
├─ choose slices
│     → short-tenor slice
│     → long-tenor slice
│
├─ initialize parameter controls
│     → initial a, b
│     → initial rho
│     → initial scaling parameters
│
├─ initialize bounds
│     → mean reversion bounds
│     → volatility scale bounds
│     → rho bounds
│
├─ initialize solver options
│     → global G1++ options
│     → slice bootstrap options
│     → scaled G2++ options
│
└─ carry previous inputs
      → already computed sigma shape
      → already computed eta shape
      → cached starts if any
```

## Output contract

```matlab
xg2ppState.mode

xg2ppState.shortTenorIdx
xg2ppState.longTenorIdx
xg2ppState.shortTenor
xg2ppState.longTenor

xg2ppState.initial.a
xg2ppState.initial.b
xg2ppState.initial.rho
xg2ppState.initial.alphaSigma
xg2ppState.initial.alphaEta

xg2ppState.bounds.lower
xg2ppState.bounds.upper

xg2ppState.options.globalG1PP
xg2ppState.options.sliceBootstrap
xg2ppState.options.scaledG2PP

xg2ppState.previous.sigmaShape
xg2ppState.previous.etaShape
xg2ppState.previous.theta
```

## Boundary

This helper should not calibrate the global G1++ model.

It should not bootstrap the two slices.

It should not set `obj.volatility_sigma` or `obj.volatility_eta`.

It only prepares the **control object**.

---

# 4. First setup architecture

```text
SETUP
│
├─ buildPricersG2PP
│   │
│   ├─ pricers.swap
│   ├─ pricers.bachelierModel
│   └─ pricers.bachelier
│
├─ buildVolSurfacePackXG2PP
│   │
│   ├─ surfacePack.volMatrix
│   ├─ surfacePack.expiryDates / expiryDatesYr
│   ├─ surfacePack.tenors / tenorsYr
│   ├─ surfacePack.startSwapDates / startSwapDatesYr
│   ├─ surfacePack.matSwapDates / matSwapDatesYr
│   ├─ surfacePack.validMask
│   └─ surfacePack.quoteRows / quoteCols
│
└─ initCalibXG2PP
    │
    ├─ xg2ppState.mode
    ├─ xg2ppState.shortTenorIdx
    ├─ xg2ppState.longTenorIdx
    ├─ xg2ppState.initial
    ├─ xg2ppState.bounds
    ├─ xg2ppState.options
    └─ xg2ppState.previous
```

# 5. Minimal call order

```matlab
pricers = obj.buildPricersG2PP();

surfacePack = obj.buildVolSurfacePackXG2PP( ...
    calibMoneyness);

xg2ppState = obj.initCalibXG2PP( ...
    surfacePack, ...
    args);
```

After this setup, nothing has been calibrated yet.

The next block will be:

```text
GLOBAL G1++ INITIALIZATION
│
├─ calibrateGlobalG1PPXG2PP
├─ bootstrapG1PPSliceXG2PP  short tenor
└─ bootstrapG1PPSliceXG2PP  long tenor
```

# 6. Design rule for this block

```text
buildPricersG2PP
    owns pricing-engine construction.

buildVolSurfacePackXG2PP
    owns market geometry.

initCalibXG2PP
    owns calibration controls.

None of them owns G2++ pricing.
None of them owns optimization.
None of them owns final parameter commitment.
```

## Response:
5/15/2026, 12:24:51 PM

> **Considering code structure for user request**
>
> The user asked for code for each helper. I’ll likely need to provide MATLAB code in full, which includes these three functions: `buildPricersG2PP`, `buildVolSurfacePackXG2PP`, and `initCalibXG2PP`. I should also consider including the `getPricingDateNum` helper but avoid the file search unless it's necessary for citations. Let's structure and keep it concise.
>
> **Structuring MATLAB functions inside class methods**
>
> MATLAB allows nested functions within the main function block, even inside class methods, so that should work. The user likely wants these inside class methods, not externally. Since they may already have `getPricingDateNum`, I can reference `obj.getPricingDateNum` without including it. I'll clarify that it’s necessary for robustness and mention the required arguments for `initCalibXG2PP`.
>
> **Deciding on citation and argument structure for code**
>
> For the code, I’ll use a default `0` if necessary, and adjust for objects that don’t have `extractSurface` with an error catch. Regarding arguments, I'll ensure to use `char` for string validation and avoid citations. Since this is just the code, citations aren’t needed unless referencing the design.
>
> Thought for 1m 1s

Paste these inside `methods` of `GaussianTwoFactors`.

```matlab
function pricers = buildPricersG2PP(obj, args)
    % G2++ Setup Helper: Builds production pricing engines needed by the
    % calibration pipeline.
    %
    % OUTPUT:
    % pricers.swap           | PricingDCF             : deterministic IRS pricer
    % pricers.bachelierModel | BachelierModel         : market quote model
    % pricers.bachelier      | PricingBachelierModel  : market swaption pricer
    % pricers.swapAlt        | PricingDCF / []        : optional alternative DCF pricer

    arguments
        obj (1,1) GaussianTwoFactors;
        args.discount = [];
        args.forecast = [];
    end

    pricers = struct();

    % Main DCF pricer, aligned with the model DCF.
    pricers.swap = PricingDCF(obj.DCF);

    % Optional alternative DCF pricer, kept for future strike-adjustment cases.
    pricers.swapAlt = [];

    if ~isempty(args.discount) && ~isempty(args.forecast)
        pricers.swapAlt = PricingDCF(DCFModel(args.discount, args.forecast));

    elseif ~isempty(args.discount)
        pricers.swapAlt = PricingDCF(DCFModel(args.discount));

    elseif ~isempty(args.forecast)
        pricers.swapAlt = PricingDCF(DCFModel(args.forecast));
    end

    if isempty(obj.swaptionVolCube)
        error("obj.swaptionVolCube must be defined before building Bachelier pricer.");
    end

    % Bachelier pricer used to convert market vol quotes into target prices.
    pricers.bachelierModel = BachelierModel( ...
        obj.swaptionVolCube, ...
        obj.DCF.discount, ...
        obj.DCF.forecast);

    pricers.bachelier = PricingBachelierModel(pricers.bachelierModel);
end
```

```matlab
function surfacePack = buildVolSurfacePackXG2PP(obj, calibMoneyness, args)
    % G2++ Setup Helper: Extracts and normalizes a swaption volatility
    % surface from the cube for XG2++ calibration.
    %
    % INPUT:
    % calibMoneyness | double | scalar : moneyness slice, default ATM = 0
    %
    % OUTPUT:
    % surfacePack | struct : normalized market surface geometry and quotes

    arguments
        obj (1,1) GaussianTwoFactors;
        calibMoneyness double = 0;
        args.dimension (1,:) char = 'moneyness';
        args.weights = [];
        args.strikes = [];
        args.validate (1,1) logical = true;
    end

    if isempty(calibMoneyness)
        calibMoneyness = 0;
    end

    if isempty(obj.swaptionVolCube)
        error("obj.swaptionVolCube is empty. Cannot build XG2++ surface pack.");
    end

    pricingDateNum = obj.getPricingDateNum(obj.pricingDate);

    % Extract surface from the production cube.
    try
        rawSurface = obj.swaptionVolCube.extractSurface( ...
            'dimension', args.dimension, ...
            'value', calibMoneyness);
    catch ME
        error("Could not extract swaption surface from cube: %s", ME.message);
    end

    % Read core fields, robustly accepting struct or object output.
    moneyness = readField(rawSurface, {'moneyness'}, false);
    if isempty(moneyness)
        moneyness = calibMoneyness;
    end

    volMatrix = readField(rawSurface, {'vol', 'volMatrix'}, true);

    expiryDates = readField(rawSurface, ...
        {'expiriesDates', 'expiryDates', 'exerciseDates'}, true);

    tenors = readField(rawSurface, {'tenors', 'swapTenors'}, true);

    startSwapDates = readField(rawSurface, ...
        {'startSwapDates', 'swapStartDates', 'deliveryDates'}, true);

    matSwapDates = readField(rawSurface, ...
        {'matSwapDates', 'maturityDates', 'swapsMaturities'}, true);

    expiryDates = expiryDates(:);
    tenors = tenors(:);

    nbExpiries = numel(expiryDates);
    nbTenors = numel(tenors);

    if size(volMatrix,1) ~= nbExpiries || size(volMatrix,2) ~= nbTenors
        error("volMatrix size must be nbExpiries x nbTenors.");
    end

    if ~isequal(size(startSwapDates), size(volMatrix))
        error("startSwapDates must have the same size as volMatrix.");
    end

    if ~isequal(size(matSwapDates), size(volMatrix))
        error("matSwapDates must have the same size as volMatrix.");
    end

    % Convert dates to year fractions from pricing date.
    expiryDatesYr = yearfracExtend(pricingDateNum, expiryDates, obj.basis);
    expiryDatesYr = expiryDatesYr(:);

    startSwapDatesYr = yearfracExtend(pricingDateNum, startSwapDates, obj.basis);
    matSwapDatesYr   = yearfracExtend(pricingDateNum, matSwapDates, obj.basis);

    tenorsYr = parseTenorVector(tenors);
    tenorsYr = tenorsYr(:);

    % Quote mask.
    validMask = isfinite(volMatrix);
    [quoteRows, quoteCols] = find(validMask);
    quoteValues = volMatrix(validMask);

    % Optional weights.
    if isempty(args.weights)
        weights = ones(size(volMatrix));
    elseif isscalar(args.weights)
        weights = args.weights .* ones(size(volMatrix));
    else
        weights = args.weights;
        if ~isequal(size(weights), size(volMatrix))
            error("weights must be scalar or have the same size as volMatrix.");
        end
    end

    % Optional strikes.
    strikes = args.strikes;
    if ~isempty(strikes) && ~isequal(size(strikes), size(volMatrix))
        error("strikes must be empty or have the same size as volMatrix.");
    end

    if args.validate
        expiryMat = repmat(expiryDates(:), 1, nbTenors);

        if any(startSwapDates(:) <= expiryMat(:))
            error("Each swap start date must be strictly after the swaption expiry.");
        end

        if any(matSwapDates(:) <= startSwapDates(:))
            error("Each swap maturity date must be strictly after the swap start date.");
        end

        if nnz(validMask) == 0
            error("No finite volatility quote found in extracted surface.");
        end

        if any(~isfinite(expiryDatesYr))
            error("Non-finite expiry year fraction detected.");
        end

        if any(~isfinite(tenorsYr))
            error("Non-finite tenor year fraction detected.");
        end
    end

    % Main output pack.
    surfacePack = struct();

    surfacePack.moneyness = moneyness;
    surfacePack.volMatrix = volMatrix;

    surfacePack.expiryDates = expiryDates;
    surfacePack.expiryDatesYr = expiryDatesYr;

    surfacePack.tenors = tenors;
    surfacePack.tenorsYr = tenorsYr;

    surfacePack.startSwapDates = startSwapDates;
    surfacePack.startSwapDatesYr = startSwapDatesYr;

    surfacePack.matSwapDates = matSwapDates;
    surfacePack.matSwapDatesYr = matSwapDatesYr;

    surfacePack.weights = weights;
    surfacePack.strikes = strikes;

    surfacePack.validMask = validMask;
    surfacePack.quoteRows = quoteRows;
    surfacePack.quoteCols = quoteCols;
    surfacePack.quoteValues = quoteValues;

    surfacePack.nbExpiries = nbExpiries;
    surfacePack.nbTenors = nbTenors;
    surfacePack.nbQuotes = nnz(validMask);

    surfacePack.pricingDateNum = pricingDateNum;
    surfacePack.basis = obj.basis;

    % -------- nested utilities --------

    function value = readField(source, names, required)
        value = [];

        for k = 1:numel(names)
            name = names{k};

            if isstruct(source) && isfield(source, name)
                value = source.(name);
                return;
            end

            if isobject(source) && isprop(source, name)
                value = source.(name);
                return;
            end
        end

        if required
            error("Missing required field/property: %s", strjoin(names, " / "));
        end
    end

    function tenorYears = parseTenorVector(tenorInput)
        if isnumeric(tenorInput)
            tenorYears = tenorInput(:);
            return;
        end

        if isstring(tenorInput)
            tenorInput = cellstr(tenorInput(:));
        elseif iscell(tenorInput)
            tenorInput = tenorInput(:);
        else
            error("Unsupported tenor format.");
        end

        tenorYears = NaN(numel(tenorInput), 1);

        for kk = 1:numel(tenorInput)
            tenorYears(kk) = parseSingleTenor(tenorInput{kk});
        end
    end

    function y = parseSingleTenor(x)
        if isnumeric(x)
            y = double(x);
            return;
        end

        txt = strtrim(char(x));

        token = regexp(txt, '^([0-9]*\.?[0-9]+)\s*([DdWwMmYy]?)$', ...
            'tokens', 'once');

        if isempty(token)
            error("Could not parse tenor: %s", txt);
        end

        value = str2double(token{1});
        unit = upper(token{2});

        switch unit
            case {'', 'Y'}
                y = value;
            case 'M'
                y = value / 12;
            case 'W'
                y = value / 52;
            case 'D'
                y = value / 365;
            otherwise
                error("Unsupported tenor unit: %s", unit);
        end
    end
end
```

```matlab
function xg2ppState = initCalibXG2PP(obj, surfacePack, args)
    % G2++ Setup Helper: Initializes the XG2++ calibration control state.
    %
    % The function does not calibrate, does not price, and does not mutate
    % the model volatility state.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;

        args.mode (1,:) char = 'scaledSurface';

        args.shortTenorIdx (1,1) double = NaN;
        args.longTenorIdx  (1,1) double = NaN;

        args.initialA (1,1) double = NaN;
        args.initialB (1,1) double = NaN;
        args.initialRho (1,1) double = -0.50;
        args.initialAlphaSigma (1,1) double = 1.00;
        args.initialAlphaEta   (1,1) double = 1.00;

        args.aLower (1,1) double = 1e-4;
        args.aUpper (1,1) double = 5.0;
        args.bLower (1,1) double = 1e-4;
        args.bUpper (1,1) double = 5.0;

        args.rhoLower (1,1) double = -0.999;
        args.rhoUpper (1,1) double =  0.999;

        args.alphaSigmaLower (1,1) double = 1e-4;
        args.alphaSigmaUpper (1,1) double = 10.0;
        args.alphaEtaLower   (1,1) double = 1e-4;
        args.alphaEtaUpper   (1,1) double = 10.0;

        args.globalG1PPOptions = [];
        args.sliceBootstrapOptions = [];
        args.scaledG2PPOptions = [];

        args.alreadyComputedSigmaShape (:,1) double = [];
        args.alreadyComputedEtaShape   (:,1) double = [];
        args.previousTheta (:,1) double = [];

        args.terminalPolicy (1,:) char = 'flatLast';
        args.localMode (1,:) char = 'scaled';
    end

    requiredFields = ["volMatrix", "tenorsYr", "validMask", ...
                      "nbExpiries", "nbTenors", "nbQuotes"];

    for k = 1:numel(requiredFields)
        if ~isfield(surfacePack, requiredFields(k))
            error("surfacePack is missing required field: %s", requiredFields(k));
        end
    end

    tenorsYr = surfacePack.tenorsYr(:);
    validMask = surfacePack.validMask;

    if numel(tenorsYr) ~= surfacePack.nbTenors
        error("surfacePack.tenorsYr must have length nbTenors.");
    end

    if ~isequal(size(validMask), size(surfacePack.volMatrix))
        error("surfacePack.validMask must have the same size as volMatrix.");
    end

    % Select short and long tenor columns.
    shortIdx = args.shortTenorIdx;
    longIdx  = args.longTenorIdx;

    validTenorCols = find(any(validMask, 1));

    if isempty(validTenorCols)
        error("No tenor column contains finite market quotes.");
    end

    if isnan(shortIdx)
        shortIdx = validTenorCols(1);
    end

    if isnan(longIdx)
        longIdx = validTenorCols(end);
    end

    if shortIdx < 1 || shortIdx > surfacePack.nbTenors
        error("shortTenorIdx is outside the tenor grid.");
    end

    if longIdx < 1 || longIdx > surfacePack.nbTenors
        error("longTenorIdx is outside the tenor grid.");
    end

    if ~any(validMask(:, shortIdx))
        error("Selected shortTenorIdx contains no finite market quotes.");
    end

    if ~any(validMask(:, longIdx))
        error("Selected longTenorIdx contains no finite market quotes.");
    end

    % Initial mean reversions.
    initialA = args.initialA;
    initialB = args.initialB;

    if isnan(initialA)
        initialA = safePositiveDefault(obj.meanReversion_x, 0.10);
    end

    if isnan(initialB)
        initialB = safePositiveDefault(obj.meanReversion_y, 0.03);
    end

    if initialA <= 0 || initialB <= 0
        error("Initial mean reversions must be strictly positive.");
    end

    if abs(args.initialRho) >= 1
        error("initialRho must satisfy abs(initialRho) < 1.");
    end

    if args.aLower <= 0 || args.bLower <= 0
        error("Mean-reversion lower bounds must be strictly positive.");
    end

    if args.aLower >= args.aUpper
        error("Invalid bounds: aLower must be smaller than aUpper.");
    end

    if args.bLower >= args.bUpper
        error("Invalid bounds: bLower must be smaller than bUpper.");
    end

    if args.rhoLower <= -1 || args.rhoUpper >= 1 || args.rhoLower >= args.rhoUpper
        error("Invalid rho bounds.");
    end

    if args.alphaSigmaLower <= 0 || args.alphaEtaLower <= 0
        error("Volatility scale lower bounds must be strictly positive.");
    end

    if args.alphaSigmaLower >= args.alphaSigmaUpper
        error("Invalid alphaSigma bounds.");
    end

    if args.alphaEtaLower >= args.alphaEtaUpper
        error("Invalid alphaEta bounds.");
    end

    % Default solver options.
    globalG1PPOptions = args.globalG1PPOptions;
    sliceBootstrapOptions = args.sliceBootstrapOptions;
    scaledG2PPOptions = args.scaledG2PPOptions;

    if isempty(globalG1PPOptions)
        globalG1PPOptions = makeFsolveOptions();
    end

    if isempty(sliceBootstrapOptions)
        sliceBootstrapOptions = makeFsolveOptions();
    end

    if isempty(scaledG2PPOptions)
        scaledG2PPOptions = makeLsqnonlinOptions();
    end

    thetaNames = ["a"; "b"; "rho"; "alphaSigma"; "alphaEta"];

    theta0 = [
        initialA
        initialB
        args.initialRho
        args.initialAlphaSigma
        args.initialAlphaEta
    ];

    lb = [
        args.aLower
        args.bLower
        args.rhoLower
        args.alphaSigmaLower
        args.alphaEtaLower
    ];

    ub = [
        args.aUpper
        args.bUpper
        args.rhoUpper
        args.alphaSigmaUpper
        args.alphaEtaUpper
    ];

    if any(theta0 < lb) || any(theta0 > ub)
        error("Initial XG2++ parameter vector is outside bounds.");
    end

    % Output state.
    xg2ppState = struct();

    xg2ppState.mode = args.mode;
    xg2ppState.localMode = args.localMode;
    xg2ppState.terminalPolicy = args.terminalPolicy;

    xg2ppState.shortTenorIdx = shortIdx;
    xg2ppState.longTenorIdx = longIdx;
    xg2ppState.shortTenor = tenorsYr(shortIdx);
    xg2ppState.longTenor = tenorsYr(longIdx);

    xg2ppState.thetaNames = thetaNames;
    xg2ppState.theta0 = theta0;

    xg2ppState.initial = struct();
    xg2ppState.initial.a = initialA;
    xg2ppState.initial.b = initialB;
    xg2ppState.initial.rho = args.initialRho;
    xg2ppState.initial.alphaSigma = args.initialAlphaSigma;
    xg2ppState.initial.alphaEta = args.initialAlphaEta;

    xg2ppState.bounds = struct();
    xg2ppState.bounds.lower = lb;
    xg2ppState.bounds.upper = ub;
    xg2ppState.bounds.a = [args.aLower, args.aUpper];
    xg2ppState.bounds.b = [args.bLower, args.bUpper];
    xg2ppState.bounds.rho = [args.rhoLower, args.rhoUpper];
    xg2ppState.bounds.alphaSigma = [args.alphaSigmaLower, args.alphaSigmaUpper];
    xg2ppState.bounds.alphaEta = [args.alphaEtaLower, args.alphaEtaUpper];

    xg2ppState.options = struct();
    xg2ppState.options.globalG1PP = globalG1PPOptions;
    xg2ppState.options.sliceBootstrap = sliceBootstrapOptions;
    xg2ppState.options.scaledG2PP = scaledG2PPOptions;

    xg2ppState.previous = struct();
    xg2ppState.previous.sigmaShape = args.alreadyComputedSigmaShape(:);
    xg2ppState.previous.etaShape = args.alreadyComputedEtaShape(:);
    xg2ppState.previous.theta = args.previousTheta(:);

    xg2ppState.surfaceSize = size(surfacePack.volMatrix);
    xg2ppState.nbQuotes = surfacePack.nbQuotes;
    xg2ppState.createdAt = datetime("now");

    % -------- nested utilities --------

    function value = safePositiveDefault(candidate, fallback)
        if isempty(candidate) || ~isfinite(candidate) || candidate <= 0
            value = fallback;
        else
            value = candidate;
        end
    end

    function opt = makeFsolveOptions()
        try
            opt = optimoptions(@fsolve, ...
                'Display', 'none', ...
                'MaxIterations', 2000, ...
                'StepTolerance', 1e-8, ...
                'FunctionTolerance', 1e-8);
        catch
            opt = optimset( ...
                'Display', 'off', ...
                'MaxIter', 2000, ...
                'TolX', 1e-8, ...
                'TolFun', 1e-8);
        end
    end

    function opt = makeLsqnonlinOptions()
        try
            opt = optimoptions('lsqnonlin', ...
                'Display', 'none', ...
                'MaxIterations', 500, ...
                'StepTolerance', 1e-8, ...
                'FunctionTolerance', 1e-8, ...
                'OptimalityTolerance', 1e-8);
        catch
            opt = optimset( ...
                'Display', 'off', ...
                'MaxIter', 500, ...
                'TolX', 1e-8, ...
                'TolFun', 1e-8);
        end
    end
end
```

Minimal setup call:

```matlab
pricers = obj.buildPricersG2PP();

surfacePack = obj.buildVolSurfacePackXG2PP(0);

xg2ppState = obj.initCalibXG2PP(surfacePack);
```

## Response:
5/15/2026, 12:33:37 PM

Agreed. Better convention for this library:

$$
\boxed{
\text{one method = one readable responsibility}
}
$$

No nested utilities. Light validation only where failure would be silent or hard to debug.

Use this corrected setup style.

```matlab
function surfacePack = buildVolSurfacePackXG2PP(obj, calibMoneyness, args)
    % G2++ Setup Helper: Extracts and formats one swaption volatility
    % surface for the XG2++ calibration pipeline.

    arguments
        obj (1,1) GaussianTwoFactors;
        calibMoneyness double = 0;
        args.dimension (1,:) char = 'moneyness';
        args.weights = [];
        args.strikes = [];
    end

    if isempty(calibMoneyness)
        calibMoneyness = 0;
    end

    pricingDateNum = obj.getPricingDateNum(obj.pricingDate);

    rawSurface = obj.swaptionVolCube.extractSurface( ...
        'dimension', args.dimension, ...
        'value', calibMoneyness);

    volMatrix = rawSurface.vol;

    expiryDates = rawSurface.expiriesDates(:);
    tenors = rawSurface.tenors(:);

    startSwapDates = rawSurface.startSwapDates;
    matSwapDates = rawSurface.matSwapDates;

    nbExpiries = numel(expiryDates);
    nbTenors = numel(tenors);

    if size(volMatrix,1) ~= nbExpiries || size(volMatrix,2) ~= nbTenors
        error("Inconsistent volatility surface dimensions.");
    end

    expiryDatesYr = yearfracExtend(pricingDateNum, expiryDates, obj.basis);
    expiryDatesYr = expiryDatesYr(:);

    startSwapDatesYr = yearfracExtend(pricingDateNum, startSwapDates, obj.basis);
    matSwapDatesYr = yearfracExtend(pricingDateNum, matSwapDates, obj.basis);

    tenorsYr = obj.parseTenorsXG2PP(tenors);
    tenorsYr = tenorsYr(:);

    validMask = isfinite(volMatrix);
    [quoteRows, quoteCols] = find(validMask);

    if isempty(args.weights)
        weights = ones(size(volMatrix));
    else
        weights = args.weights;
    end

    surfacePack = struct();

    surfacePack.moneyness = rawSurface.moneyness;
    surfacePack.volMatrix = volMatrix;

    surfacePack.expiryDates = expiryDates;
    surfacePack.expiryDatesYr = expiryDatesYr;

    surfacePack.tenors = tenors;
    surfacePack.tenorsYr = tenorsYr;

    surfacePack.startSwapDates = startSwapDates;
    surfacePack.startSwapDatesYr = startSwapDatesYr;

    surfacePack.matSwapDates = matSwapDates;
    surfacePack.matSwapDatesYr = matSwapDatesYr;

    surfacePack.weights = weights;
    surfacePack.strikes = args.strikes;

    surfacePack.validMask = validMask;
    surfacePack.quoteRows = quoteRows;
    surfacePack.quoteCols = quoteCols;
    surfacePack.quoteValues = volMatrix(validMask);

    surfacePack.nbExpiries = nbExpiries;
    surfacePack.nbTenors = nbTenors;
    surfacePack.nbQuotes = nnz(validMask);

    surfacePack.pricingDateNum = pricingDateNum;
    surfacePack.basis = obj.basis;
end
```

This assumes `rawSurface` follows the production `SwaptionCube.extractSurface` convention. That is better than defensive generic accessors for now.

Add this as a separate helper method, not nested:

```matlab
function tenorsYr = parseTenorsXG2PP(obj, tenors)
    % G2++ Setup Helper: Converts tenor labels into year fractions.
    %#ok<INUSD>

    if isnumeric(tenors)
        tenorsYr = tenors(:);
        return;
    end

    if isstring(tenors)
        tenors = cellstr(tenors);
    end

    tenors = tenors(:);
    tenorsYr = NaN(numel(tenors), 1);

    for k = 1:numel(tenors)

        txt = strtrim(char(tenors{k}));

        token = regexp(txt, '^([0-9]*\.?[0-9]+)\s*([DdWwMmYy]?)$', ...
            'tokens', 'once');

        if isempty(token)
            error("Unsupported tenor format: %s", txt);
        end

        value = str2double(token{1});
        unit = upper(token{2});

        switch unit
            case {'', 'Y'}
                tenorsYr(k) = value;
            case 'M'
                tenorsYr(k) = value / 12;
            case 'W'
                tenorsYr(k) = value / 52;
            case 'D'
                tenorsYr(k) = value / 365;
            otherwise
                error("Unsupported tenor unit: %s", unit);
        end
    end
end
```

And the cleaner state initializer:

```matlab
function xg2ppState = initCalibXG2PP(obj, surfacePack, args)
    % G2++ Setup Helper: Initializes the XG2++ calibration control state.
    % This method prepares controls only. It does not calibrate or mutate
    % the model volatility state.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;

        args.mode (1,:) char = 'scaledSurface';

        args.shortTenorIdx (1,1) double = NaN;
        args.longTenorIdx  (1,1) double = NaN;

        args.initialA (1,1) double = NaN;
        args.initialB (1,1) double = NaN;
        args.initialRho (1,1) double = -0.50;
        args.initialAlphaSigma (1,1) double = 1.00;
        args.initialAlphaEta   (1,1) double = 1.00;

        args.aLower (1,1) double = 1e-4;
        args.aUpper (1,1) double = 5.0;
        args.bLower (1,1) double = 1e-4;
        args.bUpper (1,1) double = 5.0;

        args.rhoLower (1,1) double = -0.999;
        args.rhoUpper (1,1) double =  0.999;

        args.alphaSigmaLower (1,1) double = 1e-4;
        args.alphaSigmaUpper (1,1) double = 10.0;
        args.alphaEtaLower   (1,1) double = 1e-4;
        args.alphaEtaUpper   (1,1) double = 10.0;

        args.globalG1PPOptions = [];
        args.sliceBootstrapOptions = [];
        args.scaledG2PPOptions = [];

        args.alreadyComputedSigmaShape (:,1) double = [];
        args.alreadyComputedEtaShape   (:,1) double = [];
        args.previousTheta (:,1) double = [];

        args.terminalPolicy (1,:) char = 'flatLast';
        args.localMode (1,:) char = 'scaled';
    end

    tenorsYr = surfacePack.tenorsYr(:);
    validMask = surfacePack.validMask;

    validTenorCols = find(any(validMask, 1));

    if isempty(validTenorCols)
        error("No tenor column contains finite market quotes.");
    end

    shortIdx = args.shortTenorIdx;
    longIdx = args.longTenorIdx;

    if isnan(shortIdx)
        shortIdx = validTenorCols(1);
    end

    if isnan(longIdx)
        longIdx = validTenorCols(end);
    end

    initialA = args.initialA;
    initialB = args.initialB;

    if isnan(initialA)
        initialA = obj.meanReversion_x;
    end

    if isnan(initialB)
        initialB = obj.meanReversion_y;
    end

    thetaNames = ["a"; "b"; "rho"; "alphaSigma"; "alphaEta"];

    theta0 = [
        initialA
        initialB
        args.initialRho
        args.initialAlphaSigma
        args.initialAlphaEta
    ];

    lb = [
        args.aLower
        args.bLower
        args.rhoLower
        args.alphaSigmaLower
        args.alphaEtaLower
    ];

    ub = [
        args.aUpper
        args.bUpper
        args.rhoUpper
        args.alphaSigmaUpper
        args.alphaEtaUpper
    ];

    xg2ppState = struct();

    xg2ppState.mode = args.mode;
    xg2ppState.localMode = args.localMode;
    xg2ppState.terminalPolicy = args.terminalPolicy;

    xg2ppState.shortTenorIdx = shortIdx;
    xg2ppState.longTenorIdx = longIdx;
    xg2ppState.shortTenor = tenorsYr(shortIdx);
    xg2ppState.longTenor = tenorsYr(longIdx);

    xg2ppState.thetaNames = thetaNames;
    xg2ppState.theta0 = theta0;

    xg2ppState.initial = struct();
    xg2ppState.initial.a = initialA;
    xg2ppState.initial.b = initialB;
    xg2ppState.initial.rho = args.initialRho;
    xg2ppState.initial.alphaSigma = args.initialAlphaSigma;
    xg2ppState.initial.alphaEta = args.initialAlphaEta;

    xg2ppState.bounds = struct();
    xg2ppState.bounds.lower = lb;
    xg2ppState.bounds.upper = ub;

    xg2ppState.options = struct();
    xg2ppState.options.globalG1PP = args.globalG1PPOptions;
    xg2ppState.options.sliceBootstrap = args.sliceBootstrapOptions;
    xg2ppState.options.scaledG2PP = args.scaledG2PPOptions;

    xg2ppState.previous = struct();
    xg2ppState.previous.sigmaShape = args.alreadyComputedSigmaShape(:);
    xg2ppState.previous.etaShape = args.alreadyComputedEtaShape(:);
    xg2ppState.previous.theta = args.previousTheta(:);

    xg2ppState.surfaceSize = size(surfacePack.volMatrix);
    xg2ppState.nbQuotes = surfacePack.nbQuotes;
end
```

Main correction: `initCalibXG2PP` now **does not create solver options**. That should be a separate helper later, e.g.

```matlab
buildSolverOptionsXG2PP
```

This keeps setup readable and avoids hiding MATLAB optimizer choices inside a state initializer.

## Response:
5/15/2026, 12:39:47 PM

Good. This stage is the **shape generator**.

It should not touch G2++ yet, except for using the same market surface and the same production market-price adapters.

```text
ONE-FACTOR INITIALIZATION
│
├─ calibrateGlobalG1PP
│     → full surface constant G1++ fit
│     → aG1
│     → sigmaG1
│     → global residual diagnostics
│
├─ bootstrapSliceG1PP
│     → selected tenor column = short tenor
│     → fixed / inherited aG1
│     → piecewise σ-shape
│     → short-slice diagnostics
│
└─ bootstrapSliceG1PP
      → selected tenor column = long tenor
      → fixed / inherited aG1
      → piecewise η-shape
      → long-slice diagnostics
```

# 1. Main design choice

For sobriety, I would make `bootstrapSliceG1PP` generic:

```matlab
sigmaShapePack = obj.bootstrapSliceG1PP(surfacePack, xg2ppState, pricers, ...
    'tenorIdx', xg2ppState.shortTenorIdx, ...
    'shapeName', 'sigma');

etaShapePack = obj.bootstrapSliceG1PP(surfacePack, xg2ppState, pricers, ...
    'tenorIdx', xg2ppState.longTenorIdx, ...
    'shapeName', 'eta');
```

So we do **not** create two different methods for short and long.

---

# 2. Method responsibility map

```text
calibrateGlobalG1PP
│
├─ build full calibration set
│     → all finite quotes in surfacePack
│
├─ objective θ = [a, sigma]
│     → constant-vol one-factor price
│     → market price
│     → residual vector
│
├─ solve global least squares
│
└─ output
      → aG1
      → sigmaG1
      → thetaStar
      → residuals
      → model prices
      → market prices
```

```text
bootstrapSliceG1PP
│
├─ select one tenor column
│     → short or long tenor
│
├─ loop over expiries k = 1,...,n
│     │
│     ├─ build calibration point
│     │     → IRS
│     │     → strike
│     │     → direction
│     │     → swaptionPack
│     │     → marketPrice
│     │
│     ├─ solve local one-factor volatility σ_k
│     │     → previous buckets frozen
│     │     → current bucket optimized
│     │
│     └─ store diagnostics
│
└─ output
      → shape vector
      → time structure
      → prices
      → residuals
      → solver info
```

# 3. Required lower-level helpers

Before shipping code, the one-factor block needs a few numerical helpers. Keep them separate.

```text
G1PP NUMERICAL HELPERS
│
├─ priceSwaptionG1PP
│     → one-factor HW/G1++ swaption price
│     → accepts explicit volatility vector and time structure
│
├─ buildCalibrationPointFromSurface
│     → one surface quote
│     → IRS + swaptionPack + marketPrice
│
├─ residualGlobalG1PP
│     → vector residual over all finite quotes
│
└─ residualBucketG1PP
      → scalar residual for one slice bucket
```

The key risk: do not reuse the existing HW1F calibration too blindly unless we know its function accepts explicit volatility vector without mutating unrelated state. In your current architecture, it is safer to build a thin G1++ pricing helper that uses the same closed-form HW1F/G1++ logic but explicit inputs.

---

# 4. Data contracts

## `g1Global`

```matlab
g1Global.a
g1Global.sigma
g1Global.thetaStar
g1Global.residuals
g1Global.modelPrices
g1Global.marketPrices
g1Global.solverInfo
```

## `shapePack`

```matlab
shapePack.name              % 'sigma' or 'eta'
shapePack.tenorIdx
shapePack.tenor
shapePack.timeStructure
shapePack.timeStructureDt
shapePack.shape             % piecewise volatility vector
shapePack.modelPrices
shapePack.marketPrices
shapePack.residuals
shapePack.strikes
shapePack.directions
shapePack.solverInfo
```

# 5. Clean architecture version

```text
ONE-FACTOR INITIALIZATION
│
├─ calibrateGlobalG1PP
│   │
│   ├─ buildCalibrationPointFromSurface
│   │     → for all finite surface quotes
│   │
│   ├─ residualGlobalG1PP
│   │     θ = [a, sigma]
│   │       ↓
│   │     constant G1++ volatility state
│   │       ↓
│   │     price all one-factor swaptions
│   │       ↓
│   │     normalized price residuals
│   │
│   └─ solve
│         → g1Global.a
│         → g1Global.sigma
│         → diagnostics
│
├─ bootstrapSliceG1PP  [short tenor]
│   │
│   ├─ select quote column
│   ├─ buildCalibrationPointFromSurface
│   ├─ residualBucketG1PP
│   ├─ solve σ_k
│   └─ return sigmaShapePack
│
└─ bootstrapSliceG1PP  [long tenor]
    │
    ├─ select quote column
    ├─ buildCalibrationPointFromSurface
    ├─ residualBucketG1PP
    ├─ solve σ_k
    └─ return etaShapePack
```

# 6. Decision needed before code

The next code depends on one convention:

$$
\boxed{
\text{Do we implement } \texttt{priceSwaptionG1PP} \text{ ourselves, or wrap existing HW1F pricing?}
}
$$

My recommendation: wrap the existing HW1F formula **only if** it can accept explicit vol vector and time structure. Otherwise write a small G1PP pricing method with explicit inputs, because this block is only a shape generator and should not depend on HW1F object state too much.

## Response:
5/15/2026, 1:51:57 PM

Yes. Clean answer:

$$
\boxed{
\text{Do not wrap HW1F. Implement G1++ as a degenerate mode of your G2++ class.}
}
$$

# 1. G1++ inside `GaussianTwoFactors`

Your class already has:

```matlab
meanReversion_x
meanReversion_y
volatility_sigma
volatility_eta
correlation
```

So a one-factor Gaussian model can be represented as:

$$
r_t = x_t + \phi(t),
\qquad
dx_t = -a x_t dt + \sigma(t)dW_t.
$$

Inside `GaussianTwoFactors`, this is obtained by freezing the second factor:

$$
\eta(t)=0.
$$

or symmetrically:

$$
\sigma(t)=0.
$$

For the XG2++ initialization, use:

```matlab
G1PP-x mode:
    meanReversion_x = a
    volatility_sigma = sigmaShape
    volatility_eta = 0
```

and for the second slice, either reuse the same `x` machinery or use:

```matlab
G1PP-y mode:
    meanReversion_y = b
    volatility_eta = etaShape
    volatility_sigma = 0
```

But mathematically, for one-factor initialization, the cleanest is:

$$
\boxed{
\text{use one generic G1++ pricing helper with } a,\sigma(t)
}
$$

not the full two-factor pricer.

---

# 2. Why not set both G2++ factors equal?

If you set:

$$
a=b,\qquad \sigma(t)=\eta(t),
$$

then G2++ is not automatically G1++.

You get:

$$
r_t=x_t+y_t+\phi(t).
$$

If both OU factors have the same mean reversion, then their sum is still one Gaussian factor:

$$
z_t=x_t+y_t,
$$

but its effective volatility is:

$$
\sigma_z(t)^2
=
\sigma(t)^2+\eta(t)^2+2\rho\sigma(t)\eta(t).
$$

So the model collapses to G1++ only if you treat:

$$
z_t=x_t+y_t
$$

as the single factor with volatility:

$$
\boxed{
\sigma_{\mathrm{eff}}(t)
=
\sqrt{\sigma(t)^2+\eta(t)^2+2\rho\sigma(t)\eta(t)}
}
$$

That is not the same as “same mean reversion for both vectors.”

---

# 3. Consistency expectation

If:

$$
a=b
$$

then G2++ has a **one-factor-equivalent short-rate variance**, but the decomposition between $x$ and $y$ is not identifiable.

So calibration becomes structurally weak:

$$
(\sigma,\eta,\rho)
\quad \mapsto \quad
\sigma_{\mathrm{eff}}
$$

many parameter triples give the same one-factor variance.

Therefore:

$$
\boxed{
a=b \text{ is useful for testing reduction, not for final G2++ calibration.}
}
$$

---

# 4. Practical convention for us

For XG2++ initialization:

```text
Global G1++ calibration
    → estimate a_G1 and sigma_G1

Short-tenor G1++ slice
    → use a_G1
    → bootstrap sigmaShape

Long-tenor G1++ slice
    → use a_G1 or b seed
    → bootstrap etaShape
```

Then initialize G2++ as:

$$
a_0 = a_{G1},
\qquad
b_0 \neq a_0 \text{ or weakly separated},
\qquad
\rho_0 < 0.
$$

Example:

```matlab
a0 = g1Global.a;
b0 = max(0.05, 0.25 * g1Global.a);
rho0 = -0.50;
```

Not final truth. Just a stable starting geometry.

# 5. Bottom line

Use G1++ as its own one-factor formula inside `GaussianTwoFactors`.

Do **not** simulate it by setting both G2++ factors equal.

Use the degenerate G2++ logic only for tests:

$$
\eta=0 \Rightarrow G2++ \to G1++.
$$

That is the safe consistency check.

## Response:
5/15/2026, 1:54:28 PM

> **Focusing on concise code structure and methods**
>
> I need to come up with code but keep it concise and relevant — no external web searches here. The user asked for a substantial code answer, but I should avoid unnecessary citations, even if files are involved. I’ll probably need to provide methods like `calibrateGlobalG1PP`, `residualGlobalG1PP`, and others, relying on earlier helpers. I'll keep it simple and use class methods where I can.
>
> Thought for 44s

I’ll ship this as a first working internal block, not a final library abstraction. I’ll keep each method flat, explicit, and dependent on the setup helpers already agreed.

Paste these inside `methods` of `GaussianTwoFactors`. This block assumes your setup helpers already exist:

```matlab
buildPricersG2PP
buildVolSurfacePackXG2PP
initCalibXG2PP
getPricingDateNum
parseTenorsXG2PP
```

It also assumes `getInitialMarketZC`, `getH`, `getExpDiff` already exist.

```matlab
function g1Global = calibrateGlobalG1PP(obj, surfacePack, pricers, args)
    % G2++ Calibration Helper: Calibrates a constant-volatility G1++ model
    % to the full swaption surface. This produces the one-factor starting
    % point used before the two slice bootstraps.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';

        args.initialA (1,1) double = NaN;
        args.initialSigma (1,1) double = 0.01;

        args.useLsqnonlin (1,1) logical = true;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    if isnan(args.initialA)
        a0 = obj.meanReversion_x;
    else
        a0 = args.initialA;
    end

    theta0 = log([a0; args.initialSigma]);

    calibSet = struct([]);
    k = 0;

    for i = 1:surfacePack.nbExpiries
        for j = 1:surfacePack.nbTenors

            if ~surfacePack.validMask(i,j)
                continue
            end

            k = k + 1;

            calibSet(k) = obj.buildCalibrationPointFromSurface( ...
                surfacePack, pricers, i, j, ...
                'irs', args.irs, ...
                'notional', args.notional, ...
                'settleType', args.settleType);
        end
    end

    residualFun = @(theta) obj.residualGlobalG1PP( ...
        theta, calibSet, surfacePack, ...
        'objectiveType', args.objectiveType);

    if args.useLsqnonlin && exist('lsqnonlin', 'file') == 2
        options = optimoptions('lsqnonlin', ...
            'Display', 'none', ...
            'MaxIterations', 300, ...
            'FunctionTolerance', 1e-8, ...
            'StepTolerance', 1e-8);

        [thetaStar, resnorm, residuals, exitflag, output] = ...
            lsqnonlin(residualFun, theta0, [], [], options);
    else
        options = optimset('Display', 'off', 'MaxIter', 300);
        objFun = @(theta) sum(residualFun(theta).^2);

        [thetaStar, resnorm, exitflag, output] = ...
            fminsearch(objFun, theta0, options);

        residuals = residualFun(thetaStar);
    end

    params = exp(thetaStar);

    g1Global = struct();
    g1Global.a = params(1);
    g1Global.sigma = params(2);
    g1Global.thetaStar = thetaStar;
    g1Global.calibSet = calibSet;
    g1Global.residuals = residuals;
    g1Global.resnorm = resnorm;
    g1Global.exitflag = exitflag;
    g1Global.output = output;
    g1Global.rmse = sqrt(mean(residuals.^2));
    g1Global.maxAbsResidual = max(abs(residuals));
end
```

```matlab
function residuals = residualGlobalG1PP(obj, thetaRaw, calibSet, surfacePack, args)
    % G2++ Calibration Helper: Full-surface residual for constant G1++.

    arguments
        obj (1,1) GaussianTwoFactors;
        thetaRaw (:,1) double;
        calibSet (1,:) struct;
        surfacePack (1,1) struct;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    params = exp(thetaRaw(:));
    a = params(1);
    sigma = params(2);

    maxMat = max(surfacePack.matSwapDatesYr(:));
    volTimeStructure = maxMat + 1.0;
    volVector = sigma .* ones(2,1);

    residuals = NaN(numel(calibSet), 1);

    for k = 1:numel(calibSet)

        point = calibSet(k);

        modelPrice = obj.priceSwaptionG1PP( ...
            point.strike, ...
            point.expiry, ...
            point.paymentDates, ...
            a, ...
            volVector, ...
            volTimeStructure, ...
            'direction', point.direction, ...
            'notional', point.notional);

        switch args.objectiveType
            case 'price'
                scale = 1.0;
            otherwise
                scale = max(point.notional * point.annuity, 1e-12);
        end

        residuals(k) = sqrt(point.weight) * ...
            (modelPrice - point.marketPrice) / scale;
    end
end
```

```matlab
function shapePack = bootstrapSliceG1PP(obj, surfacePack, pricers, g1Global, args)
    % G2++ Calibration Helper: Bootstraps one G1++ piecewise volatility
    % slice. Use once for the short-tenor sigma shape and once for the
    % long-tenor eta shape.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;
        g1Global (1,1) struct;

        args.tenorIdx (1,1) double;
        args.shapeName (1,:) char = 'sigma';

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';

        args.initialVol (1,1) double = NaN;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    tenorIdx = args.tenorIdx;

    quoteRows = find(surfacePack.validMask(:, tenorIdx));
    nbSteps = numel(quoteRows);

    if isnan(args.initialVol)
        initialVol = g1Global.sigma;
    else
        initialVol = args.initialVol;
    end

    shape = NaN(nbSteps, 1);
    modelPrices = NaN(nbSteps, 1);
    marketPrices = NaN(nbSteps, 1);
    residuals = NaN(nbSteps, 1);
    strikes = NaN(nbSteps, 1);
    directions = strings(nbSteps, 1);

    timeStructure = surfacePack.expiryDatesYr(quoteRows);
    timeStructure = timeStructure(:);

    for k = 1:nbSteps

        row = quoteRows(k);

        point = obj.buildCalibrationPointFromSurface( ...
            surfacePack, pricers, row, tenorIdx, ...
            'irs', args.irs, ...
            'notional', args.notional, ...
            'settleType', args.settleType);

        if k == 1
            guess = initialVol;
        else
            guess = shape(k-1);
        end

        resFun = @(v) obj.residualBucketG1PP( ...
            v, k, shape, timeStructure, point, g1Global.a, ...
            'objectiveType', args.objectiveType);

        try
            volK = fzero(resFun, [1e-5, 1.0]);
        catch
            volK = fminsearch(@(v) resFun(abs(v)).^2, guess, ...
                optimset('Display', 'off', 'MaxIter', 100));
            volK = abs(volK);
        end

        shape(k) = volK;

        volVector = [shape(1:k); shape(k)];
        volTimeStructure = timeStructure(1:k);

        modelPrice = obj.priceSwaptionG1PP( ...
            point.strike, ...
            point.expiry, ...
            point.paymentDates, ...
            g1Global.a, ...
            volVector, ...
            volTimeStructure, ...
            'direction', point.direction, ...
            'notional', point.notional);

        modelPrices(k) = modelPrice;
        marketPrices(k) = point.marketPrice;
        residuals(k) = modelPrice - point.marketPrice;
        strikes(k) = point.strike;
        directions(k) = string(point.direction);
    end

    shapePack = struct();
    shapePack.name = args.shapeName;
    shapePack.tenorIdx = tenorIdx;
    shapePack.tenor = surfacePack.tenorsYr(tenorIdx);

    shapePack.timeStructure = timeStructure;
    shapePack.timeStructureDt = timeStructure;
    shapePack.shape = shape;

    shapePack.modelPrices = modelPrices;
    shapePack.marketPrices = marketPrices;
    shapePack.residuals = residuals;
    shapePack.strikes = strikes;
    shapePack.directions = directions;

    shapePack.g1MeanReversion = g1Global.a;
    shapePack.quoteRows = quoteRows;
end
```

```matlab
function residual = residualBucketG1PP(obj, volCandidate, step, shape, timeStructure, point, a, args)
    % G2++ Calibration Helper: Scalar residual for one G1++ bootstrap bucket.

    arguments
        obj (1,1) GaussianTwoFactors;
        volCandidate (1,1) double;
        step (1,1) double;
        shape (:,1) double;
        timeStructure (:,1) double;
        point (1,1) struct;
        a (1,1) double;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    volCandidate = abs(volCandidate);

    volSlice = shape(1:step);
    volSlice(step) = volCandidate;

    volVector = [volSlice; volCandidate];
    volTimeStructure = timeStructure(1:step);

    modelPrice = obj.priceSwaptionG1PP( ...
        point.strike, ...
        point.expiry, ...
        point.paymentDates, ...
        a, ...
        volVector, ...
        volTimeStructure, ...
        'direction', point.direction, ...
        'notional', point.notional);

    switch args.objectiveType
        case 'price'
            scale = 1.0;
        otherwise
            scale = max(point.notional * point.annuity, 1e-12);
    end

    residual = (modelPrice - point.marketPrice) / scale;
end
```

```matlab
function point = buildCalibrationPointFromSurface(obj, surfacePack, pricers, row, col, args)
    % G2++ Calibration Helper: Builds one market calibration point from the
    % normalized volatility surface.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;
        row (1,1) double;
        col (1,1) double;

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';
    end

    swapStartDate = surfacePack.startSwapDates(row, col);
    matSwapDate = surfacePack.matSwapDates(row, col);
    exerciseDate = surfacePack.expiryDates(row);
    marketVol = surfacePack.volMatrix(row, col);

    irs = copy(args.irs(1));
    irs.setNotional(args.notional);
    irs.setDates(swapStartDate, matSwapDate);

    [npv, ~, ~] = pricers.swap.IRSVanille(irs, ...
        'measures', true, ...
        'details', true);

    if isempty(surfacePack.strikes)
        strike = npv.FairSwapRate;
    else
        strike = surfacePack.strikes(row, col);
    end

    direction = obj.swaptionDirection(strike, npv.FairSwapRate);
    irs.direction = direction;

    swaption = Swaption(irs, ...
        'exerciseDate', exerciseDate, ...
        'strike', strike, ...
        'settleType', args.settleType);

    [~, fixedLegPDates, ~] = irs.fixedLeg01(swapStartDate);

    fixedLegPDates = fixedLegPDates(:).';

    paymentDates = yearfracExtend( ...
        surfacePack.pricingDateNum, ...
        fixedLegPDates(2:end), ...
        obj.basis);

    paymentDates = paymentDates(:).';

    tenorsDt = yearfracExtend( ...
        fixedLegPDates(1:end-1), ...
        fixedLegPDates(2:end), ...
        swaption.irs.fixedLeg.interestDCC);

    tenorsDt = tenorsDt(:).';

    [marketPrice, ~, annuity, bachelierVol] = pricers.bachelier.Swaption( ...
        swaption, ...
        'measures', false, ...
        'vol', marketVol, ...
        'irsRes', npv);

    point = struct();
    point.row = row;
    point.col = col;

    point.irs = irs;
    point.swaption = swaption;
    point.npv = npv;

    point.expiryDate = exerciseDate;
    point.expiry = surfacePack.expiryDatesYr(row);
    point.swapStartDate = swapStartDate;
    point.matSwapDate = matSwapDate;

    point.paymentDates = paymentDates;
    point.tenorsDt = tenorsDt;
    point.fixedLegPDates = fixedLegPDates;

    point.strike = strike;
    point.direction = direction;
    point.notional = args.notional;

    point.marketVol = marketVol;
    point.bachelierVol = bachelierVol;
    point.marketPrice = marketPrice;
    point.annuity = annuity;

    point.weight = surfacePack.weights(row, col);
end
```

Now the G1++ pricing kernel.

```matlab
function price = priceSwaptionG1PP(obj, strike, expiry, paymentDates, a, volVector, volTimeStructure, args)
    % G2++ Calibration Helper: Prices a European swaption under one-factor
    % Gaussian G1++ using the same bond-basket convention as the G2++ pricer.

    arguments
        obj (1,1) GaussianTwoFactors;
        strike (1,1) double;
        expiry (1,1) double;
        paymentDates (1,:) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;

        args.direction (1,:) char {mustBeMember(args.direction, {'PAY','REC'})} = 'PAY';
        args.notional (1,1) double = 1e6;
    end

    paymentDates = paymentDates(:);

    basket = obj.getBondBasketG1PP(expiry, paymentDates, a, volVector, volTimeStructure);

    delta = diff([expiry; paymentDates]);

    coeffs = strike .* delta;
    coeffs(end) = coeffs(end) + 1.0;

    lambda = coeffs .* basket.A0Ti(:);
    B = basket.B0Ti(:);

    xBar = obj.solveXBarG1PP(lambda, B);

    gaussian = obj.getStateGaussianG1PP(expiry, a, volVector, volTimeStructure);

    muX = gaussian.muX;
    varX = gaussian.varX;
    sigX = sqrt(max(varX, 0));

    if sigX <= 0
        basketValue = sum(lambda .* exp(-B .* muX));

        if strcmpi(args.direction, 'PAY')
            payoff = max(1 - basketValue, 0);
        else
            payoff = max(basketValue - 1, 0);
        end

        price = args.notional .* basket.P0T0 .* payoff;
        return
    end

    h0 = (xBar - muX) ./ sigX;
    h1 = h0 + B .* sigX;
    kappa = -B .* muX + 0.5 .* (B.^2) .* varX;

    if strcmpi(args.direction, 'PAY')
        expectation = normcdf(-h0) - sum(lambda .* exp(kappa) .* normcdf(-h1));
    else
        expectation = sum(lambda .* exp(kappa) .* normcdf(h1)) - normcdf(h0);
    end

    price = args.notional .* basket.P0T0 .* max(expectation, 0);
end
```

```matlab
function basket = getBondBasketG1PP(obj, expiry, paymentDates, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Builds the G1++ affine bond basket.

    arguments
        obj (1,1) GaussianTwoFactors;
        expiry (1,1) double;
        paymentDates (:,1) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    paymentDates = paymentDates(:);

    zc = obj.getInitialMarketZC([expiry, paymentDates.']);
    zc = zc(:);

    P0T0 = zc(1);
    P0Ti = zc(2:end);

    dt = paymentDates - expiry;
    B0Ti = obj.getH(a, dt);

    var0T0 = obj.varianceIntegralG1PP(0, expiry, a, volVector, volTimeStructure);

    varT0Ti = zeros(numel(paymentDates), 1);
    var0Ti = zeros(numel(paymentDates), 1);

    for i = 1:numel(paymentDates)
        Ti = paymentDates(i);

        varT0Ti(i) = obj.varianceIntegralG1PP( ...
            expiry, Ti, a, volVector, volTimeStructure);

        var0Ti(i) = obj.varianceIntegralG1PP( ...
            0, Ti, a, volVector, volTimeStructure);
    end

    A0Ti = (P0Ti ./ P0T0) .* exp(0.5 .* (varT0Ti - var0Ti + var0T0));

    basket = struct();
    basket.P0T0 = P0T0;
    basket.P0Ti = P0Ti;
    basket.A0Ti = A0Ti;
    basket.B0Ti = B0Ti(:);
end
```

```matlab
function gaussian = getStateGaussianG1PP(obj, expiry, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Computes X_T law under the T-forward measure
    % for one-factor Gaussian initialization.

    arguments
        obj (1,1) GaussianTwoFactors;
        expiry (1,1) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    muX = obj.meanForwardG1PP(0, expiry, a, volVector, volTimeStructure);
    varX = obj.varianceStateG1PP(0, expiry, a, volVector, volTimeStructure);

    gaussian = struct();
    gaussian.muX = muX;
    gaussian.varX = varX;
end
```

```matlab
function xBar = solveXBarG1PP(obj, lambda, B)
    % G2++ Calibration Helper: Solves the one-factor Jamshidian boundary.

    arguments
        obj (1,1) GaussianTwoFactors;
        lambda (:,1) double;
        B (:,1) double;
    end

    f = @(x) 1 - sum(lambda .* exp(-B .* x));

    lo = -0.10;
    hi =  0.10;

    flo = f(lo);
    fhi = f(hi);

    k = 0;

    while flo * fhi > 0 && k < 40
        width = hi - lo;
        lo = lo - 2 * width;
        hi = hi + 2 * width;

        flo = f(lo);
        fhi = f(hi);

        k = k + 1;
    end

    if flo * fhi > 0
        xBar = NaN;
        return
    end

    xBar = fzero(f, [lo, hi]);
end
```

```matlab
function output = varianceStateG1PP(obj, startDate, endDate, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Variance of X(endDate) conditional on startDate.

    arguments
        obj (1,1) GaussianTwoFactors;
        startDate (1,1) double;
        endDate (1,1) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    [bucketLeft, bucketRight] = obj.getBucketBoundsG1PP(volTimeStructure);

    idx = find((bucketRight > startDate) & (bucketLeft < endDate));

    localLeft = max(startDate, bucketLeft(idx));
    localRight = min(endDate, bucketRight(idx));

    deltaLeft = endDate - localLeft(:);
    deltaRight = endDate - localRight(:);

    vol = volVector(idx);
    vol = vol(:);

    exp2A = obj.getExpDiff(2*a, deltaLeft, deltaRight);

    output = sum((vol.^2 ./ (2*a)) .* exp2A);
end
```

```matlab
function output = meanForwardG1PP(obj, startDate, endDate, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Mean of X(endDate) under the endDate-forward
    % measure in the one-factor Gaussian model.

    arguments
        obj (1,1) GaussianTwoFactors;
        startDate (1,1) double;
        endDate (1,1) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    [bucketLeft, bucketRight] = obj.getBucketBoundsG1PP(volTimeStructure);

    idx = find((bucketRight > startDate) & (bucketLeft < endDate));

    localLeft = max(startDate, bucketLeft(idx));
    localRight = min(endDate, bucketRight(idx));

    deltaLeft = endDate - localLeft(:);
    deltaRight = endDate - localRight(:);

    vol = volVector(idx);
    vol = vol(:);

    expA = obj.getExpDiff(a, deltaLeft, deltaRight);
    exp2A = obj.getExpDiff(2*a, deltaLeft, deltaRight);

    output = sum(-(vol.^2 ./ a.^2) .* expA ...
                 + (vol.^2 ./ (2*a.^2)) .* exp2A);
end
```

```matlab
function output = varianceIntegralG1PP(obj, startDate, endDate, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Variance of integral_startDate^endDate X(u)du
    % under piecewise-constant one-factor volatility.

    arguments
        obj (1,1) GaussianTwoFactors;
        startDate (1,1) double;
        endDate (1,1) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    [bucketLeft, bucketRight] = obj.getBucketBoundsG1PP(volTimeStructure);

    idx = find((bucketRight > startDate) & (bucketLeft < endDate));

    localLeft = max(startDate, bucketLeft(idx));
    localRight = min(endDate, bucketRight(idx));

    deltaLeft = endDate - localLeft(:);
    deltaRight = endDate - localRight(:);

    vol = volVector(idx);
    vol = vol(:);

    deltaInterval = deltaLeft - deltaRight;

    expA = obj.getExpDiff(a, deltaLeft, deltaRight);
    exp2A = obj.getExpDiff(2*a, deltaLeft, deltaRight);

    output = sum((vol.^2 ./ a.^2) .* ...
        (deltaInterval - (2./a).*expA + (1./(2*a)).*exp2A));
end
```

```matlab
function [bucketLeft, bucketRight] = getBucketBoundsG1PP(obj, volTimeStructure)
    % G2++ Calibration Helper: Bucket bounds for explicit G1++ volatility
    % vectors.
    %#ok<INUSD>

    timeSteps = volTimeStructure(:).';

    bucketLeft = [0, timeSteps];
    bucketRight = [timeSteps, Inf];

    bucketLeft = bucketLeft(:);
    bucketRight = bucketRight(:);
end
```

Typical use:

```matlab
g1Global = obj.calibrateGlobalG1PP(surfacePack, pricers);

sigmaShapePack = obj.bootstrapSliceG1PP( ...
    surfacePack, pricers, g1Global, ...
    'tenorIdx', xg2ppState.shortTenorIdx, ...
    'shapeName', 'sigma');

etaShapePack = obj.bootstrapSliceG1PP( ...
    surfacePack, pricers, g1Global, ...
    'tenorIdx', xg2ppState.longTenorIdx, ...
    'shapeName', 'eta');
```

## Response:
5/15/2026, 3:33:00 PM

> **Reworking architecture without citations**
>
> The user asked for a redo of the architecture, but it's clear they likely don’t need code or citations. The file is uploaded, but there’s really no reason to cite specific class properties unless it directly ties into the architecture. I can focus on reworking the structure without citations or exact references from the file.
>
> Thought for 7s

Object:

$$
\boxed{
\text{ONE-FACTOR INITIALIZATION for XG2++}
}
$$

Mathematical responsibility:

$$
\text{produce } a_{G1},\ \bar\sigma^{short}(t),\ \bar\sigma^{long}(t)
$$

before the reduced G2++ scaling stage.

Software responsibility:

$$
\boxed{
\text{reuse G2++ bucket/variance helpers through a temporary G1++ state wrapper}
}
$$

The wrapper lives only inside:

```matlab
priceSwaptionG1PP
```

not in the residuals, not in the bootstrap loop.

---

```text
ONE-FACTOR INITIALIZATION
│
├─ calibrateGlobalG1PP
│   │
│   ├─ buildCalibrationPointFromSurface
│   │     → loop over all finite surface quotes
│   │     → IRS
│   │     → strike
│   │     → direction
│   │     → swaptionPack / paymentDates
│   │     → marketPrice
│   │
│   ├─ residualGlobalG1PP
│   │     θ = [log(a), log(sigma)]
│   │       ↓
│   │     build constant G1++ vol vector
│   │       ↓
│   │     priceSwaptionG1PP for each calibration point
│   │       ↓
│   │     normalized price residual vector
│   │
│   └─ solve
│         → g1Global.a
│         → g1Global.sigma
│         → g1Global.residuals
│         → g1Global.diagnostics
│
├─ bootstrapSliceG1PP  [short tenor]
│   │
│   ├─ select tenor column
│   │     → xg2ppState.shortTenorIdx
│   │
│   ├─ buildCalibrationPointFromSurface
│   │     → one point per finite expiry quote
│   │
│   ├─ residualBucketG1PP
│   │     σ_k
│   │       ↓
│   │     previous σ_1,...,σ_{k-1} frozen
│   │       ↓
│   │     trial piecewise G1++ vol vector
│   │       ↓
│   │     priceSwaptionG1PP
│   │       ↓
│   │     scalar price residual
│   │
│   ├─ solve local σ_k
│   │
│   └─ return sigmaShapePack
│         → short-tenor volatility shape
│         → prices / residuals / strikes / directions
│
└─ bootstrapSliceG1PP  [long tenor]
    │
    ├─ select tenor column
    │     → xg2ppState.longTenorIdx
    │
    ├─ buildCalibrationPointFromSurface
    │     → one point per finite expiry quote
    │
    ├─ residualBucketG1PP
    │     σ_k
    │       ↓
    │     previous σ_1,...,σ_{k-1} frozen
    │       ↓
    │     trial piecewise G1++ vol vector
    │       ↓
    │     priceSwaptionG1PP
    │       ↓
    │     scalar price residual
    │
    ├─ solve local σ_k
    │
    └─ return etaShapePack
          → long-tenor volatility shape
          → prices / residuals / strikes / directions
```

# Internal pricing architecture

```text
priceSwaptionG1PP
│
├─ setTemporaryG1PPState
│     → save current G2++ state
│     → set meanReversion_x = a
│     → set volatility_sigma = trial G1++ vector
│     → set volatility_eta = zeros
│     → set correlation = 0
│     → set volatilityTimeStructure = trial bucket dates
│
├─ getBondBasketG1PP
│     → uses existing G2++ primitives
│     → getInitialMarketZC
│     → getH
│     → variXOnly
│
├─ getStateGaussianG1PP
│     → uses existing G2++ primitives
│     → varTForward('x_process')
│     → meanForwardG1PP or reduced x-forward mean
│
├─ solveXBarG1PP
│     → Jamshidian boundary
│
├─ priceFromG1PPBasket
│     → closed-form one-factor Gaussian swaption price
│
└─ restoreG1PPState
      → restore original G2++ parameters even if pricing fails
```

# Helper list after architectural cleanup

## Calibration-level helpers

```text
calibrateGlobalG1PP
residualGlobalG1PP
bootstrapSliceG1PP
residualBucketG1PP
buildCalibrationPointFromSurface
```

## G1++ pricing helpers

```text
priceSwaptionG1PP
setTemporaryG1PPState
restoreG1PPState
getBondBasketG1PP
variXOnly
getStateGaussianG1PP
meanForwardG1PP
solveXBarG1PP
priceFromG1PPBasket
```

# What changes versus the previous version

We remove these explicit duplicate numerical helpers:

```text
varianceIntegralG1PP
varianceStateG1PP
getBucketBoundsG1PP
```

because the existing G2++ object already owns that logic through:

```text
getBucketBounds
getPricingIdx
deltaTLeft
deltaTRight
variX
varTForward
getH
getExpDiff
```

The price is that `priceSwaptionG1PP` becomes stateful internally, so it must save and restore object state.

# Final design rule

```text
calibrateGlobalG1PP
bootstrapSliceG1PP
residualGlobalG1PP
residualBucketG1PP
    → do not mutate object state directly.

priceSwaptionG1PP
    → only method allowed to mutate temporary G1++ state.

setTemporaryG1PPState / restoreG1PPState
    → only methods allowed to touch temporary model-state replacement.
```

This keeps the calibration pipeline readable while reusing the already-tested G2++ bucket and variance machinery.

## Response:
5/15/2026, 3:33:43 PM

> **Providing code without citations unless based on file facts**
>
> The user asked for code, so I should focus on providing just that, with minimal distractions. Since the request doesn't involve referencing uploaded files, no citations are needed. I'll stick to creating MATLAB methods with no nested utilities, light validation, and avoid overly lengthy code.
>
> Thought for 32s

Use this version. It keeps the calibration-level helpers clean and puts the temporary G1++ state wrapper only inside `priceSwaptionG1PP`.

Paste inside `methods` of `GaussianTwoFactors`.

```matlab
function g1Global = calibrateGlobalG1PP(obj, surfacePack, pricers, args)
    % G2++ Calibration Helper: Calibrates a constant-volatility G1++ model
    % to the full swaption surface.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';

        args.initialA (1,1) double = NaN;
        args.initialSigma (1,1) double = 0.01;

        args.useLsqnonlin (1,1) logical = true;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    a0 = args.initialA;
    if isnan(a0)
        a0 = obj.meanReversion_x;
    end
    if isempty(a0) || a0 <= 0
        a0 = 0.10;
    end

    theta0 = log([a0; args.initialSigma]);

    calibSet = struct([]);
    k = 0;

    for i = 1:surfacePack.nbExpiries
        for j = 1:surfacePack.nbTenors

            if ~surfacePack.validMask(i,j)
                continue
            end

            k = k + 1;

            calibSet(k) = obj.buildCalibrationPointFromSurface( ...
                surfacePack, pricers, i, j, ...
                'irs', args.irs, ...
                'notional', args.notional, ...
                'settleType', args.settleType);
        end
    end

    residualFun = @(theta) obj.residualGlobalG1PP( ...
        theta, calibSet, surfacePack, ...
        'objectiveType', args.objectiveType);

    if args.useLsqnonlin && exist('lsqnonlin', 'file') == 2

        options = optimoptions('lsqnonlin', ...
            'Display', 'none', ...
            'MaxIterations', 300, ...
            'FunctionTolerance', 1e-8, ...
            'StepTolerance', 1e-8);

        [thetaStar, resnorm, residuals, exitflag, output] = ...
            lsqnonlin(residualFun, theta0, [], [], options);

    else

        options = optimset('Display', 'off', 'MaxIter', 300);
        objFun = @(theta) sum(residualFun(theta).^2);

        [thetaStar, resnorm, exitflag, output] = ...
            fminsearch(objFun, theta0, options);

        residuals = residualFun(thetaStar);
    end

    params = exp(thetaStar);

    g1Global = struct();
    g1Global.a = params(1);
    g1Global.sigma = params(2);
    g1Global.thetaStar = thetaStar;
    g1Global.calibSet = calibSet;
    g1Global.residuals = residuals;
    g1Global.resnorm = resnorm;
    g1Global.exitflag = exitflag;
    g1Global.output = output;
    g1Global.rmse = sqrt(mean(residuals.^2));
    g1Global.maxAbsResidual = max(abs(residuals));
end
```

```matlab
function residuals = residualGlobalG1PP(obj, thetaRaw, calibSet, surfacePack, args)
    % G2++ Calibration Helper: Full-surface residual for constant G1++.

    arguments
        obj (1,1) GaussianTwoFactors;
        thetaRaw (:,1) double;
        calibSet (1,:) struct;
        surfacePack (1,1) struct;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    params = exp(thetaRaw(:));
    a = params(1);
    sigma = params(2);

    maxMat = max(surfacePack.matSwapDatesYr(:));
    volTimeStructure = maxMat + 1.0;
    volVector = sigma .* ones(2,1);

    residuals = NaN(numel(calibSet), 1);

    for k = 1:numel(calibSet)

        point = calibSet(k);

        modelPrice = obj.priceSwaptionG1PP( ...
            point.strike, ...
            point.expiry, ...
            point.paymentDates, ...
            a, ...
            volVector, ...
            volTimeStructure, ...
            'direction', point.direction, ...
            'notional', point.notional);

        switch args.objectiveType
            case 'price'
                scale = 1.0;
            otherwise
                scale = max(point.notional * point.annuity, 1e-12);
        end

        residuals(k) = sqrt(point.weight) * ...
            (modelPrice - point.marketPrice) / scale;

        if ~isfinite(residuals(k))
            residuals(k) = 1e6;
        end
    end
end
```

```matlab
function shapePack = bootstrapSliceG1PP(obj, surfacePack, pricers, g1Global, args)
    % G2++ Calibration Helper: Bootstraps one G1++ piecewise volatility
    % slice. Use once for the short-tenor shape and once for the long-tenor shape.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;
        g1Global (1,1) struct;

        args.tenorIdx (1,1) double;
        args.shapeName (1,:) char = 'sigma';

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';

        args.initialVol (1,1) double = NaN;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    tenorIdx = args.tenorIdx;
    quoteRows = find(surfacePack.validMask(:, tenorIdx));
    nbSteps = numel(quoteRows);

    initialVol = args.initialVol;
    if isnan(initialVol)
        initialVol = g1Global.sigma;
    end

    shape = NaN(nbSteps, 1);
    modelPrices = NaN(nbSteps, 1);
    marketPrices = NaN(nbSteps, 1);
    residuals = NaN(nbSteps, 1);
    strikes = NaN(nbSteps, 1);
    directions = strings(nbSteps, 1);
    solverMethod = strings(nbSteps, 1);

    timeStructure = surfacePack.expiryDatesYr(quoteRows);
    timeStructure = timeStructure(:);

    for k = 1:nbSteps

        row = quoteRows(k);

        point = obj.buildCalibrationPointFromSurface( ...
            surfacePack, pricers, row, tenorIdx, ...
            'irs', args.irs, ...
            'notional', args.notional, ...
            'settleType', args.settleType);

        if k == 1
            guess = initialVol;
        else
            guess = shape(k-1);
        end

        resFun = @(v) obj.residualBucketG1PP( ...
            v, k, shape, timeStructure, point, g1Global.a, ...
            'objectiveType', args.objectiveType);

        try
            volK = fzero(resFun, [1e-5, 1.0]);
            solverMethod(k) = "fzero";
        catch
            volK = fminsearch(@(v) resFun(abs(v)).^2, guess, ...
                optimset('Display', 'off', 'MaxIter', 100));
            volK = abs(volK);
            solverMethod(k) = "fminsearch";
        end

        shape(k) = max(volK, 1e-8);

        volVector = [shape(1:k); shape(k)];
        volTimeStructure = timeStructure(1:k);

        modelPrice = obj.priceSwaptionG1PP( ...
            point.strike, ...
            point.expiry, ...
            point.paymentDates, ...
            g1Global.a, ...
            volVector, ...
            volTimeStructure, ...
            'direction', point.direction, ...
            'notional', point.notional);

        modelPrices(k) = modelPrice;
        marketPrices(k) = point.marketPrice;
        residuals(k) = modelPrice - point.marketPrice;
        strikes(k) = point.strike;
        directions(k) = string(point.direction);
    end

    shapePack = struct();
    shapePack.name = args.shapeName;
    shapePack.tenorIdx = tenorIdx;
    shapePack.tenor = surfacePack.tenorsYr(tenorIdx);

    shapePack.timeStructure = timeStructure;
    shapePack.timeStructureDt = timeStructure;
    shapePack.shape = shape;

    shapePack.modelPrices = modelPrices;
    shapePack.marketPrices = marketPrices;
    shapePack.residuals = residuals;
    shapePack.strikes = strikes;
    shapePack.directions = directions;
    shapePack.solverMethod = solverMethod;

    shapePack.g1MeanReversion = g1Global.a;
    shapePack.quoteRows = quoteRows;
end
```

```matlab
function residual = residualBucketG1PP(obj, volCandidate, step, shape, timeStructure, point, a, args)
    % G2++ Calibration Helper: Scalar residual for one G1++ bootstrap bucket.

    arguments
        obj (1,1) GaussianTwoFactors;
        volCandidate (1,1) double;
        step (1,1) double;
        shape (:,1) double;
        timeStructure (:,1) double;
        point (1,1) struct;
        a (1,1) double;
        args.objectiveType (1,:) char = 'normalizedPrice';
    end

    volCandidate = abs(volCandidate);

    volSlice = shape(1:step);
    volSlice(step) = volCandidate;

    volVector = [volSlice; volCandidate];
    volTimeStructure = timeStructure(1:step);

    modelPrice = obj.priceSwaptionG1PP( ...
        point.strike, ...
        point.expiry, ...
        point.paymentDates, ...
        a, ...
        volVector, ...
        volTimeStructure, ...
        'direction', point.direction, ...
        'notional', point.notional);

    switch args.objectiveType
        case 'price'
            scale = 1.0;
        otherwise
            scale = max(point.notional * point.annuity, 1e-12);
    end

    residual = (modelPrice - point.marketPrice) / scale;

    if ~isfinite(residual)
        residual = 1e6;
    end
end
```

```matlab
function point = buildCalibrationPointFromSurface(obj, surfacePack, pricers, row, col, args)
    % G2++ Calibration Helper: Builds one market calibration point from the
    % normalized volatility surface.

    arguments
        obj (1,1) GaussianTwoFactors;
        surfacePack (1,1) struct;
        pricers (1,1) struct;
        row (1,1) double;
        col (1,1) double;

        args.irs (:,1) IRSVanille = obj.irsDefault;
        args.notional (1,1) double = 1e6;
        args.settleType (1,:) char = 'CashSettled';
    end

    swapStartDate = surfacePack.startSwapDates(row, col);
    matSwapDate = surfacePack.matSwapDates(row, col);
    exerciseDate = surfacePack.expiryDates(row);
    marketVol = surfacePack.volMatrix(row, col);

    irs = copy(args.irs(1));
    irs.setNotional(args.notional);
    irs.setDates(swapStartDate, matSwapDate);

    [npv, ~, ~] = pricers.swap.IRSVanille(irs, ...
        'measures', true, ...
        'details', true);

    if isempty(surfacePack.strikes)
        strike = npv.FairSwapRate;
    else
        strike = surfacePack.strikes(row, col);
    end

    direction = obj.swaptionDirection(strike, npv.FairSwapRate);
    irs.direction = direction;

    swaption = Swaption(irs, ...
        'exerciseDate', exerciseDate, ...
        'strike', strike, ...
        'settleType', args.settleType);

    [~, fixedLegPDates, ~] = irs.fixedLeg01(swapStartDate);

    fixedLegPDates = fixedLegPDates(:).';

    paymentDates = yearfracExtend( ...
        surfacePack.pricingDateNum, ...
        fixedLegPDates(2:end), ...
        obj.basis);

    paymentDates = paymentDates(:).';

    tenorsDt = yearfracExtend( ...
        fixedLegPDates(1:end-1), ...
        fixedLegPDates(2:end), ...
        swaption.irs.fixedLeg.interestDCC);

    tenorsDt = tenorsDt(:).';

    [marketPrice, ~, annuity, bachelierVol] = pricers.bachelier.Swaption( ...
        swaption, ...
        'measures', false, ...
        'vol', marketVol, ...
        'irsRes', npv);

    point = struct();
    point.row = row;
    point.col = col;

    point.irs = irs;
    point.swaption = swaption;
    point.npv = npv;

    point.expiryDate = exerciseDate;
    point.expiry = surfacePack.expiryDatesYr(row);
    point.swapStartDate = swapStartDate;
    point.matSwapDate = matSwapDate;

    point.paymentDates = paymentDates;
    point.tenorsDt = tenorsDt;
    point.fixedLegPDates = fixedLegPDates;

    point.strike = strike;
    point.direction = direction;
    point.notional = args.notional;

    point.marketVol = marketVol;
    point.bachelierVol = bachelierVol;
    point.marketPrice = marketPrice;
    point.annuity = annuity;

    point.weight = surfacePack.weights(row, col);
end
```

Now the stateful G1++ pricing block.

```matlab
function price = priceSwaptionG1PP(obj, strike, expiry, paymentDates, a, volVector, volTimeStructure, args)
    % G2++ Calibration Helper: Prices a European swaption under G1++.
    % This method temporarily maps the object to one-factor mode, reuses
    % existing G2++ bucket/variance helpers, then restores the original state.

    arguments
        obj (1,1) GaussianTwoFactors;
        strike (1,1) double;
        expiry (1,1) double;
        paymentDates (1,:) double;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;

        args.direction (1,:) char {mustBeMember(args.direction, {'PAY','REC'})} = 'PAY';
        args.notional (1,1) double = 1e6;
    end

    oldState = obj.setTemporaryG1PPState(a, volVector, volTimeStructure);

    try
        basket = obj.getBondBasketG1PP(expiry, paymentDates);
        gaussian = obj.getStateGaussianG1PP(expiry);

        price = obj.priceFromG1PPBasket( ...
            strike, expiry, paymentDates, basket, gaussian, ...
            'direction', args.direction, ...
            'notional', args.notional);

        obj.restoreG1PPState(oldState);

    catch ME
        obj.restoreG1PPState(oldState);
        rethrow(ME);
    end
end
```

```matlab
function oldState = setTemporaryG1PPState(obj, a, volVector, volTimeStructure)
    % G2++ Calibration Helper: Saves current G2++ state and sets temporary
    % one-factor G1++ state on the x-factor.

    arguments
        obj (1,1) GaussianTwoFactors;
        a (1,1) double;
        volVector (:,1) double;
        volTimeStructure (:,1) double;
    end

    oldState = struct();

    oldState.meanReversion_x = obj.meanReversion_x;
    oldState.meanReversion_y = obj.meanReversion_y;
    oldState.volatility_sigma = obj.volatility_sigma;
    oldState.volatility_eta = obj.volatility_eta;
    oldState.volatilityTimeStructure = obj.volatilityTimeStructure;
    oldState.volatilityTimeStructureDt = obj.volatilityTimeStructureDt;
    oldState.correlation = obj.correlation;

    obj.meanReversion_x = a;
    obj.meanReversion_y = a;

    obj.volatility_sigma = volVector(:).';
    obj.volatility_eta = zeros(size(obj.volatility_sigma));

    obj.volatilityTimeStructure = volTimeStructure(:).';
    obj.volatilityTimeStructureDt = [];

    obj.correlation = 0.0;
end
```

```matlab
function restoreG1PPState(obj, oldState)
    % G2++ Calibration Helper: Restores model state after temporary G1++ pricing.

    arguments
        obj (1,1) GaussianTwoFactors;
        oldState (1,1) struct;
    end

    obj.meanReversion_x = oldState.meanReversion_x;
    obj.meanReversion_y = oldState.meanReversion_y;
    obj.volatility_sigma = oldState.volatility_sigma;
    obj.volatility_eta = oldState.volatility_eta;
    obj.volatilityTimeStructure = oldState.volatilityTimeStructure;
    obj.volatilityTimeStructureDt = oldState.volatilityTimeStructureDt;
    obj.correlation = oldState.correlation;
end
```

```matlab
function basket = getBondBasketG1PP(obj, expiry, paymentDates)
    % G2++ Calibration Helper: Builds the one-factor affine bond basket.
    % Assumes the object is already in temporary G1++ state.

    arguments
        obj (1,1) GaussianTwoFactors;
        expiry (1,1) double;
        paymentDates (1,:) double;
    end

    paymentDates = paymentDates(:);

    zc = obj.getInitialMarketZC([expiry, paymentDates.']);
    zc = zc(:);

    P0T0 = zc(1);
    P0Ti = zc(2:end);

    B0Ti = obj.getH(obj.meanReversion_x, paymentDates - expiry);

    var0T0 = obj.variXOnly(0, expiry);

    varT0Ti = zeros(numel(paymentDates), 1);
    var0Ti = zeros(numel(paymentDates), 1);

    for i = 1:numel(paymentDates)
        Ti = paymentDates(i);
        varT0Ti(i) = obj.variXOnly(expiry, Ti);
        var0Ti(i) = obj.variXOnly(0, Ti);
    end

    A0Ti = (P0Ti ./ P0T0) .* exp(0.5 .* (varT0Ti - var0Ti + var0T0));

    basket = struct();
    basket.P0T0 = P0T0;
    basket.P0Ti = P0Ti;
    basket.A0Ti = A0Ti(:);
    basket.B0Ti = B0Ti(:);
end
```

```matlab
function output = variXOnly(obj, startDate, endDate)
    % G2++ Calibration Helper: One-factor integrated variance using the
    % existing x-factor variance contribution.

    arguments
        obj (1,1) GaussianTwoFactors;
        startDate (1,1) double;
        endDate (1,1) double;
    end

    output = sum(obj.variX(startDate, endDate));
end
```

```matlab
function gaussian = getStateGaussianG1PP(obj, expiry)
    % G2++ Calibration Helper: Computes the law of X_T under the T-forward
    % measure in temporary G1++ state.

    arguments
        obj (1,1) GaussianTwoFactors;
        expiry (1,1) double;
    end

    gaussian = struct();
    gaussian.muX = obj.meanForwardG1PP(0, expiry);
    gaussian.varX = obj.varTForward(0, expiry, 'process', 'x_process');
end
```

```matlab
function output = meanForwardG1PP(obj, startDate, endDate)
    % G2++ Calibration Helper: Mean of X(endDate) under the endDate-forward
    % measure in temporary G1++ state.

    arguments
        obj (1,1) GaussianTwoFactors;
        startDate (1,1) double;
        endDate (1,1) double;
    end

    idx = obj.getPricingIdx(startDate, endDate);

    deltaLeft = obj.deltaTLeft(startDate, endDate);
    deltaRight = obj.deltaTRight(startDate, endDate);

    deltaLeft = deltaLeft(:);
    deltaRight = deltaRight(:);

    sigma = obj.volatility_sigma(idx);
    sigma = sigma(:);

    a = obj.meanReversion_x;

    expA = obj.getExpDiff(a, deltaLeft, deltaRight);
    exp2A = obj.getExpDiff(2*a, deltaLeft, deltaRight);

    output = sum( ...
        -(sigma.^2 ./ a.^2) .* expA ...
        + (sigma.^2 ./ (2*a.^2)) .* exp2A);
end
```

```matlab
function xBar = solveXBarG1PP(obj, lambda, B)
    % G2++ Calibration Helper: Solves the one-factor exercise boundary.

    arguments
        obj (1,1) GaussianTwoFactors;
        lambda (:,1) double;
        B (:,1) double;
    end

    %#ok<INUSD>

    lambda = lambda(:);
    B = B(:);

    f = @(x) 1 - sum(lambda .* exp(-B .* x));

    lo = -0.10;
    hi =  0.10;

    flo = f(lo);
    fhi = f(hi);

    k = 0;

    while flo * fhi > 0 && k < 40
        width = hi - lo;

        lo = lo - 2 * width;
        hi = hi + 2 * width;

        flo = f(lo);
        fhi = f(hi);

        k = k + 1;
    end

    if flo * fhi > 0
        xBar = NaN;
        return
    end

    xBar = fzero(f, [lo, hi]);
end
```

```matlab
function price = priceFromG1PPBasket(obj, strike, expiry, paymentDates, basket, gaussian, args)
    % G2++ Calibration Helper: Computes the closed-form one-factor Gaussian
    % swaption price from a prepared bond basket and Gaussian state.

    arguments
        obj (1,1) GaussianTwoFactors;
        strike (1,1) double;
        expiry (1,1) double;
        paymentDates (1,:) double;
        basket (1,1) struct;
        gaussian (1,1) struct;

        args.direction (1,:) char {mustBeMember(args.direction, {'PAY','REC'})} = 'PAY';
        args.notional (1,1) double = 1e6;
    end

    %#ok<INUSL>

    paymentDates = paymentDates(:);

    delta = diff([expiry; paymentDates]);

    coeffs = strike .* delta;
    coeffs(end) = coeffs(end) + 1.0;

    lambda = coeffs .* basket.A0Ti(:);
    B = basket.B0Ti(:);

    xBar = obj.solveXBarG1PP(lambda, B);

    if ~isfinite(xBar)
        price = NaN;
        return
    end

    muX = gaussian.muX;
    varX = gaussian.varX;
    sigX = sqrt(max(varX, 0));

    if sigX <= 0
        basketValue = sum(lambda .* exp(-B .* muX));

        if strcmpi(args.direction, 'PAY')
            payoff = max(1 - basketValue, 0);
        else
            payoff = max(basketValue - 1, 0);
        end

        price = args.notional .* basket.P0T0 .* payoff;
        return
    end

    h0 = (xBar - muX) ./ sigX;
    h1 = h0 + B .* sigX;
    kappa = -B .* muX + 0.5 .* (B.^2) .* varX;

    if strcmpi(args.direction, 'PAY')
        expectation = normcdf(-h0) - sum(lambda .* exp(kappa) .* normcdf(-h1));
    else
        expectation = sum(lambda .* exp(kappa) .* normcdf(h1)) - normcdf(h0);
    end

    price = args.notional .* basket.P0T0 .* max(expectation, 0);
end
```

Typical call:

```matlab
g1Global = obj.calibrateGlobalG1PP(surfacePack, pricers);

sigmaShapePack = obj.bootstrapSliceG1PP( ...
    surfacePack, pricers, g1Global, ...
    'tenorIdx', xg2ppState.shortTenorIdx, ...
    'shapeName', 'sigma');

etaShapePack = obj.bootstrapSliceG1PP( ...
    surfacePack, pricers, g1Global, ...
    'tenorIdx', xg2ppState.longTenorIdx, ...
    'shapeName', 'eta');
```

```matlab
theta0 = log([aTest; sigmaTest]);
theta1 = log([0.30; 0.020]);

r0_norm = G2PP.residualGlobalG1PP( ...
    theta0, g1Global.calibSet, calibVolSurface, ...
    'objectiveType', 'normalizedPrice');

r1_norm = G2PP.residualGlobalG1PP( ...
    theta1, g1Global.calibSet, calibVolSurface, ...
    'objectiveType', 'normalizedPrice');

r0_price = G2PP.residualGlobalG1PP( ...
    theta0, g1Global.calibSet, calibVolSurface, ...
    'objectiveType', 'price');

r1_price = G2PP.residualGlobalG1PP( ...
    theta1, g1Global.calibSet, calibVolSurface, ...
    'objectiveType', 'price');

summaryCheck = table( ...
    norm(r0_norm), norm(r1_norm), ...
    norm(r0_price), norm(r1_price), ...
    'VariableNames', {'Norm_r0_normalized','Norm_r1_normalized','Norm_r0_price','Norm_r1_price'});

disp(summaryCheck);

```

```matlab
p = g1Global.calibSet(1);

modelPrice0 = G2PP.priceSwaptionG1PP( ...
    p.strike, ...
    p.expiry, ...
    p.paymentDates, ...
    aTest, ...
    sigmaTest .* ones(2,1), ...
    max(calibVolSurface.matSwapDatesYr(:)) + 1, ...
    'direction', p.direction, ...
    'notional', p.notional);

scaleOld = p.notional * p.annuity;

onePointCheck = table( ...
    p.marketPrice, ...
    modelPrice0, ...
    modelPrice0 - p.marketPrice, ...
    p.annuity, ...
    p.notional, ...
    scaleOld, ...
    'VariableNames', {'MarketPrice','ModelPrice','RawError','Annuity','Notional','OldScale'});

disp(onePointCheck);
```

# Calibration Pipeline Review

Yes. Up to now, the tests should target two layers.

```text
TEST BLOCK 1 — Structural plumbing
    → methods exist
    → surfacePack dimensions
    → xg2ppState slice indices
    → calibration point construction
    → temporary G1++ state is restored

TEST BLOCK 2 — First calibration fundamentals
    → constant G1++ global calibration runs
    → slice bootstrap runs
    → G1++ price is finite and nonnegative
    → calibrated slice instrument reprices its market target reasonably
```

Below is a sober MLX block.

```matlab
%% XG2++ calibration pipeline — first tests
% Assumption:
%   A GaussianTwoFactors object already exists in the workspace and already
%   contains a swaption volatility cube.
%
% Rename here if your object has another name.

G2PP = G2PP;   % keep or replace by your actual object name
calibMoneyness = 0;

%% 1. Structural plumbing tests

requiredMethods = [
    "buildPricersG2PP"
    "buildVolSurfacePackXG2PP"
    "initCalibXG2PP"
    "parseTenorsXG2PP"
    "buildCalibrationPointFromSurface"
    "calibrateGlobalG1PP"
    "residualGlobalG1PP"
    "bootstrapSliceG1PP"
    "residualBucketG1PP"
    "priceSwaptionG1PP"
    "setTemporaryG1PPState"
    "restoreG1PPState"
    "getBondBasketG1PP"
    "variXOnly"
    "getStateGaussianG1PP"
    "meanForwardG1PP"
    "solveXBarG1PP"
    "priceFromG1PPBasket"
    "expandVolBucketsXG2PP"
];

availableMethods = string(methods(G2PP));
missingMethods = setdiff(requiredMethods, availableMethods);

assert(isempty(missingMethods), ...
    "Missing methods: " + strjoin(missingMethods, ", "));

pricers = G2PP.buildPricersG2PP();

surfacePack = G2PP.buildVolSurfacePackXG2PP(calibMoneyness);

xg2ppState = G2PP.initCalibXG2PP(surfacePack);

assert(isequal(size(surfacePack.volMatrix), size(surfacePack.validMask)));
assert(isequal(size(surfacePack.startSwapDates), size(surfacePack.volMatrix)));
assert(isequal(size(surfacePack.matSwapDates), size(surfacePack.volMatrix)));

assert(numel(surfacePack.expiryDatesYr) == surfacePack.nbExpiries);
assert(numel(surfacePack.tenorsYr) == surfacePack.nbTenors);
assert(numel(surfacePack.quoteValues) == surfacePack.nbQuotes);
assert(numel(surfacePack.quoteRows) == surfacePack.nbQuotes);
assert(numel(surfacePack.quoteCols) == surfacePack.nbQuotes);

assert(xg2ppState.shortTenorIdx >= 1);
assert(xg2ppState.shortTenorIdx <= surfacePack.nbTenors);
assert(xg2ppState.longTenorIdx >= 1);
assert(xg2ppState.longTenorIdx <= surfacePack.nbTenors);

assert(any(surfacePack.validMask(:, xg2ppState.shortTenorIdx)));
assert(any(surfacePack.validMask(:, xg2ppState.longTenorIdx)));

disp("Structural setup tests passed.");
```

```matlab
%% 2. Calibration point construction test

row0 = surfacePack.quoteRows(1);
col0 = surfacePack.quoteCols(1);

point0 = G2PP.buildCalibrationPointFromSurface( ...
    surfacePack, pricers, row0, col0);

assert(isfinite(point0.strike));
assert(isfinite(point0.marketVol));
assert(isfinite(point0.marketPrice));
assert(isfinite(point0.annuity));
assert(point0.marketPrice >= 0);
assert(all(point0.paymentDates > point0.expiry));

assert(strcmp(point0.direction, 'PAY') || strcmp(point0.direction, 'REC'));

disp("Calibration point construction test passed.");
```

```matlab
%% 3. Temporary G1++ state restoration test

stateBefore = struct();
stateBefore.meanReversion_x = G2PP.meanReversion_x;
stateBefore.meanReversion_y = G2PP.meanReversion_y;
stateBefore.volatility_sigma = G2PP.volatility_sigma;
stateBefore.volatility_eta = G2PP.volatility_eta;
stateBefore.volatilityTimeStructure = G2PP.volatilityTimeStructure;
stateBefore.volatilityTimeStructureDt = G2PP.volatilityTimeStructureDt;
stateBefore.correlation = G2PP.correlation;

tmpVolTimeStructure = max(surfacePack.expiryDatesYr) + 1;
tmpVolVector = 0.01 .* ones(2,1);

oldState = G2PP.setTemporaryG1PPState( ...
    0.10, ...
    tmpVolVector, ...
    tmpVolTimeStructure);

G2PP.restoreG1PPState(oldState);

assert(isequaln(G2PP.meanReversion_x, stateBefore.meanReversion_x));
assert(isequaln(G2PP.meanReversion_y, stateBefore.meanReversion_y));
assert(isequaln(G2PP.volatility_sigma, stateBefore.volatility_sigma));
assert(isequaln(G2PP.volatility_eta, stateBefore.volatility_eta));
assert(isequaln(G2PP.volatilityTimeStructure, stateBefore.volatilityTimeStructure));
assert(isequaln(G2PP.volatilityTimeStructureDt, stateBefore.volatilityTimeStructureDt));
assert(isequaln(G2PP.correlation, stateBefore.correlation));

disp("Temporary G1++ state restoration test passed.");
```

```matlab
%% 4. Single G1++ price sanity test

aTest = max(G2PP.meanReversion_x, 0.05);
sigmaTest = 0.01;

volTimeStructure = max(surfacePack.matSwapDatesYr(:)) + 1;
volVector = sigmaTest .* ones(2,1);

priceG1 = G2PP.priceSwaptionG1PP( ...
    point0.strike, ...
    point0.expiry, ...
    point0.paymentDates, ...
    aTest, ...
    volVector, ...
    volTimeStructure, ...
    'direction', point0.direction, ...
    'notional', point0.notional);

assert(isfinite(priceG1));
assert(priceG1 >= 0);

assert(isequaln(G2PP.meanReversion_x, stateBefore.meanReversion_x));
assert(isequaln(G2PP.volatility_sigma, stateBefore.volatility_sigma));
assert(isequaln(G2PP.volatility_eta, stateBefore.volatility_eta));

disp("Single G1++ price sanity test passed.");
```

```matlab
%% 5. Global G1++ calibration test

g1Global = G2PP.calibrateGlobalG1PP( ...
    surfacePack, ...
    pricers, ...
    'initialA', aTest, ...
    'initialSigma', sigmaTest, ...
    'useLsqnonlin', true, ...
    'objectiveType', 'normalizedPrice');

assert(isfinite(g1Global.a));
assert(isfinite(g1Global.sigma));
assert(g1Global.a > 0);
assert(g1Global.sigma > 0);
assert(all(isfinite(g1Global.residuals)));

disp("Global G1++ calibration test passed.");
```

```matlab
%% 6. Short-tenor slice bootstrap test

sigmaShapePack = G2PP.bootstrapSliceG1PP( ...
    surfacePack, ...
    pricers, ...
    g1Global, ...
    'tenorIdx', xg2ppState.shortTenorIdx, ...
    'shapeName', 'sigma', ...
    'objectiveType', 'normalizedPrice');

assert(all(isfinite(sigmaShapePack.shape)));
assert(all(sigmaShapePack.shape > 0));
assert(all(isfinite(sigmaShapePack.modelPrices)));
assert(all(isfinite(sigmaShapePack.marketPrices)));

disp("Short-tenor G1++ bootstrap test passed.");
```

```matlab
%% 7. Long-tenor slice bootstrap test

etaShapePack = G2PP.bootstrapSliceG1PP( ...
    surfacePack, ...
    pricers, ...
    g1Global, ...
    'tenorIdx', xg2ppState.longTenorIdx, ...
    'shapeName', 'eta', ...
    'objectiveType', 'normalizedPrice');

assert(all(isfinite(etaShapePack.shape)));
assert(all(etaShapePack.shape > 0));
assert(all(isfinite(etaShapePack.modelPrices)));
assert(all(isfinite(etaShapePack.marketPrices)));

disp("Long-tenor G1++ bootstrap test passed.");
```

```matlab
%% 8. Reprice one calibrated slice instrument

k = numel(sigmaShapePack.shape);

rowK = sigmaShapePack.quoteRows(k);
colK = sigmaShapePack.tenorIdx;

pointK = G2PP.buildCalibrationPointFromSurface( ...
    surfacePack, pricers, rowK, colK);

volVectorK = G2PP.expandVolBucketsXG2PP( ...
    sigmaShapePack.shape(1:k), ...
    xg2ppState.terminalPolicy);

volTimeStructureK = sigmaShapePack.timeStructure(1:k);

priceK = G2PP.priceSwaptionG1PP( ...
    pointK.strike, ...
    pointK.expiry, ...
    pointK.paymentDates, ...
    g1Global.a, ...
    volVectorK, ...
    volTimeStructureK, ...
    'direction', pointK.direction, ...
    'notional', pointK.notional);

normalizedErrorK = abs(priceK - pointK.marketPrice) / ...
    max(pointK.notional * pointK.annuity, 1e-12);

assert(isfinite(priceK));
assert(priceK >= 0);
assert(isfinite(normalizedErrorK));

resultOneInstrument = table( ...
    pointK.expiry, ...
    surfacePack.tenorsYr(colK), ...
    pointK.marketPrice, ...
    priceK, ...
    normalizedErrorK, ...
    'VariableNames', {'Expiry','Tenor','MarketPrice','ModelPrice','NormalizedError'});

disp(resultOneInstrument);
```

```matlab
%% 9. Compact diagnostics

diagnostics = struct();

diagnostics.globalG1.a = g1Global.a;
diagnostics.globalG1.sigma = g1Global.sigma;
diagnostics.globalG1.rmse = g1Global.rmse;
diagnostics.globalG1.maxAbsResidual = g1Global.maxAbsResidual;

diagnostics.shortSlice.maxAbsPriceError = max(abs( ...
    sigmaShapePack.modelPrices - sigmaShapePack.marketPrices));

diagnostics.longSlice.maxAbsPriceError = max(abs( ...
    etaShapePack.modelPrices - etaShapePack.marketPrices));

diagnostics.shortSlice.nbBuckets = numel(sigmaShapePack.shape);
diagnostics.longSlice.nbBuckets = numel(etaShapePack.shape);

disp(diagnostics);
```

One detail to watch: if `bootstrapSliceG1PP` still contains the hardcoded line

```matlab
volVector = [shape(1:k); shape(k)];
```

replace it by

```matlab
volVector = G2PP.expandVolBucketsXG2PP( ...
    shape(1:k), ...
    xg2ppState.terminalPolicy);
```

or pass `'terminalPolicy'` directly into `bootstrapSliceG1PP`. Otherwise the test still runs, but `terminalPolicy` is not actually being tested.

## Response:
5/18/2026, 12:13:10 PM

They have different jobs.

```text
priceSwaptionG1PP
    = state wrapper + orchestration

priceFromG1PPBasket
    = pure pricing formula once inputs are prepared
```

# 1. `priceSwaptionG1PP`

This is the method called by calibration residuals.

It receives the natural calibration inputs:

```matlab
strike
expiry
paymentDates
a
volVector
volTimeStructure
```

Then it does the dangerous part:

```matlab
oldState = obj.setTemporaryG1PPState(...)
```

So its responsibility is:

```text
save G2++ state
→ set temporary G1++ state
→ build basket
→ build Gaussian law
→ call closed-form price
→ restore G2++ state
```

It owns object-state safety.

---

# 2. `priceFromG1PPBasket`

This method assumes the state-dependent objects are already prepared:

```matlab
basket
gaussian
```

So it does only the mathematical pricing:

$$
P(0,T_0)
\mathbb E^{T_0}
\left[
\left(
1-\sum_i c_i A_i e^{-B_i X}
\right)^+
\right].
$$

It owns:

```text
coefficients c_i
Jamshidian boundary xBar
h0, h1, kappa
payer/receiver formula
final price
```

It should not mutate the object.

---

# 3. Why having two is useful

Because it separates the only dangerous part from the formula:

```text
priceSwaptionG1PP
    can mutate object temporarily.

priceFromG1PPBasket
    must never mutate object.
```

This makes bugs easier to locate.

If pricing is wrong but state restoration is fine, inspect `priceFromG1PPBasket`.

If the object gets polluted after calibration, inspect `priceSwaptionG1PP`.

---

# 4. Should we keep both?

I would keep both.

This is not over-fragmentation. It is the right split:

```text
wrapper / state safety
formula / numerical pricing
```

But I would maybe rename the second one to make its role clearer:

```matlab
pricePreparedG1PPBasket
```

or

```matlab
priceG1PPFromBasket
```

Then the architecture reads better:

```text
priceSwaptionG1PP
│
├─ setTemporaryG1PPState
├─ getBondBasketG1PP
├─ getStateGaussianG1PP
├─ priceG1PPFromBasket
└─ restoreG1PPState
```

# 5. Minimal rule

Only this method should appear in calibration residuals:

```matlab
priceSwaptionG1PP
```

This method should remain internal:

```matlab
k = 5;

rowK = etaShapePack.quoteRows(k);
colK = etaShapePack.tenorIdx;

pointK = G2PP.buildCalibrationPointFromSurface( ...
    calibVolSurface, pricers, rowK, colK);

volSolved = etaShapePack.shape(k);

rLow = G2PP.residualBucketG1PP( ...
    0.50 * volSolved, k, etaShapePack.shape, ...
    etaShapePack.timeStructure, pointK, g1Global.a, ...
    'objectiveType', 'price');

rMid = G2PP.residualBucketG1PP( ...
    volSolved, k, etaShapePack.shape, ...
    etaShapePack.timeStructure, pointK, g1Global.a, ...
    'objectiveType', 'price');

rHigh = G2PP.residualBucketG1PP( ...
    1.50 * volSolved, k, etaShapePack.shape, ...
    etaShapePack.timeStructure, pointK, g1Global.a, ...
    'objectiveType', 'price');

table(rLow, rMid, rHigh)
```

So the calibration layer never handles baskets or Gaussian states directly.

```matlab
%% Minimal G1++ pricing audit

row0 = surfacePack.quoteRows(1);
col0 = surfacePack.quoteCols(1);

point0 = G2PP.buildCalibrationPointFromSurface( ...
    surfacePack, pricers, row0, col0);

stateBefore.meanReversion_x = G2PP.meanReversion_x;
stateBefore.meanReversion_y = G2PP.meanReversion_y;
stateBefore.volatility_sigma = G2PP.volatility_sigma;
stateBefore.volatility_eta = G2PP.volatility_eta;
stateBefore.volatilityTimeStructure = G2PP.volatilityTimeStructure;
stateBefore.volatilityTimeStructureDt = G2PP.volatilityTimeStructureDt;
stateBefore.correlation = G2PP.correlation;

aTest = max(G2PP.meanReversion_x, 0.05);
volTimeStructure = max(surfacePack.matSwapDatesYr(:)) + 1.0;

volLow  = 0.005 .* ones(2,1);
volMid  = 0.010 .* ones(2,1);
volHigh = 0.020 .* ones(2,1);

priceLow = G2PP.priceSwaptionG1PP( ...
    point0.strike, point0.expiry, point0.paymentDates, ...
    aTest, volLow, volTimeStructure, ...
    'direction', point0.direction, ...
    'notional', point0.notional);

priceMid = G2PP.priceSwaptionG1PP( ...
    point0.strike, point0.expiry, point0.paymentDates, ...
    aTest, volMid, volTimeStructure, ...
    'direction', point0.direction, ...
    'notional', point0.notional);

priceHigh = G2PP.priceSwaptionG1PP( ...
    point0.strike, point0.expiry, point0.paymentDates, ...
    aTest, volHigh, volTimeStructure, ...
    'direction', point0.direction, ...
    'notional', point0.notional);

assert(isfinite(priceLow));
assert(isfinite(priceMid));
assert(isfinite(priceHigh));
assert(priceLow <= priceMid + 1e-8);
assert(priceMid <= priceHigh + 1e-8);

pricePAY = G2PP.priceSwaptionG1PP( ...
    point0.strike, point0.expiry, point0.paymentDates, ...
    aTest, volMid, volTimeStructure, ...
    'direction', 'PAY', ...
    'notional', point0.notional);

priceREC = G2PP.priceSwaptionG1PP( ...
    point0.strike, point0.expiry, point0.paymentDates, ...
    aTest, volMid, volTimeStructure, ...
    'direction', 'REC', ...
    'notional', point0.notional);

oldState = G2PP.setTemporaryG1PPState(aTest, volMid, volTimeStructure);
basketMid = G2PP.getBondBasketG1PP(point0.expiry, point0.paymentDates);
G2PP.restoreG1PPState(oldState);

deltaMid = diff([point0.expiry; point0.paymentDates(:)]);

coeffsMid = point0.strike .* deltaMid;
coeffsMid(end) = coeffsMid(end) + 1.0;

parityRHS = point0.notional .* ...
    (basketMid.P0T0 - sum(coeffsMid .* basketMid.P0Ti(:)));

parityError = (pricePAY - priceREC) - parityRHS;

assert(abs(parityError) <= 1e-6 * ...
    max(1, abs(pricePAY) + abs(priceREC) + abs(parityRHS)));

assert(isequaln(G2PP.meanReversion_x, stateBefore.meanReversion_x));
assert(isequaln(G2PP.meanReversion_y, stateBefore.meanReversion_y));
assert(isequaln(G2PP.volatility_sigma, stateBefore.volatility_sigma));
assert(isequaln(G2PP.volatility_eta, stateBefore.volatility_eta));
assert(isequaln(G2PP.volatilityTimeStructure, stateBefore.volatilityTimeStructure));
assert(isequaln(G2PP.volatilityTimeStructureDt, stateBefore.volatilityTimeStructureDt));
assert(isequaln(G2PP.correlation, stateBefore.correlation));

auditG1PP = table( ...
    priceLow, priceMid, priceHigh, pricePAY, priceREC, parityError, ...
    'VariableNames', {'PriceLow','PriceMid','PriceHigh','PricePAY','PriceREC','ParityError'});
```

```matlab
priceSwaptionG1PP
```
```matlab
priceSwaptionG1PP
```
```matlab
priceSwaptionG1PP
```
```matlab
priceSwaptionG1PP
```

SHAPE ASSEMBLY
│
├─ buildShapeStateG2PP
├─ alignShapeBucketsG2PP
└─ initializeScaledG2PPState

SCALED G2++ CALIBRATION
│
├─ buildFullSurfaceCalibrationSetG2PP
├─ residualScaledG2PP
├─ solveScaledG2PP
├─ setTemporaryScaledG2PPState
├─ restoreScaledG2PPState
└─ commitScaledG2PP

---

