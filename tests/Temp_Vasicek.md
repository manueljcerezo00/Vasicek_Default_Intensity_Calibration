# Calibration Procedure Review

**User:** Anonymous  
**Created:** 5/12/2026 21:05:48  
**Updated:** 5/13/2026 0:00:04  
**Exported:** 5/13/2026 0:07:28  
**Link:** [https://chatgpt.com/g/g-p-69f3ccd06ce88191966dfce0df25eb6f-g2/c/6a037a08-7fac-8384-8ad6-c27b237736b4](https://chatgpt.com/g/g-p-69f3ccd06ce88191966dfce0df25eb6f-g2/c/6a037a08-7fac-8384-8ad6-c27b237736b4)  

## Response:
5/12/2026, 11:48:00 PM

> **Correcting misunderstanding and refining citation approach**
>
> It looks like I need to apologize and provide the correct summary with file citations. Since the file content doesn't have line markers, I'll just use the parsed text pages and cite the file properly with the format indicated. I'll make sure to include the required citations from earlier.
>
> Thought for 17s

You’re right. I had blurred two things.

The paper’s procedure is **not** “G2++ bootstrap bucket by bucket on a co-terminal vector.” That was a reasonable engineering variant, but it is **not Bonino-Casalini**.

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

If your manager wants to keep mean reversions fixed, then the production variant is:

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

was our engineering shortcut. It is **not Bonino**.

Bonino’s dimensional reduction is instead:

$$
\sigma_k=\alpha\sigma^s_k,
\qquad
\eta_k=\beta\eta^s_k.
$$

That is the key proposition we must now encode.

## Response:
5/12/2026, 11:54:54 PM

> **Clarifying analogy and citation for shape scaling**
>
> The user asks about an analogy related to scaling. I'll differentiate multiplicative scaling of shape from an exponential transformation. For the multiplicative scaling, applying positive constants (alpha, beta) preserves the shape of the function, but adjusts its level. Regarding citations, it seems I can reference the file without line numbers, as previous responses did the same.
>
> Thought for 8s

Yes, but they act at **different levels**.

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

So yes: **multiplicative scaling becomes additive after log transformation**. But conceptually:

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

## Response:
5/12/2026, 11:58:34 PM

> **Clarifying citation format for paper reference**
>
> I need to decide how to make citations for the user's paper. I’ve already used file content in context, but it doesn’t include specific line numbers. Instructions suggest I need line ranges, which wasn’t available in earlier results. I’ll check if a different method can give more precise citations.
>
> **Reassessing strategy and reuse of functions**
>
> The user wants a detailed but concise strategy. I’ll focus on rethinking the pipeline from G2++ to Bonino, while identifying reusable functions. Functions like `buildCalibrationGrid` and `buildPricers` seem useful. For Bonino's Step 2, I might reuse the existing calibrator. I’ll need specific adapter functions for slice bootstrap.
>
> **Revisiting vector integration and Bonino Step 3 implementation**
>
> I can propose integrating the vector after `g2ppBucketPriceResidual`. However, to calibrate the full surface, we need instrument packs and market prices. I’ll create a function, like `initializeBoninoStateG2PP`, that accepts `sigmaShape/etaShape`. For now, we can implement Step 3's scaled G2++ layer and add the initialization hook without fully implementing G1++.
>
> **Focusing on Bonino Step 3 implementation and functions**
>
> Since mclick doesn't provide line numbers, we’ll cite with file markers. For the strategy, I won’t continue the local loop and will pivot to focus on Step 3’s global calibration. We’ll implement necessary functions like `boninoScaledG2PPCalibration`, `validateBoninoShapesG2PP`, and others, to orchestrate the process based on G1++ calibration.
>
> Thought for 45s

Object: **Bonino-compatible calibration layer after your current `g2ppBucketPriceResidual` stage**.

The correction is structural: **do not keep extending the one-bucket G2++ residual as the main Bonino path**. That helper is still useful as a test tool or alternative local calibration. But Bonino-Casalini’s final G2++ step is not local bucket-by-bucket. They first build two volatility shapes from two G1++ slice bootstraps, then globally calibrate scalar multipliers and correlation on the whole swaption surface. The paper explicitly uses shortest-tenor and longest-tenor G1++ slices for Step 2, then sets $\sigma=\alpha\sigma^s$, $\eta=\beta\eta^s$, and calibrates the reduced G2++ model globally. `ssrn-3598285 (1).pdf` `ssrn-3598285 (1).pdf`

So the strategy is:

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



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)