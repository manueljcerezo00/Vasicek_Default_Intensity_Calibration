# Surface-level engine audit


> **Providing MATLAB code for building a table**
>
> I’ll answer the user's request for code involving the calibration pipeline. The class should handle surfacePack quoteRows/Cols and generate bootTenorResults. Given the context from the file, it seems unnecessary to overcomplicate, so I’ll avoid excessive formalization and keep the code focused. It'll likely be MATLAB with a loop over the data.
>
> **Addressing citations while providing code**
>
> I’ll include citations for file-based details, especially since the method comes from files. While code doesn’t always need citations, if I’m using information from a file (like GaussianTwoFactors), I’ll make sure to cite it. Since the search results don’t have line numbers, I might have to open the file for the correct citation.
>
> **Final format and addressing citations for code**
>
> In my final answer, I’ll include the code and mention isolating the adapter and replacing the exact function. Since the file search didn’t provide line numbers, I’ll include the citation marker without range details, as that's what I can work with. I’m confident that’s enough, so I’ll finish with that and include the code in three parts.
>
> **Building the table from calibration result**
>
> To build the table, I'll use the calibration result and define columns for positions like expiry, tenor, strike, and others including sigma and volatility. For sigma, I’ll check if it's a vector and pick the last active or median value. The table should be in long format with one row per calibration block.
>
> Thought for 26s

I’ll make the diagnostic table builder independent of the final plotting block. The only fragile part is the calibration executor name, so I’ll isolate it in one local adapter function.

Yes. Build the table as a **post-calibration ledger**: one row per calibrated block $(k,j)$. The diagnostic code should not depend on plotting. It should first create:

```matlab
bootTenorResults
```

with stable columns. Since your `GaussianTwoFactors` object stores bucketed volatility, mean reversion, correlation, and time-structure state, the table must explicitly save the calibrated parameter state after each block, not just the final object state. `GaussianTwoFactors.txt`

Below is the scaffold.

---

## 1. Main MLX block

```matlab
%% Build bootstrapped-tenor calibration result table
% Required workspace:
%   G2PP
%   surfacePack
%   pricers
%
% Output:
%   bootTenorResults : one row per calibrated bucket/tenor block

rows = surfacePack.quoteRows(:);
cols = surfacePack.quoteCols(:);

bootTenorResults = buildBootTenorResultsG1PP( ...
    G2PP, surfacePack, pricers, rows, cols, ...
    'direction', 'PAY', ...
    'notional', 1.0, ...
    'defaultExitFlag', NaN);

disp(bootTenorResults);
```

---

## 2. Builder function

This function loops over the calibration grid and records the output into a clean table.

```matlab
function bootTenorResults = buildBootTenorResultsG1PP(G2PP, surfacePack, pricers, rows, cols, args)
    % Builds a diagnostic table for the bootstrapped-tenor G1++ calibration.
    %
    % One row = one calibrated block (bucket step k, tenor j).
    %
    % The actual calibration call is isolated in:
    %
    %     runOneBootTenorCalibrationG1PP
    %
    % so the diagnostic table remains stable even if the calibration
    % function name/signature changes.

    arguments
        G2PP
        surfacePack
        pricers
        rows (:,1) double
        cols (:,1) double
        args.direction (1,:) char = 'PAY'
        args.notional (1,1) double = 1.0
        args.defaultExitFlag (1,1) double = NaN
    end

    records = struct([]);
    kRec = 0;

    for i = 1:numel(rows)
        for j = 1:numel(cols)

            row = rows(i);
            col = cols(j);

            kRec = kRec + 1;

            try
                point = G2PP.buildCalibrationPointFromSurface( ...
                    surfacePack, pricers, row, col);

                calibOut = runOneBootTenorCalibrationG1PP( ...
                    G2PP, surfacePack, pricers, row, col, point);

                rec = packBootTenorRecordG1PP( ...
                    G2PP, point, calibOut, row, col, i, j, ...
                    args.defaultExitFlag);

            catch ME
                rec = emptyBootTenorRecordG1PP(row, col, i, j);
                rec.errorMessage = string(ME.message);
            end

            records(kRec) = rec;
        end
    end

    bootTenorResults = struct2table(records);

    % Stable ordering for diagnostics.
    bootTenorResults = sortrows(bootTenorResults, ...
        {'BucketIdx','TenorYr','Row','Col'});
end
```

---

## 3. Calibration adapter

This is the only part you should adapt to your exact function name.

```matlab
function calibOut = runOneBootTenorCalibrationG1PP(G2PP, surfacePack, pricers, row, col, point)
    % Adapter around the current G1++ bootstrapped-tenor calibration function.
    %
    % Replace the body with the exact function you are currently using.
    %
    % Required output convention:
    %   calibOut.a
    %   calibOut.sigma
    %   calibOut.rmse
    %   calibOut.maxAbsResidual
    %   calibOut.exitflag
    %   calibOut.residuals
    %   calibOut.modelPrice
    %   calibOut.marketPrice

    %#ok<INUSD>

    % ------------------------------------------------------------
    % OPTION A: object method
    % ------------------------------------------------------------
    if any(strcmp(methods(G2PP), 'bootstrapSliceG1PP'))

        calibOut = G2PP.bootstrapSliceG1PP( ...
            surfacePack, pricers, row, col);

        return;
    end

    % ------------------------------------------------------------
    % OPTION B: standalone function
    % ------------------------------------------------------------
    if exist('bootstrapSliceG1PP', 'file') == 2

        calibOut = bootstrapSliceG1PP( ...
            G2PP, surfacePack, pricers, row, col);

        return;
    end

    % ------------------------------------------------------------
    % OPTION C: manual fallback from current object state
    % ------------------------------------------------------------
    % This does NOT calibrate. It only records the state. Useful if you
    % already ran the calibration externally and want a ledger.
    calibOut = struct;
    calibOut.a = G2PP.meanReversion_x;
    calibOut.sigma = localExtractLastFinite(G2PP.volatility_sigma);
    calibOut.volatility_sigma = G2PP.volatility_sigma;
    calibOut.volatilityTimeStructure = G2PP.volatilityTimeStructure;
    calibOut.volatilityTimeStructureDt = G2PP.volatilityTimeStructureDt;
    calibOut.rmse = NaN;
    calibOut.maxAbsResidual = NaN;
    calibOut.exitflag = NaN;
    calibOut.residuals = NaN;
    calibOut.modelPrice = NaN;
    calibOut.marketPrice = NaN;
end
```

---

## 4. Record packer

This normalizes outputs even if your calibration result uses slightly different field names.

```matlab
function rec = packBootTenorRecordG1PP(G2PP, point, calibOut, row, col, bucketIdx, tenorIdx, defaultExitFlag)
    % Converts one calibration output into one stable diagnostic row.

    expiry = localGetField(point, ["expiry","Expiry","currMaturity"], NaN);
    paymentDates = localGetField(point, ["paymentDates","ulSwapSteps"], NaN);
    strike = localGetField(point, ["strike","Strike"], NaN);
    direction = string(localGetField(point, ["direction","Direction"], "PAY"));
    notional = localGetField(point, ["notional","Notional"], 1.0);

    if isnumeric(paymentDates) && all(isfinite(paymentDates))
        lastPayment = max(paymentDates(:));
        tenorYr = lastPayment - expiry;
        nPayments = numel(paymentDates);
    else
        lastPayment = NaN;
        tenorYr = NaN;
        nPayments = NaN;
    end

    aVal = localGetField(calibOut, ["a","meanReversion","meanReversion_x","aStar"], NaN);
    sigmaVal = localGetField(calibOut, ["sigma","vol","volatility","sigmaStar"], NaN);

    if isnan(sigmaVal)
        sigmaVec = localGetField(calibOut, ["volatility_sigma","sigmaBuckets"], NaN);
        sigmaVal = localExtractLastFinite(sigmaVec);
    end

    residuals = localGetField(calibOut, ["residuals","errors","calibrationErrors"], NaN);
    rmseVal = localGetField(calibOut, ["rmse","RMSE"], NaN);

    if isnan(rmseVal) && isnumeric(residuals)
        rmseVal = sqrt(mean(residuals(:).^2, 'omitnan'));
    end

    maxAbsResidual = localGetField(calibOut, ["maxAbsResidual","maxAbsError","maxError"], NaN);

    if isnan(maxAbsResidual) && isnumeric(residuals)
        maxAbsResidual = max(abs(residuals(:)), [], 'omitnan');
    end

    exitflag = localGetField(calibOut, ["exitflag","ExitFlag"], defaultExitFlag);

    modelPrice = localGetField(calibOut, ["modelPrice","priceModel"], NaN);
    marketPrice = localGetField(calibOut, ["marketPrice","priceMarket"], NaN);

    volTimeStructure = localGetField(calibOut, ["volatilityTimeStructure"], G2PP.volatilityTimeStructure);
    volTimeStructureDt = localGetField(calibOut, ["volatilityTimeStructureDt"], G2PP.volatilityTimeStructureDt);

    rec.Row = row;
    rec.Col = col;

    rec.BucketIdx = bucketIdx;
    rec.TenorIdx = tenorIdx;

    rec.ExpiryYr = expiry;
    rec.TenorYr = tenorYr;
    rec.LastPaymentYr = lastPayment;
    rec.NPayments = nPayments;

    rec.BucketRight = expiry;

    rec.Strike = strike;
    rec.Direction = direction;
    rec.Notional = notional;

    rec.a = aVal;
    rec.sigma = sigmaVal;

    rec.rmse = rmseVal;
    rec.maxAbsResidual = maxAbsResidual;
    rec.exitflag = exitflag;

    rec.nQuotes = localGetNQuotes(residuals);

    rec.modelPrice = modelPrice;
    rec.marketPrice = marketPrice;

    rec.volatilityTimeStructure = {volTimeStructure};
    rec.volatilityTimeStructureDt = {volTimeStructureDt};

    rec.errorMessage = "";
end
```

---

## 5. Empty record for failed blocks

```matlab
function rec = emptyBootTenorRecordG1PP(row, col, bucketIdx, tenorIdx)

    rec.Row = row;
    rec.Col = col;

    rec.BucketIdx = bucketIdx;
    rec.TenorIdx = tenorIdx;

    rec.ExpiryYr = NaN;
    rec.TenorYr = NaN;
    rec.LastPaymentYr = NaN;
    rec.NPayments = NaN;

    rec.BucketRight = NaN;

    rec.Strike = NaN;
    rec.Direction = "";
    rec.Notional = NaN;

    rec.a = NaN;
    rec.sigma = NaN;

    rec.rmse = NaN;
    rec.maxAbsResidual = NaN;
    rec.exitflag = NaN;

    rec.nQuotes = 0;

    rec.modelPrice = NaN;
    rec.marketPrice = NaN;

    rec.volatilityTimeStructure = {NaN};
    rec.volatilityTimeStructureDt = {NaN};

    rec.errorMessage = "";
end
```

---

## 6. Utility helpers

```matlab
function value = localGetField(S, names, defaultValue)
    % Robust field extraction from struct-like calibration outputs.

    value = defaultValue;

    if ~isstruct(S)
        return;
    end

    for k = 1:numel(names)
        name = char(names(k));

        if isfield(S, name)
            value = S.(name);
            return;
        end
    end
end

function x = localExtractLastFinite(v)
    % Extracts last finite scalar from a vector-like object.

    x = NaN;

    if ~isnumeric(v)
        return;
    end

    v = v(:);
    v = v(isfinite(v));

    if ~isempty(v)
        x = v(end);
    end
end

function n = localGetNQuotes(residuals)

    if isnumeric(residuals)
        n = sum(isfinite(residuals(:)));
    else
        n = 0;
    end
end
```

---

## 7. Minimal validation after table construction

```matlab
%% Validate bootTenorResults table

requiredVars = [
    "BucketIdx"
    "TenorYr"
    "BucketRight"
    "a"
    "sigma"
    "rmse"
    "maxAbsResidual"
    "exitflag"
    "nQuotes"
];

missingVars = setdiff(requiredVars, string(bootTenorResults.Properties.VariableNames));
assert(isempty(missingVars), "Missing variables: " + strjoin(missingVars, ", "));

bootTenorResults.finiteOK = isfinite(bootTenorResults.a) & ...
                             isfinite(bootTenorResults.sigma) & ...
                             isfinite(bootTenorResults.rmse);

bootTenorResults.admissibleOK = bootTenorResults.a > 0 & ...
                                bootTenorResults.sigma > 0;

bootTenorResults.optimizerOK = bootTenorResults.exitflag > 0 | ...
                               isnan(bootTenorResults.exitflag);

bootTenorResults.dataOK = bootTenorResults.nQuotes > 0 | ...
                          isnan(bootTenorResults.rmse);

bootTenorResults.blockOK = bootTenorResults.finiteOK & ...
                           bootTenorResults.admissibleOK & ...
                           bootTenorResults.optimizerOK;

disp(bootTenorResults(:, ...
    {'Row','Col','BucketIdx','TenorYr','a','sigma','rmse','maxAbsResidual','exitflag','blockOK','errorMessage'}));
```

---

## 8. Then run the previous diagnostic block

Once this table exists, the earlier diagnostic code works directly with:

```matlab
R = bootTenorResults;
```

The important design point is this: do **not** let plotting code call the calibration again. Calibration produces the ledger; diagnostics consume the ledger. That separation protects you from mutable-object side effects and from optimizer randomness.

## Response:
5/26/2026, 10:20:20 AM

> **Clarifying citation syntax and formats**
>
> I need to provide a brief answer on the organization without getting into code. Files need citations, but if the mclick return is one blob without line markers, I might focus on page references for PDFs. For mclick content, I'll use general file citations, and for snippets, I'll note details like loops, seeds, and calibration.
>
> **Clarifying citation details for file and PDF**
>
> Okay, I need to include file citations when using file_search, but the initial result mentions filecite without line markers. For PDFs, I might cite the file itself in a general way without adding specific line references. Snippets from pasted text are easier to cite, but for PDFs, I'll mention pages instead.
>
> Thought for 16s

Yes. Reset the structure. The test should not be “one huge diagnostic script.” It should be **two consolidated MLX blocks** with shared knobs at the top and almost no helper sprawl.

## 1. Context read

From your snippet, the intended workflow is already clear:

You test several pricing dates, load curves and swaption market data per date, rebuild `G2PP`, build a calibration surface pack, run global G1++, then run two bootstrap slices: short-tenor `sigma` and long-tenor `eta`. The snippet also stores shapes and time structures across dates, and reuses previous shapes as seeds. `Pasted text.txt` `Pasted text.txt`

Important correction: **mean reversion is no longer a stability object.** It is now an exogenous knob. So medium-term stability should focus on:

$$
\sigma^{global}_d,
\qquad
\sigma^{short}_{d,k},
\qquad
\eta^{long}_{d,k},
\qquad
\text{pricing errors}.
$$

Not on $a_d$.

Your current snippet still calls `calibrateGlobalG1PP` with an initial `A`, but for the test framework we should treat $a$ as fixed input, not a calibrated output. `Pasted text.txt`

---

# 2. Proposed organization

## Block A - Global G1++ stability test

**Object.**

$$
d \mapsto \sigma^{global}_d
$$

where $d$ is the pricing date.

**Question.**

Does the global G1++ calibration produce a stable volatility level across dates?

**Knobs.**

At the top:

```matlab
testGlobal = true;
pricingDates = {...};

fixedA = 0.05;
sigmaSeed0 = sigma0;

objectiveType = 'normalizedPrice';
useLsqnonlin = true;

plotDiagnostics = true;
verbose = true;
```

**Diagnostics.**

For each date:

$$
\sigma_d^{global},
\qquad
\text{mean error}_d,
\qquad
\text{max error}_d,
\qquad
\text{success flag}_d.
$$

Then visually:

1. line plot of $\sigma^{global}_d$;
2. line plot of mean/max normalized error;
3. scatter of error versus $\sigma^{global}_d$;
4. table of failed dates.

**Mathematical interpretation.**

If $a$ is fixed, the only global degree of freedom is the volatility scale. So instability in $\sigma^{global}$ across nearby market dates means one of three things:

$$
\text{market surface moved},\quad
\text{objective is ill-conditioned},\quad
\text{engine/surface construction is inconsistent}.
$$

This is the clean first test.

---

## Block B - Bootstrapped-tenor shape stability test

**Object.**

For each pricing date $d$:

$$
\sigma^{short}_{d,k},
\qquad
\eta^{long}_{d,k},
$$

where $k$ is the volatility bucket index.

Your snippet already stores these as:

```matlab
shapeStore{d,1} = sigmaShapePack.shape;
shapeStore{d,2} = etaShapePack.shape;
timeStructStore{d,1} = sigmaShapePack.timeStructure;
timeStructStore{d,2} = etaShapePack.timeStructure;
```

`Pasted text.txt`

**Question.**

Do the bootstrapped shapes evolve smoothly across dates and across buckets?

**Knobs.**

```matlab
testBootstrap = true;

runShortTenor = true;
runLongTenor = true;

shortTenorIdx = initCalibXG2.shortTenorIdx;
longTenorIdx  = initCalibXG2.longTenorIdx;

terminalPolicy = 'flatLast';
usePreviousShapeAsSeed = true;

mediumBucketRange = [];     % empty = all buckets
jumpThresholdLog = 0.35;    % relative jump threshold
```

**Diagnostics.**

For each date and shape:

$$
\operatorname{median}(\sigma_{d,\cdot}),
\qquad
\max_k |\Delta_k \log \sigma_{d,k}|,
\qquad
\max_k |\Delta_d \log \sigma_{d,k}|,
\qquad
\text{mean/max pricing error}.
$$

Same for $\eta$.

Visuals:

1. shape paths by date;
2. heatmap of $\sigma^{short}_{d,k}$;
3. heatmap of $\eta^{long}_{d,k}$;
4. heatmap of date-to-date log jumps;
5. error paths for short and long bootstrap.

**Mathematical interpretation.**

The bootstrap estimates bucketed volatility contributions. It should not produce random saw-tooth patterns unless the market data truly implies them. The key stability objects are:

$$
\Delta_k \log \sigma_{d,k}
=
\log\sigma_{d,k+1}-\log\sigma_{d,k},
$$

and

$$
\Delta_d \log \sigma_{d,k}
=
\log\sigma_{d+1,k}-\log\sigma_{d,k}.
$$

Same for $\eta$.

The first checks **shape smoothness across buckets**.  
The second checks **medium-term stability across pricing dates**.

---

# 3. Table design

Do not build a large table with every internal field. Use two tables.

## `globalTestResults`

One row per date:

```matlab
Date
SigmaGlobal
MeanError
MaxError
Success
Message
```

## `bootstrapTestResults`

One row per date and shape type:

```matlab
Date
ShapeName          % 'sigma' or 'eta'
TenorRole          % 'short' or 'long'
MedianShape
MaxBucketLogJump
MaxDateLogJump
MeanError
MaxError
Success
Message
```

The full shapes stay in cell arrays:

```matlab
shapeStore
timeStructStore
```

That keeps the tables readable.

---

# 4. Why this is cleaner

The previous idea, one row per bucket/tenor/block, is useful later for deep diagnostics. But for your current MLX test it is too granular.

Right now the purpose is:

$$
\boxed{
\text{test stability of the global and bootstrapped calibration engines across dates}
}
$$

not:

$$
\text{audit every single optimizer block}.
$$

So we use:

```matlab
globalTestResults
bootstrapTestResults
shapeStore
timeStructStore
```

instead of a giant ledger.

---

# 5. Final proposed MLX skeleton

The script should have this shape:

```matlab
%% 0. Test knobs

%% 1. Storage allocation

%% 2. Loop over pricing dates
% 2.1 Load market data
% 2.2 Build G2PP with fixed mean reversion
% 2.3 Build calibVolSurface
% 2.4 Global G1++ calibration
% 2.5 Bootstrap short sigma shape
% 2.6 Bootstrap long eta shape
% 2.7 Store compact diagnostics
% 2.8 Update seeds

%% 3. Global diagnostics plots

%% 4. Bootstrap diagnostics plots

%% 5. Summary flags
```

No more than two small local helpers:

```matlab
computeRelativeErrors(modelPrices, marketPrices)
computeShapeJumpStats(shapeMat)
```

Everything else should remain visible in the MLX.

This is the right compromise: structured enough to be systematic, but not buried under helper functions.

## Response:
5/26/2026, 10:24:50 AM

Yes, but only for the **bootstrapped block**, not as the main global test.

For the bootstrap, fit-vs-stability is the right final diagnostic because your MLX already stores both objects:

$$
\text{fit error}_{d,\ell}
\quad\text{and}\quad
\text{shape}_{d,\ell},
$$

where $d$ is the pricing date and $\ell\in\{\text{short/sigma},\text{long/eta}\}$. Your snippet stores the bootstrapped shapes in `shapeStore{d,1}` and `shapeStore{d,2}`, and also computes mean/max errors when `modelPrices` and `marketPrices` are available. `Pasted text.txt`

## Proposed inclusion

For each date and each bootstrap slice:

$$
x_{d,\ell}
=
\max_k
\left|
\log q_{d,\ell,k+1}
-
\log q_{d,\ell,k}
\right|,
$$

where $q=\sigma$ for short-tenor bootstrap and $q=\eta$ for long-tenor bootstrap.

This is the **shape instability score** across buckets.

Then plot against:

$$
y_{d,\ell}
=
\text{mean normalized pricing error}_{d,\ell},
$$

or alternatively:

$$
y_{d,\ell}
=
\text{max normalized pricing error}_{d,\ell}.
$$

So the diagnostic is:

$$
\boxed{
\text{fit error}
\quad\text{vs}\quad
\text{shape instability}
}
$$

## Interpretation

| Region | Meaning |
|---|---|
| low error, low instability | good bootstrap |
| low error, high instability | overfitting / unstable local shape |
| high error, low instability | model cannot fit that slice |
| high error, high instability | ill-conditioned block |

This is exactly the medium-term question: not only whether the bootstrap fits, but whether it fits with a stable volatility shape.

## Where it goes

In the MLX structure:

```matlab
%% 0. Test knobs

%% 1. Global G1++ stability block

%% 2. Bootstrapped-tenor stability block

%% 3. Bootstrap fit-vs-stability diagnostic
```

But I would implement it inside the **bootstrapped block**, after the shape heatmaps.

## Knobs to include

```matlab
includeFitVsStability = true;

fitMetric = 'mean';      % 'mean' or 'max'
stabilityMetric = 'maxBucketLogJump';  % simple first version

jumpEps = 1e-12;
```

## Minimal data needed

For each date and slice:

```matlab
bootstrapTestResults.Date
bootstrapTestResults.ShapeName      % 'sigma' or 'eta'
bootstrapTestResults.TenorRole      % 'short' or 'long'
bootstrapTestResults.MedianShape
bootstrapTestResults.MaxBucketLogJump
bootstrapTestResults.MeanError
bootstrapTestResults.MaxError
bootstrapTestResults.Success
```

Then the plot is just:

```matlab
scatter(MaxBucketLogJump, MeanError)
```

with different markers for `sigma` and `eta`.

So yes: include it. It is the cleanest bridge between **calibration quality** and **parameter stability**.

## Response:
5/26/2026, 10:25:24 AM

> **Providing MATLAB code with relevant citations**
>
> I'll provide the full MATLAB code as requested, including the two consolidated blocks: global and bootstrapped, while avoiding helpers. I'll ensure the code reflects the "mean reversion no longer calibrated" request and may cite relevant file lines from previous searches where necessary. I'll infer function signatures from the available context and handle name discrepancies in the file.
>
> **Building loop for G2PP model calibration**
>
> I’ll loop over dates and load curves/volatilities. The line for building G2PP uses discountCurve twice, but I might want to switch it to forecastCurve—I'll parameterize that choice. I’ll also define scopes like quoteRows/Cols, ensuring no alteration of the surfacePack object. Lastly, I’ll ensure the shapeMat uses NaN padding where needed.
>
> Thought for 45s

Below is a **single MLX-ready script** with two consolidated blocks:

1. **Global G1++ stability**
2. **Bootstrapped short/long shape stability**, including **fit vs stability**

I assume the calibration packs expose `.shape`, `.timeStructure`, and optionally `.modelPrices`, `.marketPrices`, as in your snippet for `sigmaShapePack` and `etaShapePack`. Your uploaded snippet already follows this flow: build model per date, run `calibrateGlobalG1PP`, then run `bootstrapSliceG1PP` for short `sigma` and long `eta`. `Pasted text.txt`

```matlab
%% ========================================================
%  G1++ GLOBAL + BOOTSTRAPPED SHAPE STABILITY TEST
%  ========================================================
%
%  Purpose:
%    1. Test global G1++ calibration stability across pricing dates.
%    2. Test bootstrapped short-tenor sigma and long-tenor eta shape stability.
%    3. Compare fit quality against shape instability.
%
%  Convention:
%    Mean reversion is fixed. It is not a calibrated stability object here.
%
%  Required workspace:
%    - sigma0, eta0, b0, rho0, basis
%    - pricers
%    - initCalibXG2.shortTenorIdx
%    - initCalibXG2.longTenorIdx
%
%  Required model methods:
%    - GaussianTwoFactors(...)
%    - buildVolSurfacePackXG2PP(...)
%    - calibrateGlobalG1PP(...)
%    - bootstrapSliceG1PP(...)

clearvars -except sigma0 eta0 a0 b0 rho0 basis pricers initCalibXG2
clc;

%% ========================================================
%  0. TEST KNOBS
%  ========================================================

% --------------------------
% Date scope
% --------------------------
pricingDates = { ...
    '03/03/2026', ...
    '06/03/2026', ...
    '10/03/2026', ...
    '13/03/2026', ...
    '17/03/2026', ...
    '20/03/2026', ...
    '24/03/2026', ...
    '27/03/2026', ...
    '31/03/2026', ...
    '02/04/2026', ...
    '07/04/2026', ...
    '10/04/2026'};

nDates = numel(pricingDates);

% --------------------------
% Data source knobs
% --------------------------
discountCurveName = 'PNL_EUR_ESTR';
forecastCurveName = 'PNL_EUR_E6M';
swaptionVolName   = 'PNL_EUR_SVOL_E6M_ICAP';

useSingleCurveForFirstTest = true;

% --------------------------
% Fixed model parameters
% --------------------------
fixedA = 0.05;       % fixed G1++ mean reversion
fixedB = b0;         % kept for G2PP constructor compatibility
fixedRho = rho0;     % kept for object consistency

sigmaSeed0 = sigma0;
etaSeed0   = eta0;

% --------------------------
% Surface scope knobs
% Empty = use default quoteRows / quoteCols from buildVolSurfacePackXG2PP.
% Otherwise set explicit rows/cols, e.g. quoteRowsOverride = [1 2 3].
% --------------------------
quoteRowsOverride = [];
quoteColsOverride = [];

% --------------------------
% Global calibration knobs
% --------------------------
runGlobalBlock = true;
useLsqnonlinGlobal = true;
objectiveTypeGlobal = 'normalizedPrice';

% Extra global args, only if your method supports them.
globalExtraArgs = {};

% --------------------------
% Bootstrap knobs
% --------------------------
runBootstrapBlock = true;

runShortSigmaBootstrap = true;
runLongEtaBootstrap    = true;

shortTenorIdx = initCalibXG2.shortTenorIdx;
longTenorIdx  = initCalibXG2.longTenorIdx;

objectiveTypeBootstrap = 'normalizedPrice';
terminalPolicy = 'flatLast';

usePreviousShapeAsSeed = true;

% Extra bootstrap args, only if your method supports them.
bootstrapExtraArgs = {};

% --------------------------
% Diagnostics knobs
% --------------------------
priceErrorDenomFloor = 1e-12;
shapeLogFloor = 1e-12;

bucketJumpWarningLevel = 0.35;  % log jump: approx 42% relative move
dateJumpWarningLevel   = 0.35;

includePlots = true;
verbose = true;

%% ========================================================
%  1. STORAGE
%  ========================================================

dateDnum = NaN(nDates,1);

globalTestResults = table( ...
    strings(nDates,1), ...   % Date
    NaN(nDates,1), ...       % DateDnum
    NaN(nDates,1), ...       % FixedA
    NaN(nDates,1), ...       % SigmaGlobal
    NaN(nDates,1), ...       % MeanError
    NaN(nDates,1), ...       % MaxError
    false(nDates,1), ...     % Success
    strings(nDates,1), ...   % Message
    'VariableNames', { ...
        'Date','DateDnum','FixedA','SigmaGlobal', ...
        'MeanError','MaxError','Success','Message'});

nBootstrapRows = nDates * 2;

bootstrapTestResults = table( ...
    strings(nBootstrapRows,1), ...   % Date
    NaN(nBootstrapRows,1), ...       % DateDnum
    strings(nBootstrapRows,1), ...   % ShapeName
    strings(nBootstrapRows,1), ...   % TenorRole
    NaN(nBootstrapRows,1), ...       % TenorIdx
    NaN(nBootstrapRows,1), ...       % MedianShape
    NaN(nBootstrapRows,1), ...       % MinShape
    NaN(nBootstrapRows,1), ...       % MaxShape
    NaN(nBootstrapRows,1), ...       % MaxBucketLogJump
    NaN(nBootstrapRows,1), ...       % MaxDateLogJump
    NaN(nBootstrapRows,1), ...       % MeanError
    NaN(nBootstrapRows,1), ...       % MaxError
    false(nBootstrapRows,1), ...     % Success
    strings(nBootstrapRows,1), ...   % Message
    'VariableNames', { ...
        'Date','DateDnum','ShapeName','TenorRole','TenorIdx', ...
        'MedianShape','MinShape','MaxShape', ...
        'MaxBucketLogJump','MaxDateLogJump', ...
        'MeanError','MaxError','Success','Message'});

shapeStore = cell(nDates,2);       % col 1 = sigma short, col 2 = eta long
timeStructStore = cell(nDates,2);  % col 1 = sigma short, col 2 = eta long

seedShort = [];
seedLong  = [];

sigmaSeed = sigmaSeed0;
etaSeed   = etaSeed0;

bootRow = 0;

%% ========================================================
%  2. MAIN LOOP OVER PRICING DATES
%  ========================================================

for d = 1:nDates

    pricingDateChar = pricingDates{d};
    dateDnum(d) = datenum(pricingDateChar, 'dd/mm/yyyy');

    if verbose
        fprintf('\n====================================================\n');
        fprintf('Pricing date %d/%d: %s\n', d, nDates, pricingDateChar);
        fprintf('====================================================\n');
    end

    globalTestResults.Date(d) = string(pricingDateChar);
    globalTestResults.DateDnum(d) = dateDnum(d);
    globalTestResults.FixedA(d) = fixedA;

    %% 2.1 Load market data

    try
        discountCurve = LoaderDataBase( ...
            pricingDateChar, 'Rate', discountCurveName).data;

        forecastCurve = LoaderDataBase( ...
            pricingDateChar, 'Rate', forecastCurveName).data;

        vol_swaption = LoaderDataBase( ...
            pricingDateChar, 'SwaptionVol', swaptionVolName).data;

        if useSingleCurveForFirstTest
            forecastCurveForModel = discountCurve;
        else
            forecastCurveForModel = forecastCurve;
        end

    catch ME
        globalTestResults.Message(d) = "Market data loading failed: " + string(ME.message);

        if verbose
            warning('%s', globalTestResults.Message(d));
        end

        bootRow = bootRow + 1;
        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "sigma";
        bootstrapTestResults.TenorRole(bootRow) = "short";
        bootstrapTestResults.TenorIdx(bootRow) = shortTenorIdx;
        bootstrapTestResults.Message(bootRow) = "Skipped: market data loading failed.";

        bootRow = bootRow + 1;
        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "eta";
        bootstrapTestResults.TenorRole(bootRow) = "long";
        bootstrapTestResults.TenorIdx(bootRow) = longTenorIdx;
        bootstrapTestResults.Message(bootRow) = "Skipped: market data loading failed.";

        continue;
    end

    %% 2.2 Build model with fixed mean reversion

    try
        G2PP = GaussianTwoFactors( ...
            pricingDateChar, ...
            fixedA, ...
            fixedB, ...
            discountCurve, ...
            forecastCurveForModel, ...
            'swaptionVol', vol_swaption, ...
            'basis', basis);

        G2PP.correlation = fixedRho;

        calibVolSurface = G2PP.buildVolSurfacePackXG2PP(0);

        % Optional surface scope restriction.
        if ~isempty(quoteRowsOverride) && isfield(calibVolSurface, 'quoteRows')
            calibVolSurface.quoteRows = quoteRowsOverride(:).';
        end

        if ~isempty(quoteColsOverride) && isfield(calibVolSurface, 'quoteCols')
            calibVolSurface.quoteCols = quoteColsOverride(:).';
        end

    catch ME
        globalTestResults.Message(d) = "Model/surface construction failed: " + string(ME.message);

        if verbose
            warning('%s', globalTestResults.Message(d));
        end

        bootRow = bootRow + 1;
        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "sigma";
        bootstrapTestResults.TenorRole(bootRow) = "short";
        bootstrapTestResults.TenorIdx(bootRow) = shortTenorIdx;
        bootstrapTestResults.Message(bootRow) = "Skipped: model/surface construction failed.";

        bootRow = bootRow + 1;
        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "eta";
        bootstrapTestResults.TenorRole(bootRow) = "long";
        bootstrapTestResults.TenorIdx(bootRow) = longTenorIdx;
        bootstrapTestResults.Message(bootRow) = "Skipped: model/surface construction failed.";

        continue;
    end

    %% ====================================================
    %  BLOCK A - GLOBAL G1++ STABILITY
    %  ====================================================

    g1Global = struct;
    g1Global.a = fixedA;
    g1Global.sigma = sigmaSeed;

    if runGlobalBlock

        try
            g1Global = G2PP.calibrateGlobalG1PP( ...
                calibVolSurface, pricers, ...
                'initialA', fixedA, ...
                'initialSigma', sigmaSeed, ...
                'objectiveType', objectiveTypeGlobal, ...
                'useLsqnonlin', useLsqnonlinGlobal, ...
                globalExtraArgs{:});

            % Mean reversion is fixed by test convention.
            g1Global.a = fixedA;
            G2PP.meanReversion_x = fixedA;

            if isfield(g1Global, 'sigma')
                sigmaGlobal = g1Global.sigma;
            else
                sigmaGlobal = sigmaSeed;
            end

            globalTestResults.SigmaGlobal(d) = sigmaGlobal;

            % Fit diagnostics if prices are available.
            meanErrGlobal = NaN;
            maxErrGlobal = NaN;

            if isfield(g1Global, 'modelPrices') && isfield(g1Global, 'marketPrices')

                modelPrices = g1Global.modelPrices(:);
                marketPrices = g1Global.marketPrices(:);

                denom = max(abs(marketPrices), priceErrorDenomFloor);
                errs = abs(modelPrices - marketPrices) ./ denom;
                errs = errs(isfinite(errs));

                if ~isempty(errs)
                    meanErrGlobal = mean(errs);
                    maxErrGlobal = max(errs);
                end

            elseif isfield(g1Global, 'residuals')

                errs = abs(g1Global.residuals(:));
                errs = errs(isfinite(errs));

                if ~isempty(errs)
                    meanErrGlobal = mean(errs);
                    maxErrGlobal = max(errs);
                end
            end

            globalTestResults.MeanError(d) = meanErrGlobal;
            globalTestResults.MaxError(d) = maxErrGlobal;
            globalTestResults.Success(d) = isfinite(sigmaGlobal) && sigmaGlobal > 0;
            globalTestResults.Message(d) = "OK";

            sigmaSeed = sigmaGlobal;

            if verbose
                fprintf('Global G1++ OK | sigma = %.8f | meanErr = %.3e | maxErr = %.3e\n', ...
                    sigmaGlobal, meanErrGlobal, maxErrGlobal);
            end

        catch ME

            globalTestResults.SigmaGlobal(d) = sigmaSeed;
            globalTestResults.MeanError(d) = NaN;
            globalTestResults.MaxError(d) = NaN;
            globalTestResults.Success(d) = false;
            globalTestResults.Message(d) = "Global calibration failed: " + string(ME.message);

            g1Global = struct;
            g1Global.a = fixedA;
            g1Global.sigma = sigmaSeed;

            if verbose
                warning('%s', globalTestResults.Message(d));
            end
        end
    end

    %% ====================================================
    %  BLOCK B - BOOTSTRAPPED SHORT/LONG SHAPE STABILITY
    %  ====================================================

    if runBootstrapBlock

        %% 2.4 Short-tenor sigma bootstrap

        bootRow = bootRow + 1;

        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "sigma";
        bootstrapTestResults.TenorRole(bootRow) = "short";
        bootstrapTestResults.TenorIdx(bootRow) = shortTenorIdx;

        if runShortSigmaBootstrap

            try
                optsShort = { ...
                    'tenorIdx', shortTenorIdx, ...
                    'shapeName', 'sigma', ...
                    'objectiveType', objectiveTypeBootstrap, ...
                    'terminalPolicy', terminalPolicy, ...
                    bootstrapExtraArgs{:}};

                if usePreviousShapeAsSeed && ~isempty(seedShort)
                    optsShort = [optsShort, {'initialShape', seedShort}];
                end

                sigmaShapePack = G2PP.bootstrapSliceG1PP( ...
                    calibVolSurface, pricers, g1Global, optsShort{:});

                sigmaShape = sigmaShapePack.shape(:).';

                shapeStore{d,1} = sigmaShape;

                if isfield(sigmaShapePack, 'timeStructure')
                    timeStructStore{d,1} = sigmaShapePack.timeStructure;
                else
                    timeStructStore{d,1} = NaN;
                end

                logShape = log(max(sigmaShape, shapeLogFloor));

                if numel(logShape) >= 2
                    maxBucketJump = max(abs(diff(logShape)));
                else
                    maxBucketJump = NaN;
                end

                if ~isempty(seedShort) && numel(seedShort) == numel(sigmaShape)
                    maxDateJump = max(abs(logShape - log(max(seedShort(:).', shapeLogFloor))));
                else
                    maxDateJump = NaN;
                end

                meanErrShort = NaN;
                maxErrShort = NaN;

                if isfield(sigmaShapePack, 'modelPrices') && isfield(sigmaShapePack, 'marketPrices')

                    modelPrices = sigmaShapePack.modelPrices(:);
                    marketPrices = sigmaShapePack.marketPrices(:);

                    denom = max(abs(marketPrices), priceErrorDenomFloor);
                    errs = abs(modelPrices - marketPrices) ./ denom;
                    errs = errs(isfinite(errs));

                    if ~isempty(errs)
                        meanErrShort = mean(errs);
                        maxErrShort = max(errs);
                    end

                elseif isfield(sigmaShapePack, 'residuals')

                    errs = abs(sigmaShapePack.residuals(:));
                    errs = errs(isfinite(errs));

                    if ~isempty(errs)
                        meanErrShort = mean(errs);
                        maxErrShort = max(errs);
                    end
                end

                bootstrapTestResults.MedianShape(bootRow) = median(sigmaShape, 'omitnan');
                bootstrapTestResults.MinShape(bootRow) = min(sigmaShape, [], 'omitnan');
                bootstrapTestResults.MaxShape(bootRow) = max(sigmaShape, [], 'omitnan');
                bootstrapTestResults.MaxBucketLogJump(bootRow) = maxBucketJump;
                bootstrapTestResults.MaxDateLogJump(bootRow) = maxDateJump;
                bootstrapTestResults.MeanError(bootRow) = meanErrShort;
                bootstrapTestResults.MaxError(bootRow) = maxErrShort;
                bootstrapTestResults.Success(bootRow) = all(isfinite(sigmaShape)) && all(sigmaShape > 0);
                bootstrapTestResults.Message(bootRow) = "OK";

                seedShort = sigmaShape;

                if verbose
                    fprintf('Short sigma bootstrap OK | median = %.8f | max bucket jump = %.3f | meanErr = %.3e\n', ...
                        bootstrapTestResults.MedianShape(bootRow), maxBucketJump, meanErrShort);
                end

            catch ME

                bootstrapTestResults.Success(bootRow) = false;
                bootstrapTestResults.Message(bootRow) = "Short sigma bootstrap failed: " + string(ME.message);

                if verbose
                    warning('%s', bootstrapTestResults.Message(bootRow));
                end
            end
        else
            bootstrapTestResults.Message(bootRow) = "Skipped by knob.";
        end

        %% 2.5 Long-tenor eta bootstrap

        bootRow = bootRow + 1;

        bootstrapTestResults.Date(bootRow) = string(pricingDateChar);
        bootstrapTestResults.DateDnum(bootRow) = dateDnum(d);
        bootstrapTestResults.ShapeName(bootRow) = "eta";
        bootstrapTestResults.TenorRole(bootRow) = "long";
        bootstrapTestResults.TenorIdx(bootRow) = longTenorIdx;

        if runLongEtaBootstrap

            try
                optsLong = { ...
                    'tenorIdx', longTenorIdx, ...
                    'shapeName', 'eta', ...
                    'objectiveType', objectiveTypeBootstrap, ...
                    'terminalPolicy', terminalPolicy, ...
                    bootstrapExtraArgs{:}};

                if usePreviousShapeAsSeed && ~isempty(seedLong)
                    optsLong = [optsLong, {'initialShape', seedLong}];
                end

                etaShapePack = G2PP.bootstrapSliceG1PP( ...
                    calibVolSurface, pricers, g1Global, optsLong{:});

                etaShape = etaShapePack.shape(:).';

                shapeStore{d,2} = etaShape;

                if isfield(etaShapePack, 'timeStructure')
                    timeStructStore{d,2} = etaShapePack.timeStructure;
                else
                    timeStructStore{d,2} = NaN;
                end

                logShape = log(max(etaShape, shapeLogFloor));

                if numel(logShape) >= 2
                    maxBucketJump = max(abs(diff(logShape)));
                else
                    maxBucketJump = NaN;
                end

                if ~isempty(seedLong) && numel(seedLong) == numel(etaShape)
                    maxDateJump = max(abs(logShape - log(max(seedLong(:).', shapeLogFloor))));
                else
                    maxDateJump = NaN;
                end

                meanErrLong = NaN;
                maxErrLong = NaN;

                if isfield(etaShapePack, 'modelPrices') && isfield(etaShapePack, 'marketPrices')

                    modelPrices = etaShapePack.modelPrices(:);
                    marketPrices = etaShapePack.marketPrices(:);

                    denom = max(abs(marketPrices), priceErrorDenomFloor);
                    errs = abs(modelPrices - marketPrices) ./ denom;
                    errs = errs(isfinite(errs));

                    if ~isempty(errs)
                        meanErrLong = mean(errs);
                        maxErrLong = max(errs);
                    end

                elseif isfield(etaShapePack, 'residuals')

                    errs = abs(etaShapePack.residuals(:));
                    errs = errs(isfinite(errs));

                    if ~isempty(errs)
                        meanErrLong = mean(errs);
                        maxErrLong = max(errs);
                    end
                end

                bootstrapTestResults.MedianShape(bootRow) = median(etaShape, 'omitnan');
                bootstrapTestResults.MinShape(bootRow) = min(etaShape, [], 'omitnan');
                bootstrapTestResults.MaxShape(bootRow) = max(etaShape, [], 'omitnan');
                bootstrapTestResults.MaxBucketLogJump(bootRow) = maxBucketJump;
                bootstrapTestResults.MaxDateLogJump(bootRow) = maxDateJump;
                bootstrapTestResults.MeanError(bootRow) = meanErrLong;
                bootstrapTestResults.MaxError(bootRow) = maxErrLong;
                bootstrapTestResults.Success(bootRow) = all(isfinite(etaShape)) && all(etaShape > 0);
                bootstrapTestResults.Message(bootRow) = "OK";

                seedLong = etaShape;

                if verbose
                    fprintf('Long eta bootstrap OK | median = %.8f | max bucket jump = %.3f | meanErr = %.3e\n', ...
                        bootstrapTestResults.MedianShape(bootRow), maxBucketJump, meanErrLong);
                end

            catch ME

                bootstrapTestResults.Success(bootRow) = false;
                bootstrapTestResults.Message(bootRow) = "Long eta bootstrap failed: " + string(ME.message);

                if verbose
                    warning('%s', bootstrapTestResults.Message(bootRow));
                end
            end
        else
            bootstrapTestResults.Message(bootRow) = "Skipped by knob.";
        end
    end
end

% Remove unused preallocated rows if some were not filled.
bootstrapTestResults = bootstrapTestResults(1:bootRow,:);

%% ========================================================
%  3. HARD FLAGS AND SUMMARY TABLES
%  ========================================================

globalTestResults.GlobalOK = ...
    globalTestResults.Success & ...
    isfinite(globalTestResults.SigmaGlobal) & ...
    globalTestResults.SigmaGlobal > 0;

bootstrapTestResults.ShapeOK = ...
    bootstrapTestResults.Success & ...
    isfinite(bootstrapTestResults.MedianShape) & ...
    bootstrapTestResults.MedianShape > 0;

bootstrapTestResults.BucketJumpOK = ...
    isnan(bootstrapTestResults.MaxBucketLogJump) | ...
    bootstrapTestResults.MaxBucketLogJump <= bucketJumpWarningLevel;

bootstrapTestResults.DateJumpOK = ...
    isnan(bootstrapTestResults.MaxDateLogJump) | ...
    bootstrapTestResults.MaxDateLogJump <= dateJumpWarningLevel;

bootstrapTestResults.BootstrapOK = ...
    bootstrapTestResults.ShapeOK & ...
    bootstrapTestResults.BucketJumpOK & ...
    bootstrapTestResults.DateJumpOK;

globalSummary = table( ...
    height(globalTestResults), ...
    sum(globalTestResults.GlobalOK), ...
    max(globalTestResults.MeanError, [], 'omitnan'), ...
    max(globalTestResults.MaxError, [], 'omitnan'), ...
    min(globalTestResults.SigmaGlobal, [], 'omitnan'), ...
    max(globalTestResults.SigmaGlobal, [], 'omitnan'), ...
    'VariableNames', { ...
        'NDates','NGlobalOK','WorstMeanError','WorstMaxError', ...
        'MinSigmaGlobal','MaxSigmaGlobal'});

bootstrapSummary = groupsummary( ...
    bootstrapTestResults, ...
    {'ShapeName','TenorRole'}, ...
    {'sum','max','mean'}, ...
    {'BootstrapOK','MaxBucketLogJump','MaxDateLogJump','MeanError','MaxError'});

disp('GLOBAL TEST RESULTS');
disp(globalTestResults);

disp('GLOBAL SUMMARY');
disp(globalSummary);

disp('BOOTSTRAP TEST RESULTS');
disp(bootstrapTestResults);

disp('BOOTSTRAP SUMMARY');
disp(bootstrapSummary);

badGlobal = globalTestResults(~globalTestResults.GlobalOK,:);
badBootstrap = bootstrapTestResults(~bootstrapTestResults.BootstrapOK,:);

disp('BAD GLOBAL DATES');
disp(badGlobal);

disp('BAD BOOTSTRAP BLOCKS');
disp(badBootstrap);

%% ========================================================
%  4. BUILD SHAPE MATRICES FOR VISUAL ANALYSIS
%  ========================================================

% Sigma short matrix
maxLenSigma = 0;
for d = 1:nDates
    if ~isempty(shapeStore{d,1})
        maxLenSigma = max(maxLenSigma, numel(shapeStore{d,1}));
    end
end

sigmaShapeMat = NaN(nDates, maxLenSigma);

for d = 1:nDates
    if ~isempty(shapeStore{d,1})
        x = shapeStore{d,1}(:).';
        sigmaShapeMat(d,1:numel(x)) = x;
    end
end

% Eta long matrix
maxLenEta = 0;
for d = 1:nDates
    if ~isempty(shapeStore{d,2})
        maxLenEta = max(maxLenEta, numel(shapeStore{d,2}));
    end
end

etaShapeMat = NaN(nDates, maxLenEta);

for d = 1:nDates
    if ~isempty(shapeStore{d,2})
        x = shapeStore{d,2}(:).';
        etaShapeMat(d,1:numel(x)) = x;
    end
end

% Log matrices and jump matrices
logSigmaShapeMat = log(max(sigmaShapeMat, shapeLogFloor));
logEtaShapeMat   = log(max(etaShapeMat, shapeLogFloor));

sigmaBucketJumpMat = abs(diff(logSigmaShapeMat, 1, 2));
etaBucketJumpMat   = abs(diff(logEtaShapeMat, 1, 2));

sigmaDateJumpMat = abs(diff(logSigmaShapeMat, 1, 1));
etaDateJumpMat   = abs(diff(logEtaShapeMat, 1, 1));

dateLabels = string(datestr(dateDnum, 'dd/mm'));
sigmaBucketLabels = compose('B%d', 1:maxLenSigma);
etaBucketLabels   = compose('B%d', 1:maxLenEta);

if maxLenSigma >= 2
    sigmaBucketJumpLabels = compose('B%d-B%d', 1:maxLenSigma-1, 2:maxLenSigma);
else
    sigmaBucketJumpLabels = "none";
end

if maxLenEta >= 2
    etaBucketJumpLabels = compose('B%d-B%d', 1:maxLenEta-1, 2:maxLenEta);
else
    etaBucketJumpLabels = "none";
end

if nDates >= 2
    dateJumpLabels = compose('%s-%s', dateLabels(1:end-1), dateLabels(2:end));
else
    dateJumpLabels = "none";
end

%% ========================================================
%  5. GLOBAL VISUAL DIAGNOSTICS
%  ========================================================

if includePlots

    figure('Name','Global G1++ Stability Diagnostics','Color','w');
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(dateDnum, globalTestResults.SigmaGlobal, '-o', 'LineWidth', 1.2);
    grid on;
    datetick('x','dd/mm','keepticks');
    xlabel('Pricing date');
    ylabel('\sigma^{global}');
    title('Global G1++ volatility across dates');

    nexttile;
    plot(dateDnum, globalTestResults.MeanError, '-o', 'LineWidth', 1.2);
    hold on;
    plot(dateDnum, globalTestResults.MaxError, '-s', 'LineWidth', 1.2);
    hold off;
    grid on;
    datetick('x','dd/mm','keepticks');
    xlabel('Pricing date');
    ylabel('Normalized error');
    title('Global calibration fit error');
    legend({'Mean error','Max error'}, 'Location','best');

    nexttile;
    scatter(globalTestResults.SigmaGlobal, globalTestResults.MeanError, 60, 'filled');
    grid on;
    xlabel('\sigma^{global}');
    ylabel('Mean normalized error');
    title('Global fit vs volatility level');

    nexttile;
    bar(double(globalTestResults.GlobalOK));
    grid on;
    ylim([0 1.2]);
    set(gca,'XTick',1:nDates,'XTickLabel',dateLabels);
    xtickangle(45);
    ylabel('OK flag');
    title('Global calibration OK');

end

%% ========================================================
%  6. BOOTSTRAP VISUAL DIAGNOSTICS
%  ========================================================

if includePlots

    figure('Name','Bootstrap Shape Stability Diagnostics','Color','w');
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    nexttile;
    if ~isempty(sigmaShapeMat)
        heatmap(sigmaBucketLabels, dateLabels, sigmaShapeMat);
        title('Short-tenor \sigma shape');
    end

    nexttile;
    if ~isempty(etaShapeMat)
        heatmap(etaBucketLabels, dateLabels, etaShapeMat);
        title('Long-tenor \eta shape');
    end

    nexttile;
    tmp = bootstrapTestResults(:, ...
        {'Date','ShapeName','TenorRole','MeanError','MaxError','BootstrapOK'});
    disp('BOOTSTRAP FIT TABLE FOR VISUAL READING');
    disp(tmp);
    axis off;
    text(0, 0.8, 'See bootstrapTestResults table', 'FontSize', 12);

    nexttile;
    if maxLenSigma >= 2
        heatmap(sigmaBucketJumpLabels, dateLabels, sigmaBucketJumpMat);
        title('Short \sigma bucket log-jumps');
    else
        axis off;
        text(0,0.5,'Not enough sigma buckets','FontSize',12);
    end

    nexttile;
    if maxLenEta >= 2
        heatmap(etaBucketJumpLabels, dateLabels, etaBucketJumpMat);
        title('Long \eta bucket log-jumps');
    else
        axis off;
        text(0,0.5,'Not enough eta buckets','FontSize',12);
    end

    nexttile;
    hold on;

    idxSigma = bootstrapTestResults.ShapeName == "sigma";
    idxEta   = bootstrapTestResults.ShapeName == "eta";

    scatter( ...
        bootstrapTestResults.MaxBucketLogJump(idxSigma), ...
        bootstrapTestResults.MeanError(idxSigma), ...
        70, 'filled');

    scatter( ...
        bootstrapTestResults.MaxBucketLogJump(idxEta), ...
        bootstrapTestResults.MeanError(idxEta), ...
        70, 'filled');

    xline(bucketJumpWarningLevel, '--');
    grid on;
    xlabel('Max bucket log-jump');
    ylabel('Mean normalized error');
    title('Bootstrap fit vs stability');
    legend({'short \sigma','long \eta','jump threshold'}, 'Location','best');

    hold off;

end

%% ========================================================
%  7. SHAPE PATHS ACROSS DATES
%  ========================================================

if includePlots

    if maxLenSigma > 0
        figure('Name','Short Sigma Shape Paths','Color','w');
        plot(1:maxLenSigma, sigmaShapeMat.', '-o', 'LineWidth', 1.1);
        grid on;
        xlabel('Bucket index');
        ylabel('\sigma bucket value');
        title('Short-tenor \sigma bootstrap shapes by date');
        legend(dateLabels, 'Location','bestoutside');
    end

    if maxLenEta > 0
        figure('Name','Long Eta Shape Paths','Color','w');
        plot(1:maxLenEta, etaShapeMat.', '-o', 'LineWidth', 1.1);
        grid on;
        xlabel('Bucket index');
        ylabel('\eta bucket value');
        title('Long-tenor \eta bootstrap shapes by date');
        legend(dateLabels, 'Location','bestoutside');
    end

end

%% ========================================================
%  8. DATE-TO-DATE STABILITY HEATMAPS
%  ========================================================

if includePlots && nDates >= 2

    figure('Name','Date-to-Date Bootstrap Stability','Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile;
    if maxLenSigma > 0
        heatmap(sigmaBucketLabels, dateJumpLabels, sigmaDateJumpMat);
        title('Date-to-date log-jumps: short \sigma');
    end

    nexttile;
    if maxLenEta > 0
        heatmap(etaBucketLabels, dateJumpLabels, etaDateJumpMat);
        title('Date-to-date log-jumps: long \eta');
    end

end

%% ========================================================
%  9. FINAL ASSERTIONS
%  ========================================================

assert(any(globalTestResults.GlobalOK), ...
    'No successful global G1++ calibration block.');

if runBootstrapBlock
    assert(any(bootstrapTestResults.ShapeOK), ...
        'No successful bootstrap shape block.');
end

disp('G1++ global and bootstrapped stability MLX completed.');
```

Two implementation notes:

1. The script **forces the interpretation** that `fixedA` is the mean reversion used for the test. Even if `calibrateGlobalG1PP` returns an `a`, the diagnostic ignores it and resets `g1Global.a = fixedA`.

2. The bootstrap fit-vs-stability plot uses

$$
x=\max_k |\Delta_k \log q_{d,k}|,
\qquad
y=\text{mean normalized pricing error},
$$

with $q=\sigma$ for the short slice and $q=\eta$ for the long slice. This is the right compact diagnostic for overfitting versus stable fit.



---
