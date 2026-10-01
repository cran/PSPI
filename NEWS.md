# NEWS for PSPI

## Version 1.4 (2026-09-30) - Persistence and Transportability
- Add support for persistent model fitting (`PSPI_fit()`, `PSPI_predict()`).
- SplineBART and DSplineBART now resolve the BART/spline non-identifiability by dropping the spline intercept instead of centering BART; the previous centered implementations are removed.
- `"DSplineBART"` is accepted as a model name; `"MSplineBART"` remains an alias.
- `sim_trans()` and `sim_generalizability()`: participation model of the nonlinear scenario updated.
- Title updated to cover transportability.

## Version 1.3 (2026-01-21) - Binary Outcomes
Add support for binary outcomes.

## Version 1.2 (2025-11-1) - Bug Fixed
Fix the bug in the initialization of BART.

## Version 1.1 (2025-11-15) - Initial Release

This release introduces the **PSPI** package, which implements **Propensity Score Predictive Inference** methods for generalizing trial findings to a target population.

### Key Functionalities:
- **PSPI_generalizability**: Use Propensity Score Predictive Inference (PSPI) for generalizability analysis.
- **sim_data**: Simulate data for trial data and population data.

### License:
This package is licensed under the **GNU General Public License version 2 (GPL-2)**. The core computations leverage code derived from the **BART3** package, originally developed by Rodney Sparapani, which is also under **GPL-2**.

### Known Issues:
- No significant issues at the moment. For more details on potential limitations, please refer to the documentation.
