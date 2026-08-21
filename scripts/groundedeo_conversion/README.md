# GROUNDED-EO model conversion

The original GROUNDED-EO Gaussian Process Regression (GPR) models were trained
and serialized with **scikit-learn 1.3.0**.

Scikit-learn does not guarantee compatibility of pickled/joblib models across
different scikit-learn versions. Loading these models directly with the newer
scikit-learn version used by `gee-biophys` can therefore produce compatibility
warnings and, more importantly, is not guaranteed to reproduce the original
model behavior.

To avoid requiring an old scikit-learn version in the main package, the
original models are loaded in a dedicated environment using scikit-learn 1.3.0
and converted once to **ONNX**.

The conversion script also validates the converted model by comparing both the
predicted GPR mean and standard deviation against predictions from the original
scikit-learn model.

The resulting ONNX models can then be used by `gee-biophys` without depending
on the scikit-learn version with which the original models were trained.

## Conversion

Run the conversion using the dedicated python environment with sklearn version 1.3.0:

```bash
.venv-groundedeo/bin/python \
    scripts/groundedeo_conversion/convert_to_onnx.py \
    --model gee_biophys/models/groundedeo/fapar.pkl \
    --output gee_biophys/models/groundedeo/fapar.onnx


## Original GROUNDED-EO models

The original GROUNDED-EO Gaussian Process Regression models are available from
the upstream GROUNDED-EO repository:

- FAPAR: https://github.com/luke-a-brown/grounded-eo/blob/main/models/fapar.pkl
- LAI: https://github.com/luke-a-brown/grounded-eo/blob/main/models/lai.pkl

These models were serialized with scikit-learn 1.3.0. Because scikit-learn does
not guarantee compatibility of pickled estimators across versions, the original
models are converted to ONNX using the conversion script in this directory.

The converted ONNX models are distributed with `gee-biophys` and are used for
runtime inference.
