# PyTorch MLP

The network uses 90 transformed inputs, hidden layers 64 and 32, ReLU after each hidden layer, dropout 0.2, and one output logit. Sigmoid is applied only for inference. Trainable parameters: 7,937.

Training uses GPU (NVIDIA GeForce RTX 4050 Laptop GPU), seed 42, 2 Torch threads, unweighted BCEWithLogitsLoss, and AdamW. Learning rate 0.0005; weight decay 0.001; batch size 256. The budget is 60 epochs per candidate and early stopping after 8 epochs without improved validation AP. There are two predeclared candidates in config.json. The selected candidate ran 25 epochs and restored epoch 17. No final retraining on train+validation is done, so the validation threshold remains associated with the exact saved predictor.

Validation AP: 0.552254. Working threshold: 0.310109, chosen by maximum validation F1 with the highest-threshold tie break. These decisions never use test outcomes.

Inputs use the common preprocessing pipeline: signed log1p on monetary columns, training-fitted StandardScaler, and one-hot codes including undocumented categories kept separate. ID and target are excluded. No oversampling or class-weighted loss is used: AP evaluates ranking under the observed prevalence, while the validation threshold sets the precision/recall tradeoff. This avoids treating a weighted-loss sigmoid as an automatically calibrated probability. Even the unweighted sigmoid still requires separate calibration validation before operational use.

## Reading the learning curves

At the selected epoch, training AP is 0.573401 and validation AP is 0.552254; training loss is 0.423166 and validation loss is 0.425676. At the last epoch, training AP is 0.580146 and validation AP is 0.551845, with losses 0.420334 and 0.425317. A growing training/validation gap with no validation AP improvement is evidence of overfitting to this training sample; patience restores the earlier checkpoint. Small epoch-to-epoch AP changes are also affected by finite validation data. Both curves are measured in eval mode with dropout disabled, making their comparison consistent. This single seed and small search do not establish a universal ranking between architectures.

If the MLP trails boosting, compare the saved validation and test AP values. Trees can capture threshold effects and feature interactions efficiently on modest tabular datasets; this two-candidate MLP search may not find the best neural configuration. That is an explanation to investigate, not proof from one split.

## Backpropagation and artifact loading

The forward pass computes logits and binary cross-entropy. loss.backward() applies the chain rule through the layers to compute a gradient for each parameter. optimizer.step() uses AdamW to update those parameters and apply weight decay. zero_grad() clears old gradients before each minibatch. Validation runs without gradients and cannot update weights.

Code: src/neural.py::train_candidate. Run `python -m src.neural train`. Files include both candidate checkpoints, all epoch histories, selected.json, train-fitted preprocessor.joblib, and learning_curves.png. `load_predictor().predict_proba(raw_frame)[:, 1]` loads the exact selected checkpoint and returns positive-class scores. Saved checkpoint reload is checked against in-memory validation predictions with exact equality during training. Predictions are model probability estimates, not proven calibrated individual risks.
