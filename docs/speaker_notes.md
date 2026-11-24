# Presentation script and speaker notes

15 slides, about 40 seconds each: roughly 10 minutes plus the live demo.

## Slide 1: Title

Good afternoon. We predict which credit-card clients will default next month, using six months of payment history for 30,000 real clients of a Taiwanese bank (UCI, 2005). The point is not to approve or reject anyone automatically, but to rank clients so a review team sees the riskiest ones first. About 22% of clients default, so we measure ranking quality with Average Precision. Each of us will present our own part and can explain the whole pipeline.

## Slide 2: Rank clients so reviewers see likely defaulters first

The task is binary classification. Inputs are the 23 features known by the end of September 2005; the output is a probability of default in October and a review decision at a threshold chosen on validation. Success is Average Precision on clients the model never saw: it rewards putting real defaulters at the top of the list. A random ranking gets AP equal to the default rate, 0.221, so that is the bar to beat. Accuracy would be misleading: predicting "no default" for everyone is 78% accurate and useless.

## Slide 3: 30,000 real clients of one Taiwanese bank, 2005

The data is the public UCI "Default of Credit Card Clients" set (Yeh, 2009, CC BY 4.0). It has 30,000 rows, 23 predictors and no missing values. Features fall into four groups: client profile, monthly repayment status PAY_0 to PAY_6, bill amounts and payment amounts, April to September 2005. The structural audit found 56 rows that repeat another row's features, and 21 of those groups have different labels — this drives our split design. Some category codes, for example PAY = -2 and 0, are not documented by UCI, so we keep them as separate categories instead of guessing what they mean.

## Slide 4: The latest repayment status carries most of the signal

All outcome statistics here use the training set only. The chart shows the default rate by September repayment status. Clients with code 0 default 13% of the time and those who paid duly 17%; one month late raises it to 34%, two months late to 69%. The relationship is not linear, so we one-hot encode the codes. Money columns are heavily skewed, include negative bills (409 in September) and many zero payments (17%), so we use a signed log. Codes 4 to 8 have very few clients, so their rates are noisy.

## Slide 5: The test set stayed sealed until every decision was frozen

We split once with StratifiedGroupKFold: 20 folds, 14 for train, 3 for validation, 3 for test, seed 42. Rows with identical features form a group and always land in the same split, so the model cannot be tested on an exact copy of a training client. The scaler and one-hot encoder learn only from train. Validation chooses hyperparameters, the early-stopping epoch, the model and the threshold. Before opening the test set we wrote every setting and model file hash to frozen_selection.json; evaluation refuses to run if anything changed.

## Slide 6: Four model families, one shared input pipeline

Every model sees the same 23 features through the same train-fitted preprocessing. The baseline always predicts "no default". Logistic regression tried C in [0.1, 1.0, 10.0]; histogram gradient boosting tried 3 predeclared configurations; the MLP tried 4 configurations that vary width, learning rate, dropout and weight decay. The search space was written in config.json before training. Within each family and overall, the highest validation AP wins. Gradient boosting won by a small margin, 0.5555 against 0.5523 for the MLP, and that choice was frozen.

## Slide 7: A small MLP, trained with backpropagation on a GPU

The network takes the 90 preprocessed inputs, passes them through two hidden layers of 64 and 32 units with ReLU and dropout 0.2, and outputs one logit. Sigmoid turns it into a probability. Each training step: the forward pass computes logits and binary cross-entropy; loss.backward() uses the chain rule to get the gradient of the loss for every weight; optimizer.step() lets AdamW move each weight against its gradient, with weight decay 0.001. Batches of 256 are shuffled each epoch. We train up to 60 epochs and stop after 8 epochs without a better validation AP, then restore the best weights. The loss is unweighted: the threshold, not the loss, handles the class imbalance.

## Slide 8: Train AP keeps rising, validation AP flattens: mild overfitting

Both curves are measured after each epoch with dropout switched off, so they are comparable. Validation AP rises fast for about six epochs and then flattens near 0.552. Training AP keeps climbing to 0.580. That widening gap is mild overfitting: the network starts to fit noise in the training clients. Early stopping keeps the weights from epoch 17, where validation AP peaked at 0.5523, and stopped training at epoch 25. We see no underfitting: both curves are well above the 0.221 baseline. The loss curves (in artifacts/neural/learning_curves.png) tell the same story: validation BCE stops improving at about 0.426.

## Slide 9: Boosting and MLP tie on test; both far above the baseline

These are the only numbers computed on the 4,500 test clients, after everything was frozen. AP is ranking quality; the baseline equals the default rate, 0.221. Gradient boosting reaches 0.5662 and the MLP 0.5664. To check whether the gaps are real, we resampled the test clients 1,000 times and scored both models on the same resample. Boosting minus MLP is -0.0004 with interval [-0.013, +0.012]: it contains zero, so we cannot tell them apart — which is not the same as proving they are identical. Boosting beats logistic regression by +0.011, but that interval [-0.003, +0.026] also contains zero. The neural network did not need to win; it matches the best classical model.

## Slide 10: A 0.34 threshold catches 138 more defaulters than 0.5

A probability is not a decision. With the usual 0.5 cut-off the selected model flags 586 clients and misses 604 of 995 defaulters, recall 39%. We chose the threshold that maximises F1 on validation, 0.339, and applied it unchanged to test. Recall rises to 53% and missed defaults fall to 466, but false alarms grow from 195 to 392. F1 is a teaching choice: a real bank would set the threshold from its review capacity and the cost of a missed default versus an unnecessary review.

## Slide 11: PAY_0 dominates; errors come from what the data cannot see

Permutation importance shuffles one column at a time on the test set and measures how much AP drops. Shuffling PAY_0, the September repayment status, drops AP by 0.199; every other feature drops it by less than 0.010. Monthly features are correlated, so they can stand in for each other and share importance. This is predictive reliance, not causation. Looking at single clients: client 904 scored 0.74 because of a recent delay but did not default. Client 27520 scored only 0.08 and defaulted — nothing in the six months of history signalled it. Events like job loss are simply not in the data.

## Slide 12: What this result does not show

We want to be precise about the claim. The test clients come from the same bank and the same months as the training clients, so this is not a test of the future or of another bank. The scores rank well but were never calibrated, so 0.7 does not mean a 70% chance. The F1 threshold ignores real costs. The data contains sex, age and marital status, and we did not audit errors by group. Finally, one split, one seed and a small search cannot rank algorithms in general. Each limitation maps to a concrete next step on the right.

## Slide 13: Score a held-out client with one command

The demo reloads the frozen gradient boosting pipeline, including the preprocessing fitted on train, checks the SHA-256 hashes, and scores test client number 0 from raw features. The score is 0.7125, above the threshold 0.3391, so the decision is "PRIORITIZE FOR REVIEW". Only after the prediction does it print the true label, 1. Any other test client can be scored with --index, and the predictor accepts a new raw 23-column record.

## Slide 14: References and AI assistance

These are our sources: the UCI dataset and the original paper by Yeh and Lien, the Grinsztajn et al. study on why tree models are strong on tabular data, and the scikit-learn and PyTorch libraries. AI coding assistants helped with code scaffolding, documentation drafts, review and slide layout. We checked every number against the saved artifacts and each of us can explain any part of the code.

## Slide 15: Questions

Thank you. To summarise: on 4,500 unseen clients, gradient boosting and the PyTorch MLP rank defaulters equally well, about 2.6 times the chance level of AP, under a protocol where the test set was never used for any decision. We are happy to take questions; each of us can walk through the code.
