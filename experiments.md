============================================================
TF-IDF BASELINE — TEST RESULTS
============================================================
--- label ---
              precision    recall  f1-score   support

    accurate       0.88      0.76      0.81       905
 exaggerated       0.51      0.63      0.57       241
 understated       0.21      0.45      0.29        53

    accuracy                           0.72      1199
   macro avg       0.53      0.62      0.56      1199
weighted avg       0.77      0.72      0.74      1199

--- claimed_direction ---
              precision    recall  f1-score   support

          up       0.55      0.68      0.61       229
        down       0.35      0.32      0.33        56
     neutral       0.89      0.84      0.86       914

    accuracy                           0.78      1199
   macro avg       0.59      0.61      0.60      1199
weighted avg       0.80      0.78      0.79      1199

--- actual_direction ---
              precision    recall  f1-score   support

          up       0.53      0.54      0.54       400
        down       0.63      0.60      0.62       503
     neutral       0.45      0.47      0.46       296

    accuracy                           0.55      1199
   macro avg       0.54      0.54      0.54      1199
weighted avg       0.55      0.55      0.55      1199
============================================================
BERT+AMIC (frozen) — TEST RESULTS
============================================================
--- label ---
              precision    recall  f1-score   support

    accurate       0.80      0.94      0.86       905
 exaggerated       0.59      0.34      0.43       241
 understated       0.00      0.00      0.00        53

    accuracy                           0.78      1199
   macro avg       0.46      0.43      0.43      1199
weighted avg       0.72      0.78      0.74      1199

--- claimed_direction ---
              precision    recall  f1-score   support

          up       0.73      0.45      0.55       229
        down       0.64      0.16      0.26        56
     neutral       0.84      0.96      0.89       914

    accuracy                           0.82      1199
   macro avg       0.74      0.52      0.57      1199
weighted avg       0.81      0.82      0.80      1199

--- actual_direction ---
              precision    recall  f1-score   support

          up       0.44      0.26      0.33       400
        down       0.47      0.76      0.58       503
     neutral       0.48      0.24      0.32       296

============================================================ 
ABLATION (last layer unfrozen) — TEST RESULTS - Early Stopped at Epoch 16. Best Checkpoint at Epoch 11.
============================================================
--- label ---
              precision    recall  f1-score   support

    accurate       0.81      0.96      0.88       905
 exaggerated       0.70      0.39      0.50       241
 understated       0.00      0.00      0.00        53

    accuracy                           0.80      1199
   macro avg       0.51      0.45      0.46      1199
weighted avg       0.76      0.80      0.76      1199

--- claimed_direction ---
              precision    recall  f1-score   support

          up       0.73      0.46      0.56       229
        down       0.89      0.14      0.25        56
     neutral       0.84      0.96      0.90       914

    accuracy                           0.83      1199
   macro avg       0.82      0.52      0.57      1199
weighted avg       0.82      0.83      0.80      1199
...
    accuracy                           0.48      1199
   macro avg       0.48      0.44      0.44      1199
weighted avg       0.48      0.48      0.46      1199



============================================================
CORRECT ON ALL THREE TARGETS  (8 sampled)
============================================================

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: Investors bought the dip in stocks, which were initially lower to start the new trading week on fear...
  User:  @TopStockAlerts1

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: 🚀CPI steady, AI &amp; defense stocks thrive! $INTC $AMD $LHX surge. Get 3-5 daily curated picks (tec...
  User:  @BreastPumpsDir

  Label: exaggerated  |  Claimed: up  |  Actual: up
  Tweet: 👆👇 $RKLB 🛰🛰🛰🛰🛰🛰🛰 Beyond me, though researching the many job adverts related to the above jobs is mor...
  User:  @Gykiwi03

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: 🚨Is the #Iran War ending?  An Armed Services member just sold General Dynamics.  $GD makes tanks. Wa...
  User:  @FinancePotentia

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: 🌟 Cathie Wood Loads Up on $KTOS Kratos Defense While Shares Surge on Iran News! 📈🛡️ ARK adds this au...
  User:  @CS_MarketingIR

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: $BA A defense name that is giving us a bullish engulfing candlestick on the daily timeframe to break...
  User:  @TradeLikeGates

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: $RKLB 🌏🇳🇿🇺🇲🇦🇺🇨🇦🇩🇪🇬🇧🇯🇵🇰🇷...✨️ Let's fly 🚀...
  User:  @Gykiwi03

  Label: exaggerated  |  Claimed: up  |  Actual: down
  Tweet: @ddale8 Tomahawk cruise missiles are exclusively manufactured in the United States by RTX Corporatio...
  User:  @WhoDat35
