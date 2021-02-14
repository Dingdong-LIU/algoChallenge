# Changes in Trading Algorithm

## Team profile

| Team Name | AlphaBoom |
| --- | ----------- |
| University | The Hong Kong University of Science and Technology |
| Team Members | LIU, Dingdong |
| | FENG, Xinyu |
| | ZHOU, Shixu |
| | FENG, Jinglong |

## Changes in Algorithm

The algorithm still can be devided into *prediction* and *trade* modules. But some details are changed.

### Prediction Module

1. Didn't use LSTM for Forex prediction. Used random forest with gradient boosting instead.
2. Add a model selection module. A *mature model* and a *new model* will be compared and selected periodically. The mature model is the pretrained model.

*Note:*  

* The new model is the model trained while trading. The mature model is an pretrained model. The selection is based on the R2 score on historical data.
* During our experiment, we both find an suitable architecture for prediction. So a pretrained, more general model and a specific model can co-exist.

### Trading Module

1. Introduced the concept of *ideal position size* to decide the volume of a trade order.
2. Circumstance to tigger a trade, distinguish between sell and buy.

*Note:*  

* ideal_position_size = money_at_risk / cents_at_risk , where money at risk equals to the largest loss we could bear at the percentage of our available asset before the trading decision. We could not actually calculate the cents at risk under this model, so we modify it as the difference between the actual price and the predicted price of the same trading day, showing the confidence of accurate trading at the point.
* To decide whether to trade at the trading day, we used the previous day actual price, current day actual price, and future day predicted price to make comparison. If the current day actual price is the largest or smallest price among the three, we decide to sell or buy. Other occasions are when three prices are on the same trend, then we wait until the next extreme point.
