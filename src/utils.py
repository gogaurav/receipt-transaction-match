import pandas as pd


def get_rank(probs, preds, indices, y_test=None, threshold=0.5):
    test_preds_df = pd.DataFrame(probs[1].values, index=indices, columns=["probs"])
    test_preds_df["preds"] = preds
    # display(test_preds_df.head())
    grouper = test_preds_df.groupby(level=['receipt_id', 'company_id',
                                           'matched_transaction_id'])
    test_preds_df['match_rank'] = (grouper['probs'].rank('min', ascending=False)
                                   .astype(int))
    # make all the values 0 for the groups where we do not have any probability >= 0.5
    test_preds_df.loc[test_preds_df.match_rank.eq(1) 
                      & test_preds_df.probs.lt(threshold), "match_rank"] = 0
    test_preds_df["match_rank"] = (test_preds_df
                                   .groupby(level=['receipt_id', 'company_id',
                                                   'matched_transaction_id'])
                                   ['match_rank']
                                  .transform(lambda x: 0 if 0 in x.values else x))
    if y_test is not None:
        test_preds_df["actual"] = y_test
        print(test_preds_df.head())
    return test_preds_df
