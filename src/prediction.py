import joblib
from utils import get_rank
import argparse
import pandas as pd
import os


dirname = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Receipt Transaction Match")
parser.add_argument("input_file_name", type=str, help="Enter file name"
                    ". Please place the file in data directory")
parser.add_argument("output_file_name", type=str, help="Enter just the file"
                    "name; output files would be stored in the output dir")
args = parser.parse_args()

model = joblib.load(os.path.join(dirname, "../models/xgb_0818.pickle"))
x_test = pd.read_csv(os.path.join(dirname, f"../data/{args.input_file_name}"),
                     sep=':')
x_test = x_test.set_index(['receipt_id', 'company_id',
                            'matched_transaction_id', 'feature_transaction_id'])
probs = pd.DataFrame(model.predict_proba(x_test))
preds = model.predict(x_test)
test_preds_df = get_rank(probs, preds, x_test.index)
test_preds_df.to_csv(os.path.join(dirname, f"../output/{args.output_file_name}"))
