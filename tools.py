# Implement the following:
# Drag your texts here
# No labels found. Would you like to begin labelling?
# Summary: This appears to be English language. There are only 1000 records. There were no labels.
# Summary: English detected. Using existing language models (Ulmfit, BERT). Best category ROC AUC were...
# Suggestion - label more data to imporve performance.
# Labeling: I have grouped the source texts into 12 groups. What labels would you give to these groups? Would you
# like more groups? Would you like less groups?
# Summary: I found 10 000 unlabelled texts. I built a language model from it. I have grouped the texts into 12 groups.

import pandas as pd


def prepare_data(source_file="./data/tagged_data.csv", min_tag_count=200, max_tag_count=800,
                 keep_tag_types=["Label"]):

    tagged_data_df = pd.read_csv(source_file, sep="\t")
    tagged_data_df = tagged_data_df[tagged_data_df["Tag_Type"].isin(keep_tag_types)]

    # Filter tags by min and max tag counts
    tag_counts = tagged_data_df["Tag_Value"].value_counts()
    filtered_tag_counts = tag_counts.where(lambda x: (min_tag_count < x) & (x < max_tag_count)).dropna()
    keep_tags = filtered_tag_counts.index.values
    tagged_data_df = tagged_data_df[tagged_data_df["Tag_Value"].isin(keep_tags)]

    # Reduce data columns. Only Keep input text and encoded tag values
    tagged_data_df["Review_Text"] = tagged_data_df["Review_Title"] + " " + tagged_data_df["Review_Detail_Body"]
    pivot_keep_cols = ["Review_Link", "Review_Text"]

    # Encode the tag values
    tagged_data_df = pd.concat([tagged_data_df[pivot_keep_cols], pd.get_dummies(tagged_data_df["Tag_Value"])], axis=1)
    tagged_data_df = tagged_data_df.groupby(pivot_keep_cols).sum().reset_index()

    col_names = list(tagged_data_df.columns)
    label_col_names = col_names[len(pivot_keep_cols):]
    # print("col_names :", col_names)
    # print("label_col_names :", label_col_names)
    # print("len(label_col_names) :", len(label_col_names))

    return tagged_data_df, label_col_names

